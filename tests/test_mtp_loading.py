"""Separately published predictors enter the same HF loader as the backbone."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from sensenova_loop.models.modeling_sensenova_loop import SenseNovaLoopForConditionalGeneration
from test_mtp import make_model


def save_with_companion(model, path, *, sharded):
    model.save_pretrained(path)
    source = path / "model.safetensors"
    weights = load_file(source)
    predictor = {key: value for key, value in weights.items() if key.startswith("mtp.")}
    backbone = {key: value for key, value in weights.items() if not key.startswith("mtp.")}
    save_file(predictor, path / "mtp.safetensors", metadata={"format": "pt"})
    if not sharded:
        save_file(backbone, source, metadata={"format": "pt"})
        return
    source.unlink()
    entries = list(backbone.items())
    middle = len(entries) // 2
    index = {}
    for number, part in enumerate((entries[:middle], entries[middle:]), 1):
        filename = f"model-{number:05}.safetensors"
        save_file(dict(part), path / filename, metadata={"format": "pt"})
        index.update(dict.fromkeys(dict(part), filename))
    (path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(value.numel() * value.element_size() for value in backbone.values())},
        "weight_map": index,
    }))


@pytest.mark.parametrize("sharded", [False, True])
def test_unindexed_companion_loads_exact_values_in_requested_dtype(tmp_path, sharded):
    original = make_model()
    save_with_companion(original, tmp_path, sharded=sharded)
    before = {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}
    loaded, info = SenseNovaLoopForConditionalGeneration.from_pretrained(
        tmp_path, local_files_only=True, dtype=torch.bfloat16, device_map={"": "cpu"},
        output_loading_info=True,
    )
    assert not info["missing_keys"] and not info["unexpected_keys"]
    assert not loaded._mtp_missing_keys
    assert before == {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}
    for name, value in original.mtp.state_dict().items():
        actual = loaded.mtp.state_dict()[name]
        assert actual.dtype == torch.bfloat16
        torch.testing.assert_close(actual, value.to(torch.bfloat16), rtol=0, atol=0)
    ids = torch.tensor([[2, 3, 4]])
    ar = loaded.generate(input_ids=ids, max_new_tokens=5, do_sample=False)
    mtp = loaded.generate_mtp(input_ids=ids, max_new_tokens=5)
    assert torch.equal(ar, mtp.sequences)


def test_existing_checkpoint_head_takes_precedence_over_companion(tmp_path, monkeypatch):
    original = make_model()
    original.save_pretrained(tmp_path)
    save_file({"mtp.fc.weight": torch.zeros_like(original.mtp.fc.weight)}, tmp_path / "mtp.safetensors")

    def unexpected_lookup(*args, **kwargs):
        pytest.fail("An existing predictor must not resolve a second checkpoint")

    monkeypatch.setattr("sensenova_loop.models.mtp_loading.cached_file", unexpected_lookup)
    loaded = SenseNovaLoopForConditionalGeneration.from_pretrained(tmp_path, local_files_only=True)
    torch.testing.assert_close(loaded.mtp.fc.weight, original.mtp.fc.weight, rtol=0, atol=0)


@pytest.mark.parametrize("missing_file", [False, True])
def test_incomplete_or_missing_companion_cannot_generate(tmp_path, missing_file):
    save_with_companion(make_model(), tmp_path, sharded=True)
    path = tmp_path / "mtp.safetensors"
    if missing_file:
        path.unlink()
    else:
        weights = load_file(path)
        del weights["mtp.fc.weight"]
        save_file(weights, path)
    loaded = SenseNovaLoopForConditionalGeneration.from_pretrained(tmp_path, local_files_only=True)
    assert "mtp.fc.weight" in loaded._mtp_missing_keys
    with pytest.raises(ValueError, match="missing MTP weights"):
        loaded.generate_mtp(input_ids=torch.tensor([[2, 3]]), max_new_tokens=2)


def test_companion_cannot_replace_backbone_weights(tmp_path):
    original = make_model()
    save_with_companion(original, tmp_path, sharded=True)
    path = tmp_path / "mtp.safetensors"
    weights = load_file(path)
    weights["lm_head.weight"] = torch.zeros_like(original.lm_head.weight)
    save_file(weights, path)
    with pytest.raises(ValueError, match="only MTP weights"):
        SenseNovaLoopForConditionalGeneration.from_pretrained(tmp_path, local_files_only=True)


@pytest.mark.parametrize("sharded", [False, True])
def test_companion_is_available_to_real_disk_offload_hooks(tmp_path, sharded):
    save_with_companion(make_model(), tmp_path, sharded=sharded)
    resident = SenseNovaLoopForConditionalGeneration.from_pretrained(tmp_path, local_files_only=True)
    offloaded = SenseNovaLoopForConditionalGeneration.from_pretrained(
        tmp_path, local_files_only=True, device_map={"model": "cpu", "lm_head": "cpu", "mtp": "disk"},
        offload_folder=tmp_path / "offload",
    )
    ids = torch.tensor([[2, 3, 4]])
    expected = resident.generate_mtp(input_ids=ids, max_new_tokens=5).sequences
    actual = offloaded.generate_mtp(input_ids=ids, max_new_tokens=5).sequences
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("has_companion", [False, True])
def test_explicit_weight_file_preserves_hf_loading_contract(tmp_path, relative, has_companion, monkeypatch):
    original = make_model()
    save_with_companion(original, tmp_path, sharded=False)
    if not has_companion:
        (tmp_path / "mtp.safetensors").unlink()
    path = tmp_path / "model.safetensors"
    if relative:
        monkeypatch.chdir(tmp_path)
        path = Path("model.safetensors")
    # PEFT is not a project dependency. In its absence HF accepts weight-file
    # paths; installed PEFT has its own earlier adapter-probing limitation.
    monkeypatch.setattr("transformers.integrations.peft.is_peft_available", lambda: False)
    loaded = SenseNovaLoopForConditionalGeneration.from_pretrained(
        path, config=original.config, generation_config=original.generation_config, local_files_only=True,
    )
    ids = torch.tensor([[2, 3, 4]])
    expected = original.generate(input_ids=ids, max_new_tokens=3, do_sample=False)
    assert torch.equal(loaded.generate(input_ids=ids, max_new_tokens=3, do_sample=False), expected)
    if has_companion:
        torch.testing.assert_close(loaded.mtp.fc.weight, original.mtp.fc.weight, rtol=0, atol=0)
        assert torch.equal(loaded.generate_mtp(input_ids=ids, max_new_tokens=3).sequences, expected)
    else:
        assert len(loaded._mtp_missing_keys) == 15
        with pytest.raises(ValueError, match="missing MTP weights"):
            loaded.generate_mtp(input_ids=ids, max_new_tokens=3)
