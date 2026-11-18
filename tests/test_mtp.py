"""MTP acceptance, hybrid rollback, checkpoint loading and public entry points."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import GenerationConfig, SynthIDTextWatermarkingConfig

from sensenova_loop.cli import parse_args
from sensenova_loop.models.modeling_sensenova_loop import SenseNovaLoopForConditionalGeneration
from sensenova_loop.mtp import verify_candidate
from sensenova_loop.speculative_cache import CacheCheckpoint
from test_model import tiny_config


def make_model(hybrid=False):
    torch.manual_seed(17)
    config = tiny_config()
    config.text_config.mtp_num_hidden_layers = 1
    config.text_config.mtp_use_dedicated_embeddings = False
    if hybrid:
        config.text_config.layer_types = ["linear_attention"] * 3 + ["full_attention"]
        config.text_config.linear_num_key_heads = config.text_config.linear_num_value_heads = 2
        config.text_config.linear_key_head_dim = config.text_config.linear_value_head_dim = 16
    model = SenseNovaLoopForConditionalGeneration(config).eval()
    model.generation_config.eos_token_id = None
    return model


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("budget,drafts", [(1, 3), (2, 3), (8, 1), (9, 3)])
def test_greedy_mtp_matches_autoregressive_tokens(hybrid, budget, drafts):
    model = make_model(hybrid)
    inputs = torch.tensor([[2, 3, 4]])
    expected = model.generate(input_ids=inputs, max_new_tokens=budget, do_sample=False,
                              temperature=None, top_p=None, top_k=None, repetition_penalty=1.1)
    actual = model.generate_mtp(input_ids=inputs, max_new_tokens=budget,
                                mtp_tokens=drafts, repetition_penalty=1.1)
    assert torch.equal(actual.sequences, expected)
    assert actual.sequences.shape[1] == inputs.shape[1] + budget
    assert actual.metrics["accepted_tokens"] <= actual.metrics["draft_tokens"]


def test_rejection_replays_accepted_prefix_with_independent_recurrent_slots(monkeypatch):
    model = make_model(hybrid=True)
    inputs = torch.tensor([[2, 3, 4]])
    expected = model.generate(input_ids=inputs, max_new_tokens=7, do_sample=False,
                              temperature=None, top_p=None, top_k=None)
    monkeypatch.setattr("sensenova_loop.mtp.verify_candidate", lambda p, q, token, **kw:
                        (False, p.argmax(-1, keepdim=True)))
    result = model.generate_mtp(input_ids=inputs, max_new_tokens=7, mtp_tokens=3)
    assert torch.equal(result.sequences, expected)
    assert result.metrics["accepted_tokens"] == 0
    assert result.metrics["replay_forward_calls"] > 0


def test_all_accepted_drafts_and_eos_stop_at_target_boundary():
    model = make_model(hybrid=True)
    with torch.no_grad():
        model.lm_head.weight.zero_()
    inputs = torch.tensor([[2, 3, 4]])
    result = model.generate_mtp(input_ids=inputs, max_new_tokens=9, mtp_tokens=3)
    assert result.sequences[0, 3:].tolist() == [0] * 9
    assert result.metrics["accepted_tokens"] == result.metrics["draft_tokens"] > 0
    assert result.metrics["replay_forward_calls"] == 0
    model.generation_config.eos_token_id = [0, 127]
    model.generation_config.min_new_tokens = 2
    stopped = model.generate_mtp(input_ids=inputs, max_new_tokens=9, mtp_tokens=3)
    assert stopped.sequences[0, 3:].tolist() == [1, 1, 0]
    assert stopped.metrics["accepted_tokens"] == 2


def test_sampling_rejection_uses_positive_probability_difference(monkeypatch):
    target = torch.tensor([[0.6, 0.1, 0.3]])
    draft = torch.tensor([[0.1, 0.7, 0.2]])
    monkeypatch.setattr(torch, "rand", lambda *a, **kw: torch.tensor(0.9))
    observed = []

    def sample(weights, count):
        observed.append(weights / weights.sum())
        return torch.tensor([[0]])

    monkeypatch.setattr(torch, "multinomial", sample)
    keep, correction = verify_candidate(target, draft, torch.tensor([[1]]), do_sample=True)
    assert keep is False and correction.item() == 0
    torch.testing.assert_close(observed[0], torch.tensor([[5 / 6, 0, 1 / 6]]))
    # The unconditional mass is accepted mass plus rejected mass redistributed
    # by that residual. This independently recovers the specified target p.
    accepted_mass = torch.minimum(target, draft)
    recovered = accepted_mass + (1 - accepted_mass.sum()) * observed[0]
    torch.testing.assert_close(recovered, target)


def test_sampling_accepts_when_target_probability_exceeds_draft(monkeypatch):
    monkeypatch.setattr(torch, "rand", lambda *a, **kw: torch.tensor(0.999))
    token = torch.tensor([[0]])
    keep, actual = verify_candidate(torch.tensor([[0.8, 0.2]]), torch.tensor([[0.1, 0.9]]),
                                     token, do_sample=True)
    assert keep and torch.equal(actual, token)


def test_checkpoint_restore_includes_convolution_and_recurrent_values():
    model = make_model(hybrid=True)
    with torch.inference_mode():
        cache = model(input_ids=torch.tensor([[2, 3]]), use_cache=True).past_key_values
        checkpoint = CacheCheckpoint(cache)
        model(input_ids=torch.tensor([[80, 81, 82]]), past_key_values=cache, use_cache=True)
        checkpoint.restore()
        assert cache.get_seq_length() == 2
        for layer, states in checkpoint.recurrent:
            for name, values in states.items():
                for key, value in values.items():
                    torch.testing.assert_close(getattr(layer, name)[key], value, rtol=0, atol=0)
        actual = model(input_ids=torch.tensor([[4]]), past_key_values=cache, use_cache=True)
        fresh = model(input_ids=torch.tensor([[2, 3, 4]]), use_cache=False)
    torch.testing.assert_close(actual.logits[:, -1], fresh.logits[:, -1], atol=1e-5, rtol=1e-4)
    assert len({id(layer) for layer in cache.layers}) == 8


def test_mtp_weights_roundtrip_and_missing_head_is_not_used(tmp_path):
    model = make_model()
    model.save_pretrained(tmp_path)
    loaded, info = SenseNovaLoopForConditionalGeneration.from_pretrained(
        tmp_path, local_files_only=True, output_loading_info=True)
    assert not info["missing_keys"] and not loaded._mtp_missing_keys
    for name, parameter in model.mtp.state_dict().items():
        torch.testing.assert_close(loaded.mtp.state_dict()[name], parameter, rtol=0, atol=0)
    path = tmp_path / "model.safetensors"
    tensors = load_file(path)
    save_file({name: value for name, value in tensors.items() if name.startswith("mtp.")},
               tmp_path / "mtp.safetensors")
    del tensors["mtp.fc.weight"]
    save_file(tensors, path, metadata={"format": "pt"})
    missing = SenseNovaLoopForConditionalGeneration.from_pretrained(tmp_path, local_files_only=True)
    with pytest.raises(ValueError, match="missing MTP weights"):
        missing.generate_mtp(input_ids=torch.tensor([[2, 3]]), max_new_tokens=2)


@pytest.mark.parametrize("drafts", [-1, True, 1.5])
def test_bad_mtp_budget_is_rejected_before_inference(drafts):
    with pytest.raises(ValueError, match="mtp_tokens"):
        make_model().generate_mtp(input_ids=torch.tensor([[2, 3]]), mtp_tokens=drafts)


def test_cli_and_jsonl_mtp_options(tmp_path):
    args = parse_args(["--prompt", "Hello", "--mtp-tokens", "3"])
    assert args.requests[0].generation["mtp_tokens"] == 3
    path = tmp_path / "requests.jsonl"
    path.write_text(json.dumps({"prompt": "Hello", "mtp_tokens": 2}) + "\n")
    args = parse_args(["--jsonl", str(path), "--mtp-tokens", "3"])
    assert args.requests[0].generation["mtp_tokens"] == 2
    path.write_text(json.dumps({"prompt": "Hello", "mtp_tokens": -1}) + "\n")
    with pytest.raises(SystemExit):
        parse_args(["--jsonl", str(path)])


def test_missing_configuration_and_padded_batches_are_rejected():
    plain = SenseNovaLoopForConditionalGeneration(tiny_config()).eval()
    with pytest.raises(ValueError, match="does not configure an MTP head"):
        plain.generate_mtp(input_ids=torch.tensor([[2, 3]]))
    model = make_model()
    with pytest.raises(ValueError, match="single nonempty conversation"):
        model.generate_mtp(input_ids=torch.tensor([[2, 3], [4, 5]]))
    with pytest.raises(ValueError, match="unpadded"):
        model.generate_mtp(input_ids=torch.tensor([[0, 3]]), attention_mask=torch.tensor([[0, 1]]))


@pytest.mark.parametrize("count,dedicated", [(2, False), (1, True)])
def test_unsupported_optional_mtp_configuration_preserves_ar_loading(tmp_path, count, dedicated):
    model = make_model()
    ids = torch.tensor([[2, 3, 4]])
    expected = model.generate(input_ids=ids, max_new_tokens=3, do_sample=False)
    model.config.text_config.mtp_num_hidden_layers = count
    model.config.text_config.mtp_use_dedicated_embeddings = dedicated
    model.save_pretrained(tmp_path)
    loaded = SenseNovaLoopForConditionalGeneration.from_pretrained(tmp_path, local_files_only=True)
    actual = loaded.generate(input_ids=ids, max_new_tokens=3, do_sample=False)
    assert torch.equal(actual, expected)
    assert loaded.mtp is None
    with pytest.raises(ValueError, match="Loop MTP .*requires"):
        loaded.generate_mtp(input_ids=ids, max_new_tokens=3)


@pytest.mark.parametrize("options", [
    {"num_beams": 2}, {"num_return_sequences": 2}, {"stop_strings": "stop"},
    {"guidance_scale": 2}, {"generation_config": GenerationConfig()},
    {"unknown_option": 1}, {"use_cache": False},
])
def test_direct_model_api_rejects_unsupported_arguments_before_prefill(options, monkeypatch):
    model = make_model()
    monkeypatch.setattr(model.model, "forward", lambda **kwargs: pytest.fail("Unexpected prefill"))
    with pytest.raises(ValueError, match="Unsupported MTP arguments"):
        model.generate_mtp(input_ids=torch.tensor([[2, 3, 4]]), max_new_tokens=2, **options)


@pytest.mark.parametrize("offset", [0, 11])
def test_explicit_two_dimensional_positions_match_three_axis_mtp_and_ar(offset):
    model = make_model(hybrid=True)
    ids = torch.tensor([[2, 3, 4]])
    positions = torch.arange(offset, offset + 3).unsqueeze(0)
    expected = model.generate(input_ids=ids, position_ids=positions, max_new_tokens=5, do_sample=False)
    actual = model.generate_mtp(input_ids=ids, position_ids=positions, max_new_tokens=5)
    expanded = model.generate_mtp(input_ids=ids, position_ids=positions.unsqueeze(0).expand(3, -1, -1),
                                  max_new_tokens=5)
    assert torch.equal(actual.sequences, expected)
    assert torch.equal(actual.sequences, expanded.sequences)


@pytest.mark.parametrize("positions", [
    torch.tensor([[100, 1, 2]]),
    torch.tensor([[[100, 1, 2]], [[200, 11, 12]], [[300, 21, 22]]]),
    torch.tensor([[[0, 1, 2]], [[100, 3, 4]], [[200, 13, 14]], [[300, 23, 24]]]),
])
def test_explicit_positions_follow_ar_each_axis_after_the_last_prompt_token(positions):
    model = make_model(hybrid=True)
    ids = torch.tensor([[2, 3, 4]])
    captured = []

    def capture(_module, _args, kwargs):
        captured.append(kwargs["position_ids"].clone())

    hook = model.model.language_model.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        expected = model.generate(input_ids=ids, position_ids=positions, max_new_tokens=5, do_sample=False)
        # Capture the actual AR continuation rather than deriving its position
        # from the implementation under test. Nonmonotonic/unequal axes expose
        # accidental global-max or scalar-position continuation.
        next_ar = captured[1]
        if next_ar.ndim == 2:
            next_ar = next_ar.unsqueeze(0).expand(4, -1, -1)
        captured.clear()
        actual = model.generate_mtp(input_ids=ids, position_ids=positions, max_new_tokens=5)
        first_verified = captured[1]
        torch.testing.assert_close(first_verified[..., :1], next_ar, rtol=0, atol=0)
        assert torch.equal(actual.sequences, expected)
        for block in captured[1:]:
            # Axes retain their own offsets across proposals and any replay.
            offsets = (next_ar - next_ar[:1]).expand_as(block)
            torch.testing.assert_close(block - block[:1], offsets, rtol=0, atol=0)
    finally:
        hook.remove()


@pytest.mark.parametrize("field,value", [
    ("watermarking_config", SynthIDTextWatermarkingConfig(ngram_len=3, keys=[1, 2])),
    ("token_healing", True), ("max_time", 1.0),
])
def test_unsupported_generation_config_is_rejected_before_prefill(field, value, monkeypatch):
    model = make_model()
    setattr(model.generation_config, field, value)
    monkeypatch.setattr(model.model, "forward", lambda **kwargs: pytest.fail("Unexpected prefill"))
    with pytest.raises(ValueError, match="MTP does not support"):
        model.generate_mtp(input_ids=torch.tensor([[2, 3, 4]]), max_new_tokens=3, do_sample=True)
