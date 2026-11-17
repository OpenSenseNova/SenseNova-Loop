"""Opt-in checkpoint tests; only the primary process should run these on a GPU."""

import os

import pytest
import torch
from PIL import Image

from sensenova_loop.inference import LoopInference, build_messages

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def engine():
    model_path = os.environ.get("SENSENOVA_LOOP_MODEL")
    if not model_path:
        pytest.skip("Set SENSENOVA_LOOP_MODEL to opt in to checkpoint tests")
    if not torch.cuda.is_available():
        pytest.fail("Checkpoint tests require a CUDA GPU")
    return LoopInference(model_path, local_files_only=True)


def test_text_answer_and_independent_requests(engine):
    messages = build_messages("What is 13 times 7? Reply with the number only.")
    assert engine.generate(messages, max_new_tokens=16).strip() == "91"
    assert engine.generate(messages, max_new_tokens=16).strip() == "91"


def test_multi_turn_history(engine):
    messages = [
        {"role": "user", "content": "Remember this code: 7319."},
        {"role": "assistant", "content": "I will remember the code."},
        {"role": "user", "content": "What code did I give you? Reply with the code only."},
    ]
    assert engine.generate(messages, max_new_tokens=16).strip() == "7319"


def test_multiple_images(engine, tmp_path):
    paths = []
    for color in ("red", "blue"):
        path = tmp_path / f"{color}.png"
        Image.new("RGB", (256, 256), color).save(path)
        paths.append(path)
    messages = build_messages(
        "Name the solid color of the first image, then the second image. Reply in English.", paths
    )
    response = engine.generate(messages, max_new_tokens=32).lower()
    assert "red" in response and "blue" in response
    assert response.index("red") < response.index("blue")


def test_routing_precision_and_cache_slots(engine):
    from transformers import DynamicCache

    fp32_parameters = [param for name, param in engine.model.named_parameters()
                       if ".mhc_attn." in name or ".mhc_ffn." in name
                       or name.endswith((".A_log", ".dt_bias"))]
    assert len(fp32_parameters) == 144
    assert all(param.dtype == torch.float32 for param in fp32_parameters)
    inputs = engine.processor.apply_chat_template(
        build_messages("Hello"), add_generation_prompt=True, enable_thinking=False,
        tokenize=True, return_dict=True, return_tensors="pt",
    ).to(engine.device)
    with torch.inference_mode():
        output = engine.model(**inputs, use_cache=True, logits_to_keep=1)
        foreign = engine.model(**inputs, past_key_values=DynamicCache(), use_cache=True, logits_to_keep=1)
    assert len(output.past_key_values.layers) == 56
    assert output.past_key_values.get_seq_length() == inputs["input_ids"].shape[1]
    assert len(foreign.past_key_values.layers) == 56
    torch.testing.assert_close(output.logits, foreign.logits)


def test_repository_model_matches_checkpoint_implementation(engine):
    import json
    from pathlib import Path

    from safetensors import safe_open
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    path = os.environ["SENSENOVA_LOOP_MODEL"]
    reference_class = get_class_from_dynamic_module(
        "modeling_sensenova_loop.SenseNovaLoopForConditionalGeneration",
        os.environ.get("SENSENOVA_LOOP_REFERENCE_SOURCE", path), local_files_only=True,
    )
    assert reference_class is not type(engine.model)
    assert reference_class.__module__.startswith("transformers_modules.")
    reference = reference_class.from_pretrained(
        path, config=reference_class.config_class.from_pretrained(path, local_files_only=True),
        local_files_only=True,
        dtype=torch.bfloat16, device_map={"": str(engine.device)}, attn_implementation="sdpa",
    ).eval()
    # The independent implementation's default BF16 loader also rounds these tensors.
    # Restore its reference values directly from the checkpoint, without using native metadata.
    index = json.loads((Path(path) / "model.safetensors.index.json").read_text())["weight_map"]
    for name, parameter in reference.named_parameters():
        if ".mhc_attn." in name or ".mhc_ffn." in name or name.endswith((".A_log", ".dt_bias")):
            with safe_open(str(Path(path) / index[name]), framework="pt", device="cpu") as source:
                saved = source.get_tensor(name)
            assert saved.dtype == torch.float32
            parameter.data = saved.to(engine.device)
    inputs = engine.processor.apply_chat_template(
        build_messages("What is 13 times 7? Reply with the number only."),
        add_generation_prompt=True, enable_thinking=False, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(engine.device)
    with torch.inference_mode():
        actual_logits = engine.model(**inputs, use_cache=True, logits_to_keep=4).logits
        expected_logits = reference(**inputs, use_cache=True, logits_to_keep=4).logits
        torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
        actual = engine.model.generate(**inputs, max_new_tokens=8, do_sample=False,
                                       temperature=None, top_p=None, top_k=None)
        expected = reference.generate(**inputs, max_new_tokens=8, do_sample=False,
                                      temperature=None, top_p=None, top_k=None)
    assert torch.equal(actual, expected)
    del reference


@pytest.mark.parametrize("mode", ["balanced", "balanced-cpu", "low"])
def test_cpu_offload_text_and_vision(engine, mode, tmp_path):
    offloaded = LoopInference(os.environ["SENSENOVA_LOOP_MODEL"], local_files_only=True,
                              vram_mode="balanced" if mode.startswith("balanced") else mode,
                              max_memory_gib={"balanced": 8, "balanced-cpu": 0.001}.get(mode))
    response = offloaded.generate(build_messages("What is 13 times 7? Reply with the number only."),
                                  max_new_tokens=4)
    assert response.strip() == "91"
    path = tmp_path / "red.png"
    Image.new("RGB", (256, 256), "red").save(path)
    result = offloaded.generate_result(build_messages("Name the solid color. One English word only.", [path]),
                                       max_new_tokens=8, profile=True)
    assert "red" in result.text.lower()
    assert result.metrics["elapsed_seconds"] > 0
    assert result.metrics["peak_allocated_gib"] > 0
    del offloaded
