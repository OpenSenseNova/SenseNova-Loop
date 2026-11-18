"""Bounded MTP checks on an opt-in local checkpoint, not an accuracy benchmark."""

import os
from pathlib import Path

import pytest
import torch
from PIL import Image
from safetensors import safe_open

from sensenova_loop.inference import LoopInference, build_messages

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def engine():
    path = os.environ.get("SENSENOVA_LOOP_MODEL")
    if not path:
        pytest.skip("Set SENSENOVA_LOOP_MODEL to opt in to checkpoint tests")
    if not torch.cuda.is_available():
        pytest.fail("Checkpoint tests require a CUDA GPU")
    return LoopInference(path, local_files_only=True)


def compare_greedy(engine, messages, budget=12):
    inputs = engine.processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
        return_tensors="pt", return_dict=True,
    ).to(engine.device)
    expected = engine.model.generate(**inputs, max_new_tokens=budget, do_sample=False,
                                      temperature=None, top_p=None, top_k=None)
    result = engine.model.generate_mtp(**inputs, max_new_tokens=budget, mtp_tokens=3)
    assert torch.equal(result.sequences, expected)
    assert result.metrics["verification_rounds"] > 0
    print("MTP greedy metrics:", result.metrics, flush=True)


def test_released_mtp_weights_are_loaded_exactly(engine):
    source = Path(os.environ["SENSENOVA_LOOP_MODEL"]) / "mtp.safetensors"
    parameters = dict(engine.model.named_parameters())
    with safe_open(source, framework="pt", device="cpu") as saved:
        assert len(list(saved.keys())) == 15
        for name in saved.keys():
            assert name in parameters
            torch.testing.assert_close(parameters[name].detach().cpu(), saved.get_tensor(name), rtol=0, atol=0)


def test_text_greedy_and_independent_calls(engine):
    for prompt in ["What is 13 times 7? Reply with the number only.",
                   "Continue this sequence: Monday, Tuesday, Wednesday,"]:
        compare_greedy(engine, build_messages(prompt))


def test_mtp_visual_and_history(engine, tmp_path):
    image = tmp_path / "red.png"
    Image.new("RGB", (128, 128), "red").save(image)
    compare_greedy(engine, build_messages("Name the solid color. One English word only.", [image]), budget=8)
    compare_greedy(engine, [{"role": "user", "content": "Remember the number 7319."},
                            {"role": "assistant", "content": "I will remember 7319."},
                            {"role": "user", "content": "Repeat the number only."}], budget=8)


def test_thinking_sampling_uses_mtp_and_honors_budget(engine):
    torch.manual_seed(42)
    result = engine.generate_result(build_messages("What is 13 times 7?"), enable_thinking=True,
                                    max_new_tokens=24, do_sample=True, temperature=1.0,
                                    top_p=0.95, mtp_tokens=3, profile=True)
    assert 0 < result.output_tokens <= 24
    assert result.metrics["mtp"]["verification_rounds"] > 0
    assert result.metrics["mtp"]["draft_tokens"] > 0
    assert result.metrics["peak_allocated_gib"] > 0
    print("MTP sampling metrics:", result.metrics, flush=True)


def test_cpu_weight_offload_hooks_preserve_tiny_mtp_tokens(engine):
    from accelerate import cpu_offload
    from test_mtp import make_model

    # Exercise real dispatch hooks without another full-size checkpoint load.
    model = make_model().to(engine.device)
    ids = torch.tensor([[2, 3, 4]], device=engine.device)
    expected = model.generate_mtp(input_ids=ids, max_new_tokens=5).sequences
    cpu_offload(model, execution_device=engine.device, offload_buffers=True)
    actual = model.generate_mtp(input_ids=ids, max_new_tokens=5).sequences
    assert torch.equal(actual, expected)
