from types import SimpleNamespace

import pytest
import torch
from transformers import BatchFeature

from sensenova_loop.inference import GenerationResult, LoopInference


@pytest.fixture
def engine():
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            self.template_options = kwargs
            return BatchFeature({"input_ids": torch.tensor([[11, 12, 13]]),
                                 "attention_mask": torch.ones(1, 3, dtype=torch.long)})

        def decode(self, tokens, **kwargs):
            assert kwargs == {"skip_special_tokens": True}
            return ",".join(map(str, tokens.tolist()))

    class Model:
        config = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=32))

        def generate(self, **kwargs):
            self.options = kwargs
            assert not torch.is_grad_enabled()
            return torch.tensor([[11, 12, 13, 91, 99]])

    instance = object.__new__(LoopInference)
    instance.device = torch.device("cpu")
    instance.processor = Processor()
    instance.model = Model()
    return instance


def test_only_new_tokens_are_decoded_and_greedy_clears_sampling_defaults(engine):
    assert engine.generate([{"role": "user", "content": "Hello"}], max_new_tokens=2) == "91,99"
    assert engine.model.options["do_sample"] is False
    assert engine.model.options["temperature"] is None
    assert engine.model.options["top_p"] is None
    assert engine.model.options["top_k"] is None
    assert engine.model.options["use_cache"] is True
    assert engine.processor.template_options["enable_thinking"] is False


def test_sampling_and_thinking_reach_their_respective_components(engine):
    engine.generate([{"role": "user", "content": "Hello"}], max_new_tokens=2,
                    enable_thinking=True, do_sample=True, temperature=0.5, top_p=0.8)
    assert engine.processor.template_options["enable_thinking"] is True
    assert engine.model.options["do_sample"] is True
    assert engine.model.options["temperature"] == 0.5
    assert engine.model.options["top_p"] == 0.8


def test_context_budget_includes_new_tokens(engine):
    with pytest.raises(ValueError, match="exceeds the model context"):
        engine.generate([{"role": "user", "content": "Hello"}], max_new_tokens=30)
    assert not hasattr(engine.model, "options")


@pytest.mark.parametrize("messages", [[], {}, [[{"role": "user", "content": "Hello"}]]])
def test_only_one_nonempty_conversation_is_accepted(engine, messages):
    with pytest.raises(ValueError, match="message dictionaries"):
        engine.generate(messages)


def test_cli_saves_and_prints_generated_text(tmp_path, monkeypatch, capsys):
    from sensenova_loop.cli import main

    class Engine:
        def __init__(self, model_path, **kwargs):
            assert model_path == "local-checkpoint"
            assert kwargs["local_files_only"] is True

        def generate_result(self, messages, **kwargs):
            assert messages[-1]["content"][-1]["text"] == "Hello"
            assert kwargs["max_new_tokens"] == 8
            return GenerationResult("你好", 3, 2, {})

    monkeypatch.setattr("sensenova_loop.inference.LoopInference", Engine)
    output = tmp_path / "nested" / "answer.txt"
    main(["--model-path", "local-checkpoint", "--local-files-only", "--prompt", "Hello",
          "--max-new-tokens", "8", "--output", str(output)])
    assert output.read_text(encoding="utf-8") == "你好\n"
    assert capsys.readouterr().out == "你好\n"


def test_structured_result_counts_and_cpu_profiling(engine):
    result = engine.generate_result([{"role": "user", "content": "Hello"}],
                                    max_new_tokens=2, profile=True)
    assert result.text == "91,99"
    assert (result.input_tokens, result.output_tokens) == (3, 2)
    assert result.metrics["elapsed_seconds"] > 0
    assert "peak_allocated_gib" not in result.metrics
