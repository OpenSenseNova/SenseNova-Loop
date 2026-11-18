import json

import pytest

from sensenova_loop.batch import read_requests
from sensenova_loop.cli import main, parse_args
from sensenova_loop.inference import GenerationResult

DEFAULTS = dict(max_new_tokens=16, enable_thinking=False, do_sample=False,
                temperature=0.7, top_p=0.9, top_k=None, repetition_penalty=1.0)


def test_batch_resolves_image_paths_and_overrides(tmp_path):
    image = tmp_path / "image.png"
    image.touch()
    path = tmp_path / "samples.jsonl"
    path.write_text(json.dumps({"id": "vqa", "question": "Color?", "image": image.name,
                                "max_new_tokens": 8, "think": True, "seed": 42}) + "\n")
    request, = read_requests(path, DEFAULTS)
    assert request.messages[-1]["content"][0]["image"] == str(image)
    assert request.generation["max_new_tokens"] == 8
    assert request.generation["enable_thinking"] is True
    assert request.seed == 42


@pytest.mark.parametrize("sample", [[], {"prompt": ""}, {"prompt": "Hi", "max_new_tokens": 0},
                                    {"prompt": "Hi", "do_sample": "false"},
                                    {"prompt": "Hi", "seed": True},
                                    {"messages": [{"role": "unknown", "content": "Hi"}]}])
def test_bad_batch_reports_line_before_model_loading(tmp_path, sample):
    path = tmp_path / "bad.jsonl"
    path.write_text("\n" + json.dumps(sample) + "\n")
    with pytest.raises(ValueError, match="bad.jsonl:2:"):
        read_requests(path, DEFAULTS)


def test_batch_cli_writes_jsonl_and_profiles(tmp_path, monkeypatch, capsys):
    class Engine:
        def __init__(self, *args, **kwargs):
            self.load_metrics = {"elapsed_seconds": 0.1}

        def generate_result(self, messages, **kwargs):
            return GenerationResult(messages[-1]["content"][-1]["text"], 5, 2,
                                    {"elapsed_seconds": 0.2})

    monkeypatch.setattr("sensenova_loop.inference.LoopInference", Engine)
    path = tmp_path / "samples.jsonl"
    path.write_text('{"id":1,"prompt":"你好"}\n{"id":2,"question":"Hello"}\n', encoding="utf-8")
    output = tmp_path / "results.jsonl"
    main(["--jsonl", str(path), "--output", str(output), "--profile"])
    captured = capsys.readouterr()
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [r["answer"] for r in records] == ["你好", "Hello"]
    assert [r["id"] for r in records] == [1, 2]
    assert output.read_text() == captured.out
    assert len(json.loads(captured.err)["requests"]) == 2


def test_u1_option_aliases():
    args = parse_args(["--model_path", "checkpoint", "--question", "Hello", "--max_new_tokens", "8",
                       "--do_sample", "--top_p", "0.8", "--top_k", "10", "--repetition_penalty", "1.1"])
    assert args.model_path == "checkpoint" and args.max_new_tokens == 8
    assert args.top_k == 10 and args.repetition_penalty == 1.1


def test_failed_generation_preserves_existing_output(tmp_path, monkeypatch):
    class Engine:
        def __init__(self, *args, **kwargs):
            pass

        def generate_result(self, *args, **kwargs):
            raise RuntimeError("generation failed")

    monkeypatch.setattr("sensenova_loop.inference.LoopInference", Engine)
    output = tmp_path / "answer.txt"
    output.write_text("previous answer", encoding="utf-8")
    with pytest.raises(RuntimeError, match="generation failed"):
        main(["--prompt", "Hello", "--output", str(output)])
    assert output.read_text() == "previous answer"


@pytest.mark.parametrize("override,expected", [(None, "global"), ("record", "record")])
def test_history_system_prompt_precedence(tmp_path, override, expected):
    sample = {"messages": [{"role": "system", "content": "original"},
                           {"role": "user", "content": "Hello"}]}
    if override is not None:
        sample["system_prompt"] = override
    path = tmp_path / "history.jsonl"
    path.write_text(json.dumps(sample) + "\n")
    request, = read_requests(path, DEFAULTS, system_prompt="global")
    assert request.messages[0] == {"role": "system", "content": expected}
    assert len(request.messages) == 2
    assert request.source["messages"][0]["content"] == "original"
