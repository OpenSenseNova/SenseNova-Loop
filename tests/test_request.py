import subprocess
import sys

import pytest

from sensenova_loop.cli import parse_args
from sensenova_loop.request import build_messages, validate_generation_options


def test_image_order_and_system_prompt(tmp_path):
    first, second = tmp_path / "first.png", tmp_path / "second.png"
    first.touch()
    second.touch()
    messages = build_messages("Compare these images.", [first, second], "Be concise.")
    assert messages[0] == {"role": "system", "content": "Be concise."}
    assert messages[1]["content"] == [
        {"type": "image", "image": str(first)},
        {"type": "image", "image": str(second)},
        {"type": "text", "text": "Compare these images."},
    ]


@pytest.mark.parametrize("argv", [
    ["--prompt", " "],
    ["--prompt", "Hello", "--max-new-tokens", "0"],
    ["--prompt", "Hello", "--do-sample", "--temperature", "nan"],
    ["--prompt", "Hello", "--do-sample", "--top-p", "1.1"],
    ["--prompt", "Hello", "--image", "/nonexistent/sensenova-loop-test.png"],
])
def test_invalid_cli_requests_fail_before_loading_a_checkpoint(argv):
    with pytest.raises(SystemExit) as error:
        parse_args(argv)
    assert error.value.code == 2


def test_help_does_not_import_model_dependencies():
    result = subprocess.run(
        [sys.executable, "-c", "from sensenova_loop.cli import parse_args; "
         "import sys; "
         "assert 'torch' not in sys.modules; "
         "parse_args(['--help'])"],
        capture_output=True, text=True, check=True,
    )
    assert "--image" in result.stdout


@pytest.mark.parametrize("tokens", [0, -1, True, 1.5])
def test_token_budget_must_be_a_positive_integer(tokens):
    with pytest.raises(ValueError, match="positive integer"):
        validate_generation_options(tokens, False, 0.7, 0.9)
