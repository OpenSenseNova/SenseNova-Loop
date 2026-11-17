"""Input construction and generation option validation, before model execution."""

import math
from pathlib import Path


def build_messages(prompt: str, images=(), system_prompt: str | None = None) -> list[dict]:
    """Build one user turn from text and local images, in their supplied order."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    content = []
    for image in images:
        path = Path(image).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"Image is not a file: {path}")
        content.append({"type": "image", "image": str(path)})
    content.append({"type": "text", "text": prompt})
    messages = with_system_prompt([{"role": "user", "content": content}], system_prompt)
    validate_messages(messages)
    return messages


def with_system_prompt(messages, system_prompt):
    """Apply an explicit instruction, replacing a leading system turn if present."""
    if system_prompt is None:
        return messages
    if not isinstance(system_prompt, str):
        raise ValueError("system_prompt must be a string")
    history = messages[1:] if messages and messages[0].get("role") == "system" else messages
    return [{"role": "system", "content": system_prompt}, *history]


def validate_messages(messages):
    """Validate the shared, model-independent conversation container contract."""
    if not isinstance(messages, list) or not messages or any(
        not isinstance(message, dict) for message in messages
    ):
        raise ValueError("messages must be a non-empty list of message dictionaries")
    for message in messages:
        if message.get("role") not in ("system", "user", "assistant", "tool"):
            raise ValueError("Unsupported or missing message role")
        if not isinstance(message.get("content"), (str, list)):
            raise ValueError("Message content must be text or a list of content blocks")


def validate_generation_options(max_new_tokens, do_sample, temperature, top_p,
                                top_k=None, repetition_penalty=1.0):
    """Validate the common generation contract for the CLI and Python API."""
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if do_sample and (
        not math.isfinite(temperature) or temperature <= 0
        or not math.isfinite(top_p) or not 0 < top_p <= 1
    ):
        raise ValueError("Sampling requires finite temperature > 0 and 0 < top_p <= 1")
    if top_k is not None and (type(top_k) is not int or top_k < 0):
        raise ValueError("top_k must be a nonnegative integer or None")
    if not math.isfinite(repetition_penalty) or repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be finite and positive")
