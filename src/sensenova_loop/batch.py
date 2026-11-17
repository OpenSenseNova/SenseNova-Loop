"""JSONL request decoding and preflight for sequential text/VQA inference."""

import json
from dataclasses import dataclass
from pathlib import Path

from .request import build_messages, validate_generation_options, validate_messages, with_system_prompt


@dataclass
class BatchRequest:
    source: dict
    messages: list[dict]
    generation: dict
    seed: int


def read_requests(path, defaults, *, seed=0, system_prompt=None):
    """Resolve image paths relative to JSONL; validate all records before loading."""
    path = Path(path).expanduser().resolve(strict=True)
    requests = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
                if not isinstance(sample, dict):
                    raise ValueError("each record must be an object")
                options = {name: sample.get(name, value) for name, value in defaults.items()}
                if "think" in sample:
                    options["enable_thinking"] = sample["think"]
                for name in ("enable_thinking", "do_sample"):
                    if type(options[name]) is not bool:
                        raise ValueError(f"{name} must be boolean")
                validate_generation_options(**{k: v for k, v in options.items() if k != "enable_thinking"})
                sample_seed = sample.get("seed", seed)
                if type(sample_seed) is not int:
                    raise ValueError("seed must be an integer")
                selected_system_prompt = sample.get("system_prompt", system_prompt)
                if "messages" in sample:
                    messages = sample["messages"]
                    validate_messages(messages)
                    # Explicit messages use text history; image inputs use image/images below.
                    if any(not isinstance(message.get("content"), str) for message in messages):
                        raise ValueError("JSONL messages require string content; use image/images for VQA")
                    if "image" in sample or "images" in sample:
                        raise ValueError("messages cannot be combined with image/images")
                    messages = with_system_prompt(messages, selected_system_prompt)
                else:
                    prompt = sample.get("prompt", sample.get("question"))
                    if not isinstance(prompt, str):
                        raise ValueError("prompt or question must be a string")
                    images = sample.get("images", sample.get("image", []))
                    if isinstance(images, str):
                        images = [images]
                    if not isinstance(images, list) or any(not isinstance(image, str) for image in images):
                        raise ValueError("image/images must be a path or list of paths")
                    paths = [path.parent / Path(image).expanduser() for image in images]
                    messages = build_messages(prompt, paths, selected_system_prompt)
                requests.append(BatchRequest(sample, messages, options, sample_seed))
            except (ValueError, TypeError, OSError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not requests:
        raise ValueError(f"{path}: no requests found")
    return requests
