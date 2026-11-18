"""Command line interface for text generation and visual question answering."""

import argparse
import json
import sys
from pathlib import Path

from .batch import BatchRequest, read_requests
from .request import build_messages, validate_generation_options


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", "--model_path", default="sensenova/SenseNova-Loop-9B-Preview")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", "--question")
    source.add_argument("--jsonl", type=Path, help="Text or VQA requests; image paths relative to this file")
    parser.add_argument("--image", action="append", default=[], help="Local image; repeat for multiple images")
    parser.add_argument("--system-prompt")
    parser.add_argument("--device", default="cuda:0", help="Device within CUDA_VISIBLE_DEVICES")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--attn-backend", "--attn_backend", choices=["sdpa", "eager"], default="sdpa")
    parser.add_argument("--vram-mode", "--vram_mode", choices=["full", "balanced", "low"], default="full")
    parser.add_argument("--max-memory-gib", type=float, help="GPU weight budget for balanced mode")
    parser.add_argument("--max-new-tokens", "--max_new_tokens", type=int, default=256)
    parser.add_argument("--mtp-tokens", "--mtp_tokens", type=int, default=0,
                        help="MTP draft tokens per verification round; 0 disables MTP")
    parser.add_argument("--think", action="store_true", help="Include the model's thinking in the output")
    parser.add_argument("--do-sample", "--do_sample", action="store_true", help="Enable sampling; default is greedy")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", "--top_p", type=float, default=0.9)
    parser.add_argument("--top-k", "--top_k", type=int)
    parser.add_argument("--repetition-penalty", "--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, help="Response file; JSONL for batch input")
    parser.add_argument("--output-dir", "--output_dir", type=Path, help="Batch directory; writes answers.jsonl")
    parser.add_argument("--output-format", choices=["text", "json"], default="text", help="Single-request stdout/file format")
    parser.add_argument("--profile", action="store_true", help="Synchronized load/generation time and CUDA peak memory")
    args = parser.parse_args(argv)
    args.generation = dict(max_new_tokens=args.max_new_tokens, enable_thinking=args.think,
                           do_sample=args.do_sample, temperature=args.temperature, top_p=args.top_p,
                           top_k=args.top_k, repetition_penalty=args.repetition_penalty,
                           mtp_tokens=args.mtp_tokens)
    try:
        validate_generation_options(**{k: v for k, v in args.generation.items() if k != "enable_thinking"})
        if args.output is not None and args.output_dir is not None:
            raise ValueError("Use only one of --output and --output-dir")
        if args.jsonl is not None:
            if args.image:
                raise ValueError("Put image paths in JSONL records when using --jsonl")
            args.requests = read_requests(args.jsonl, args.generation, seed=args.seed,
                                          system_prompt=args.system_prompt)
        else:
            if args.output_dir is not None:
                raise ValueError("--output-dir requires --jsonl")
            messages = build_messages(args.prompt, args.image, args.system_prompt)
            args.requests = [BatchRequest({}, messages, args.generation, args.seed)]
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return args


def main(argv=None):
    args = parse_args(argv)
    import torch

    from .inference import LoopInference

    engine = LoopInference(args.model_path, device=args.device, dtype=getattr(torch, args.dtype),
                           attn_backend=args.attn_backend, local_files_only=args.local_files_only,
                           vram_mode=args.vram_mode, max_memory_gib=args.max_memory_gib, profile=args.profile)
    output = args.output
    if args.jsonl is not None and output is None:
        output = (args.output_dir or Path("outputs")) / "answers.jsonl"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
    stream = None
    profiles = []
    try:
        for request in args.requests:
            torch.manual_seed(request.seed)
            result = engine.generate_result(request.messages, **request.generation, profile=args.profile)
            payload = dict(request.source, answer=result.text, input_tokens=result.input_tokens,
                           output_tokens=result.output_tokens, metrics=result.metrics)
            text = json.dumps(payload, ensure_ascii=False) if args.jsonl is not None or args.output_format == "json" else result.text
            if output is not None and stream is None:
                stream = output.open("w", encoding="utf-8")
            if stream is not None:
                stream.write(text + "\n")
                stream.flush()
            print(text)
            if args.profile:
                profiles.append(result.metrics)
    finally:
        if stream is not None:
            stream.close()
    if args.profile:
        print(json.dumps({"load": engine.load_metrics, "requests": profiles}), file=sys.stderr)
