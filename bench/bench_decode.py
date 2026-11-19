"""Measure mHC sinkhorn decode overhead: reference vs shipped vs compiled.

The mHC routers call sinkhorn once per router, 96 times per decode token
(2 routers x 24 looped layers x 2 visits), and each call runs 20 alternating
row/column normalizations built from 40 aten::logsumexp reductions over
two-element tensors. This script reproduces the decode measurement approach
behind bench/decode_overhead.md:

  1. load the Preview checkpoint in 4-bit (bitsandbytes) with the token
     embedding and the vision tower parked in system RAM, like the
     reference measurement on an 8 GB GPU,
  2. prefill a 2048-token passage question,
  3. teacher-force 8 greedy tokens and compare per-step logits across arms
     (the reference is bit-deterministic, so its A/A floor is 0.0),
  4. run a manual greedy decode loop and report the median time per token.

Arms:
  reference : the pre-change aten::logsumexp sinkhorn, kept inline below
  shipped   : the repository implementation (elementwise pair reduction)
  compiled  : torch.compile(reference, dynamic=False), the fusion ceiling

Usage:
  python bench/bench_decode.py --model-path /path/to/SenseNova-Loop-9B-Preview

Requires a CUDA GPU and the checkpoint's weight shards; it is not part of
the CPU test suite.
"""

import argparse
import json
import statistics
import sys
import time

import torch


def reference_sinkhorn(logits, iterations):
    """Pre-change implementation: alternating aten::logsumexp normalizations."""
    logits = logits.float()
    for _ in range(iterations):
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        logits = logits - torch.logsumexp(logits, dim=-2, keepdim=True)
    return logits.exp()


FILLER = (
    "The maintenance manual states that the pump must be primed before the "
    "valve opens, and the sensor reports pressure every second until the "
    "steady state is reached. "
)


def load_model(model_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    import sensenova_loop

    sensenova_loop.register()
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
        quantization_config=quant,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    # Park the embedding and vision tower in system RAM, as in the
    # reference measurement on an 8 GB GPU.
    model.model.language_model.embed_tokens.to("cpu")
    for name, child in model.model.named_children():
        if "vision" in type(child).__name__.lower():
            child.to("cpu")
    torch.cuda.empty_cache()
    return model, tokenizer


def build_prompt(tokenizer, ctx):
    messages = [{
        "role": "user",
        "content": "Read the passage and answer the final question.\n\n"
        + FILLER * 400
        + "\n\nQuestion: what must happen before the valve opens?",
    }]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return tokenizer(prompt, return_tensors="pt").input_ids[:, :ctx]


def forced_logits(model, embed, ids, forced):
    """Prefill plus teacher-forced steps; returns stacked per-step logits."""
    with torch.inference_mode():
        embeddings = embed(ids).to(0)
        out = model(inputs_embeds=embeddings, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1, :]
        rows = [logits.detach().clone().float().cpu()]
        for token in forced:
            step = embed(ids.new_full((1, 1), token)).to(0)
            out = model(inputs_embeds=step, past_key_values=past, use_cache=True)
            logits = out.logits[:, -1, :]
            rows.append(logits.detach().clone().float().cpu())
    return torch.stack(rows)


def greedy_decode(model, embed, ids, steps):
    """Manual greedy loop; returns (token ids, per-token milliseconds)."""
    with torch.inference_mode():
        embeddings = embed(ids).to(0)
        out = model(inputs_embeds=embeddings, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1, :]
        tokens, times = [], []
        for _ in range(steps):
            next_id = int(logits.argmax(dim=-1).item())
            tokens.append(next_id)
            torch.cuda.synchronize()
            start = time.perf_counter()
            step = embed(ids.new_full((1, 1), next_id)).to(0)
            out = model(inputs_embeds=step, past_key_values=past, use_cache=True)
            logits = out.logits[:, -1, :]
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    return tokens, times


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--forced", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--arms", default="reference,shipped,compiled")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("This benchmark requires a CUDA GPU.")

    import sensenova_loop.models.modeling_sensenova_loop as modeling

    # Capture the shipped implementation before any arm patching.
    shipped_fn = modeling.sinkhorn

    model, tokenizer = load_model(args.model_path)
    embed = model.model.language_model.embed_tokens
    ids = build_prompt(tokenizer, args.ctx)

    # Collect the forced greedy sequence under the reference implementation.
    modeling.sinkhorn = reference_sinkhorn
    with torch.inference_mode():
        embeddings = embed(ids).to(0)
        out = model(inputs_embeds=embeddings, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1, :]
        forced = []
        for _ in range(args.forced):
            next_id = int(logits.argmax(dim=-1).item())
            forced.append(next_id)
            step = embed(ids.new_full((1, 1), next_id)).to(0)
            out = model(inputs_embeds=step, past_key_values=past, use_cache=True)
            logits = out.logits[:, -1, :]

    compiled_sinkhorn = torch.compile(reference_sinkhorn, dynamic=False)
    arms = {
        "reference": reference_sinkhorn,
        "shipped": shipped_fn,
        "compiled": compiled_sinkhorn,
    }
    selected = {name: arms[name] for name in args.arms.split(",") if name in arms}

    reference_rows = forced_logits(model, embed, ids, forced)
    baseline_tokens, baseline_times = None, None
    results = {}

    for name, impl in selected.items():
        modeling.sinkhorn = impl
        rows = forced_logits(model, embed, ids, forced)
        delta = (rows - reference_rows).abs().max().item()
        agree = (rows.argmax(-1) == reference_rows.argmax(-1)).float().mean().item()
        tokens, times = greedy_decode(model, embed, ids, args.steps)
        results[name] = {
            "forced_max_logit_delta": delta,
            "forced_argmax_agreement": agree,
            "median_ms_per_token": round(statistics.median(times), 1),
            "tok_per_s": round(args.steps / (sum(times) / 1000), 2),
            "greedy_ids": tokens,
        }
        if name == "reference":
            baseline_tokens, baseline_times = tokens, times
        match = sum(a == b for a, b in zip(tokens, baseline_tokens or []))
        print(f"{name:10s} forced_max_delta={delta:.6g} argmax={agree:.3f} "
              f"median={results[name]['median_ms_per_token']} ms/token "
              f"tok/s={results[name]['tok_per_s']} greedy_match={match}/{args.steps}")

    # A/A floor of the reference itself (bit-determinism check).
    modeling.sinkhorn = reference_sinkhorn
    aa = (forced_logits(model, embed, ids, forced) - reference_rows).abs().max().item()
    print(f"reference A/A forced max delta: {aa}")

    output = {
        "model": args.model_path,
        "ctx": args.ctx,
        "forced_steps": args.forced,
        "greedy_steps": args.steps,
        "reference_aa_max_delta": aa,
        "arms": results,
    }
    path = args.out or "decode_overhead_result.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
