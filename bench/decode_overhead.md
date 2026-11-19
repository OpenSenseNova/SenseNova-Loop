# mHC router decode overhead benchmark

This benchmark measures what the 20-iteration Sinkhorn normalization in the
mHC routers costs during decode, and what the elementwise pair reduction in
`_logsumexp_pair` recovers. It is the measurement behind the router
performance report and the pair-reduction change.

## Why the routers dominate decode

The Preview checkpoint's mHC block spans layers 4 through 27 (24 physical
layers) with 2 residual streams, 20 Sinkhorn iterations, and a loop count
of 2. Each of the 24 looped layer executions runs two mHC routers
(attention and FFN), and the loop executes every layer twice:

    2 routers x 24 layers x 2 visits = 96 sinkhorn calls per decode token

Each sinkhorn call performs 20 alternating row and column log-space
normalizations, that is 40 `aten::logsumexp` reductions, and every one of
those reductions runs over a two-element tensor:

    96 x 40 = 3,840 two-element logsumexp reductions per decode token

Each `aten::logsumexp` dispatch lowers to an amax / masked_fill / sub /
exp / sum / log / add composite whose two reduce kernels are far more
expensive on tiny tensors than the arithmetic they perform.

## Method

`bench_decode.py` reproduces the measurement approach of the reported
results:

1. Load the Preview checkpoint in 4-bit (bitsandbytes, double quantization,
   BF16 compute) with the token embedding and the vision tower parked in
   system RAM.
2. Prefill a 2048-token passage question.
3. Teacher-force 8 greedy tokens and compare per-step logits across arms.
   The reference implementation is bit-deterministic, so its A/A floor is
   exactly 0.0.
4. Run a manual greedy decode loop for 32 tokens with
   `torch.cuda.synchronize()` around each step and report the median
   milliseconds per token.

Arms: `reference` is the pre-change `aten::logsumexp` implementation (kept
inline in the script), `shipped` is the repository implementation, and
`compiled` wraps the reference in `torch.compile(dynamic=False)` as the
fusion ceiling.

## Reported results

Reference hardware: RTX 5060 Laptop GPU (8 GB), Windows, Preview weights at
4-bit, PyTorch 2.14.1+cu132 with Transformers 5.18.0 (the reference
measurement environment), 2048-token context, greedy decoding, median over
32 decode steps.

| Arm | Median ms/token | Throughput |
| --- | ---: | ---: |
| reference (aten::logsumexp) | 1539.0 | 0.65 tok/s |
| torch.compile(reference), dynamic=False | 158.5 | 3.77 tok/s over the full 32-token window |

The throughput column is total wall time over the 32-token window divided
by 32, so it includes the arm's slow early steps; the median column is the
per-step median and is the number to compare.

Profiler attribution over 3 decode steps of the reference arm:

- about 7,680 `aten::logsumexp` calls per decode token
- their child reduce kernels sum to roughly 1.1 s of the 1.43 s GPU-busy
  time per token
- about 43,397 kernel launches per token
- every logsumexp reduction processes a two-element tensor

CPU dispatch profile of one full sinkhorn call at the decode shape
(1, 1, 2, 2), PyTorch 2.14.1, counted with `torch.profiler`:

| Implementation | aten ops per call | logsumexp | reduce ops | median ms per call |
| --- | ---: | ---: | ---: | ---: |
| reference | 882 | 80 | 80 (amax + sum) | 0.550 |
| shipped pair reduction | 602 | 0 | 0 | 0.534 |

The shipped arm removes every reduce dispatch and cuts the aten op count
per call by about a third; CPU wall time per call moves little because
Python dispatch dominates at this size on CPU.

TODO(measurement): shipped-arm GPU decode median pending a free GPU on the
measurement machine. Run
`PYTHONPATH=src python bench/bench_decode.py --model-path <Preview dir>`
and record the shipped arm's median here and in the two upstream drafts;
until then the compile ceiling above bounds the available win.

## Why the closed form was rejected

The 2x2 doubly stochastic limit has a closed form,
`t = sigmoid((l11 + l22 - l12 - l21) / 2)`, and it is not a drop-in
replacement. Alternating normalization preserves the odds ratio, but the
20-iteration operator converges with factor
`((sqrt(r) - 1) / (sqrt(r) + 1))^2`, which approaches 1 for saturated
logits; the router bias initialization (log(15) on the mixing diagonal)
pushes toward exactly that near-identity saturated regime. The model was
trained against the 20-iteration operator, not the limit.

Measured on the same checkpoint: teacher-forcing 8 greedy tokens, the
reference is bit-deterministic (A/A max logit delta exactly 0.0), while the
closed form differs by a max logit delta of 9.84375 despite agreeing on
every greedy argmax over those steps. The elementwise pair reduction is the
path that preserves the trained operator bit-for-bit.

The log1p form of the pair identity, `m + log1p(exp(-|a - b|))`, was
measured as well and also rejected: aten rounds `1 + exp(-d)` before the
log, so the log1p shortcut differs from aten by up to 2 float32 ulp (max
output delta 2.38e-7 over the 120,304-matrix corpus) and is therefore not
the operator the model was trained against.

## Related report

The GGML side of the same discovery is reported in the upstream repository
issue about the llama.cpp build (fused mHC Sinkhorn op), where fusing the
same 20 iterations into one op took decode from 12.1 to 44.5 tok/s on an
RTX 3080 and a fused-parity test measured KLD 0 against the unfused graph.

## Reproducing

```bash
python bench/bench_decode.py --model-path /path/to/SenseNova-Loop-9B-Preview
```

The CPU parity and launch-count checks run without a GPU:

```bash
python -m pytest tests/test_sinkhorn.py -v -s
```
