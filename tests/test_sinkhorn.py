"""Parity and launch-count tests for the mHC sinkhorn pair reduction.

The reference oracle is the pre-change eager implementation, kept verbatim
below. The shipped sinkhorn replaces each two-element logsumexp with an
elementwise evaluation of the same reduction; these tests prove the two
agree bit-for-bit and that the pair path drops the reduce dispatches.

CPU aten op counts are a launch-count proxy: on CUDA, aten::logsumexp
lowers to a similar composite whose two reduce kernels made up roughly
1.1 s of the 1.43 s GPU-busy time per decode token in the Preview profile
documented in bench/decode_overhead.md.
"""

import torch
from torch.profiler import ProfilerActivity, profile

from sensenova_loop.models.modeling_sensenova_loop import sinkhorn


def reference_sinkhorn(logits, iterations):
    """Pre-change implementation: alternating aten::logsumexp normalizations."""
    logits = logits.float()
    for _ in range(iterations):
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        logits = logits - torch.logsumexp(logits, dim=-2, keepdim=True)
    return logits.exp()


def corpus():
    """Randomized and adversarial 2x2 float32 logit matrices."""
    g = torch.Generator().manual_seed(20261118)
    parts = [
        torch.randn(40_000, 2, 2, generator=g),
        torch.rand(30_000, 2, 2, generator=g) * 20 - 10,
        torch.rand(20_000, 2, 2, generator=g) * 100 - 50,
    ]
    # Router-realistic magnitudes: the bias init puts log(15) on the mixing
    # diagonal and 0 off it, near the saturated mixing regime.
    router = torch.empty(30_000, 2, 2)
    diagonal = torch.log(torch.tensor(15.0))
    router[..., 0, 0] = diagonal + torch.randn(30_000, generator=g) * 3
    router[..., 1, 1] = diagonal + torch.randn(30_000, generator=g) * 3
    router[..., 0, 1] = torch.randn(30_000, generator=g) * 3
    router[..., 1, 0] = torch.randn(30_000, generator=g) * 3
    parts.append(router)

    adversarial = [
        torch.zeros(2, 2),                          # all-equal zeros
        torch.full((2, 2), 3.7),                     # all-equal
        torch.full((2, 2), -2.5),                    # all-equal negative
        torch.full((2, 2), 1e20),                    # equal at huge magnitude
        torch.tensor([[5.0, 5.0], [7.0, 7.0]]),      # equal pairs along dim -1
        torch.tensor([[5.0, 7.0], [5.0, 7.0]]),      # equal pairs along dim -2
        torch.tensor([[2.70805, 0.0], [0.0, 2.70805]]),  # exact router init bias
        torch.tensor([[60.0, -60.0], [-40.0, 40.0]]),    # saturated mixing
        torch.tensor([[3.4e38, -3.4e38], [1e38, -1e38]]),  # subtraction overflow
        torch.tensor([[1e30, -1e30], [1e-30, -1e-30]]),
        torch.tensor([[1.1754944e-38, 1.4e-45], [0.0, 5.0e-8]]),  # subnormals
        torch.tensor([[0.1, 0.0], [0.3, 0.2]]),     # small non-saturated mix
    ]
    for matrix in adversarial:
        parts.append(matrix.expand(20, 2, 2).contiguous())

    # Pairs one float32 ulp apart, both orderings.
    base = torch.randn(64, generator=g)
    ulp = torch.stack([base, torch.nextafter(base, torch.tensor(float("inf")))], dim=-1)
    parts.append(torch.stack([ulp, torch.flip(ulp, dims=[-1])], dim=-2))
    return torch.cat(parts)


def test_sinkhorn_pair_path_is_bit_exact_against_the_aten_reference():
    """The shipped sinkhorn must reproduce the eager reference exactly.

    Pass criterion: max output delta over the full corpus is exactly 0.0,
    matching the A/A floor of the reference itself.
    """
    logits = corpus()
    expected = reference_sinkhorn(logits, 20)
    actual = sinkhorn(logits, 20)

    aa_floor = (reference_sinkhorn(logits, 20) - expected).abs().max().item()
    max_delta = (actual - expected).abs().max().item()
    print(f"\ncorpus matrices: {logits.shape[0]} x 20 iterations, float32, CPU")
    print(f"reference A/A floor max delta: {aa_floor}")
    print(f"shipped-vs-reference max delta: {max_delta} (pass criterion: 0.0)")

    assert aa_floor == 0.0
    assert torch.equal(actual, expected), f"not bit-exact; max delta {max_delta}"


def test_sinkhorn_decode_and_prefill_shapes_match_the_reference():
    """Decode-shaped (1, 1, 2, 2) and larger prefills stay bit-exact too."""
    g = torch.Generator().manual_seed(7)
    for shape in [(1, 1, 2, 2), (2, 2), (4, 7, 2, 2), (1, 512, 2, 2)]:
        logits = torch.randn(*shape, generator=g) * 6
        assert torch.equal(sinkhorn(logits, 20), reference_sinkhorn(logits, 20)), shape


def test_sinkhorn_streams4_uses_the_aten_fallback():
    """Four-stream checkpoints keep the aten path unchanged."""
    g = torch.Generator().manual_seed(11)
    logits = torch.randn(4096, 4, 4, generator=g)
    assert torch.equal(sinkhorn(logits, 20), reference_sinkhorn(logits, 20))


def test_sinkhorn_outputs_remain_doubly_stochastic():
    """Sanity for the mixing matrices on benign inputs (house tolerance)."""
    torch.manual_seed(3)
    matrix = sinkhorn(torch.randn(64, 2, 2), 20)
    torch.testing.assert_close(matrix.sum(-1), torch.ones(64, 2), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(matrix.sum(-2), torch.ones(64, 2), rtol=1e-4, atol=1e-4)


def _aten_counts(fn, calls=1):
    fn()  # warm up outside the profiled window
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(calls):
            fn()
    entries = [e for e in prof.key_averages() if e.key.startswith("aten::")]
    total = sum(e.count for e in entries)
    logsumexp = sum(e.count for e in entries if e.key == "aten::logsumexp")
    reduces = sum(e.count for e in entries if e.key in ("aten::amax", "aten::sum"))
    return total, logsumexp, reduces


def test_sinkhorn_pair_path_removes_reduce_dispatches():
    """Launch-count smoke: one sinkhorn call, reference vs shipped, on CPU.

    The reference keeps 40 aten::logsumexp composites per call (each with an
    amax and a sum reduce); the shipped pair path keeps none of them. Exact
    totals vary across torch versions, so the assertion is on the reduces.
    """
    logits = torch.randn(1, 1, 2, 2)
    reference = _aten_counts(lambda: reference_sinkhorn(logits, 20))
    shipped = _aten_counts(lambda: sinkhorn(logits, 20))

    print(f"\nreference one-call aten ops: total={reference[0]} "
          f"logsumexp={reference[1]} reduces={reference[2]}")
    print(f"shipped   one-call aten ops: total={shipped[0]} "
          f"logsumexp={shipped[1]} reduces={shipped[2]}")

    assert reference[1] >= 40 and reference[2] >= 40
    assert shipped[1] == 0, "pair path still dispatches aten::logsumexp"
    assert shipped[2] == 0, "pair path still dispatches reduce kernels"
    assert shipped[0] < reference[0], "pair path did not reduce dispatch count"
