"""Token generation strategies with a common sequence and metrics contract."""

from dataclasses import dataclass

import torch


@dataclass
class TokenGenerationOutput:
    sequences: torch.Tensor
    metrics: dict


def generate_tokens(model, inputs, *, mtp_tokens=0, **options):
    """Select the requested strategy and return its tokens and request metrics.

    Request preparation, decoding and profiling consume this contract without
    depending on the strategy's model entry point or result representation.
    """
    if mtp_tokens:
        result = model.generate_mtp(**inputs, **options, mtp_tokens=mtp_tokens)
        return TokenGenerationOutput(result.sequences, {"mtp": result.metrics})
    sequences = model.generate(**inputs, **options, use_cache=True)
    return TokenGenerationOutput(sequences, {})
