"""Single-conversation speculative decoding using the checkpoint's MTP head."""

from dataclasses import dataclass

import torch
from transformers import LogitsProcessorList

from .request import validate_generation_options
from .speculative_cache import CacheCheckpoint, crop_draft_cache


@dataclass
class MTPGenerationOutput:
    sequences: torch.Tensor
    metrics: dict


def verify_candidate(target, draft, token, *, do_sample):
    """Accept a draft token or draw from the exact residual target distribution.

    Sampling accepts with min(1, p(token)/q(token)). On rejection the correction
    is drawn from normalized max(p-q, 0), not from the unmodified target.
    """
    if not do_sample:
        correction = target.argmax(-1, keepdim=True)
        return bool(torch.equal(correction, token)), correction
    ratio = target.gather(-1, token) / draft.gather(-1, token)
    if bool((torch.rand((), device=target.device) < ratio.clamp(max=1)).item()):
        return True, token
    residual = (target - draft).clamp_min(0)
    return False, torch.multinomial(residual, 1)


class _SamplingPolicy:
    def __init__(self, model, inputs, max_new_tokens, do_sample, temperature, top_p, top_k,
                 repetition_penalty):
        self.config, _ = model._prepare_generation_config(None, max_new_tokens=max_new_tokens,
                           max_length=inputs["input_ids"].shape[1] + max_new_tokens,
                           do_sample=do_sample, temperature=temperature if do_sample else None,
                           top_p=top_p if do_sample else None, top_k=top_k if do_sample else None,
                           repetition_penalty=repetition_penalty, use_cache=True)
        if self.config.num_beams != 1 or self.config.num_return_sequences != 1:
            raise ValueError("MTP supports one sequence without beam search")
        if self.config.guidance_scale not in (None, 1) or self.config.stop_strings:
            raise ValueError("MTP does not support classifier-free guidance or stop strings")
        if self.config.watermarking_config is not None:
            raise ValueError("MTP does not support watermarking logits processors")
        if self.config.token_healing or self.config.max_time is not None:
            raise ValueError("MTP does not support token healing or max_time")
        model._prepare_special_tokens(self.config, kwargs_has_attention_mask=True,
                                      device=inputs["input_ids"].device, batch_size=1)
        self.processors = model._get_logits_processor(
            self.config, input_ids_seq_length=inputs["input_ids"].shape[1],
            encoder_input_ids=inputs["input_ids"], logits_processor=LogitsProcessorList(),
            device=inputs["input_ids"].device, model_kwargs=inputs,
        )
        eos = self.config.eos_token_id
        self.eos = set(eos if isinstance(eos, (list, tuple)) else [] if eos is None else [eos])

    def distribution(self, prefix, logits):
        scores = self.processors(prefix, logits.float().clone())
        return scores.softmax(-1) if self.config.do_sample else scores

    def sample(self, distribution):
        return (torch.multinomial(distribution, 1) if self.config.do_sample
                else distribution.argmax(-1, keepdim=True))

    def finished(self, token):
        return int(token.item()) in self.eos


def _prefill(model, inputs):
    # Capture the actual post-vision embeddings and M-RoPE positions passed by
    # the backbone. No second vision encoder pass or per-layer hidden history.
    captured = {}

    def capture(_module, _args, kwargs):
        captured["embeddings"] = kwargs["inputs_embeds"]
        captured["positions"] = kwargs.get("position_ids")

    handle = model.model.language_model.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        output = model.model(**inputs, use_cache=True, output_hidden_states=False)
    finally:
        handle.remove()
    positions = captured["positions"]
    if positions is None:
        length = inputs["input_ids"].shape[1]
        positions = torch.arange(length, device=inputs["input_ids"].device).view(1, 1, -1).expand(4, 1, -1)
    elif positions.ndim == 2:
        positions = positions.unsqueeze(0).expand(4, -1, -1)
    return output, captured["embeddings"], positions


@torch.inference_mode()
def generate_mtp(model, inputs, *, max_new_tokens=256, mtp_tokens=3, do_sample=False,
                 temperature=0.7, top_p=0.9, top_k=None, repetition_penalty=1.0):
    """Verify MTP proposals in blocks, restoring hybrid state on rejection.

    All returned tokens are validated by the target model. Sampling preserves
    its distribution; the extra random draws do not preserve seeded AR output.
    Inputs are a single unpadded text/image conversation with a fresh cache.
    """
    validate_generation_options(max_new_tokens, do_sample, temperature, top_p, top_k,
                                repetition_penalty, mtp_tokens)
    if mtp_tokens == 0:
        raise ValueError("generate_mtp requires mtp_tokens > 0")
    if getattr(model, "_mtp_configuration_error", None):
        raise ValueError(model._mtp_configuration_error)
    if model.mtp is None:
        raise ValueError("The checkpoint does not configure an MTP head")
    if getattr(model, "_mtp_missing_keys", ()):
        raise ValueError("The checkpoint is missing MTP weights: " + ", ".join(model._mtp_missing_keys))
    ids = inputs["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
        raise ValueError("MTP requires a single nonempty conversation")
    mask = inputs.get("attention_mask")
    if mask is not None and (mask.ndim != 2 or not bool(mask.bool().all())):
        raise ValueError("MTP requires an unpadded conversation")
    if inputs.get("past_key_values") is not None:
        raise ValueError("MTP requires a fresh conversation cache")
    if ids.shape[1] + max_new_tokens > model.config.text_config.max_position_embeddings:
        raise ValueError("Prompt plus max_new_tokens exceeds the model context length")
    policy = _SamplingPolicy(model, inputs, max_new_tokens, do_sample, temperature,
                              top_p, top_k, repetition_penalty)
    metrics = dict(draft_tokens=0, accepted_tokens=0, verification_rounds=0,
                   target_forward_calls=1, replay_forward_calls=0)
    prompt_length = ids.shape[1]
    prefill, embeddings, positions = _prefill(model, inputs)
    cache = prefill.past_key_values
    previous = prefill.last_hidden_state[:, -1:].clone()
    first = policy.sample(policy.distribution(ids, model.lm_head(previous)[:, -1]))
    sequence = torch.cat((ids, first), dim=1)
    limit = prompt_length + max_new_tokens
    if sequence.shape[1] == limit or policy.finished(first):
        return MTPGenerationOutput(sequence, dict(metrics, acceptance_rate=0.0))

    draft_cache = None
    if prompt_length > 1:
        _, draft_cache = model.mtp(prefill.last_hidden_state[:, :-1], embeddings[:, 1:], positions[-3:, :, 1:])
    # Continue each supplied axis from its last token, matching HF generation.
    # Four-axis inputs also retain their separate text/causal-mask positions.
    next_position = positions[:, :, -1:] + 1
    del prefill, embeddings, positions

    def rope(start, length):
        return start + torch.arange(length, device=start.device, dtype=start.dtype).view(1, 1, -1)

    def target_forward(tokens, start):
        metrics["target_forward_calls"] += 1
        return model.model.language_model(input_ids=tokens, position_ids=rope(start, tokens.shape[1]),
                                          past_key_values=cache, use_cache=True,
                                          output_hidden_states=False)

    while sequence.shape[1] < limit:
        prefix_length = cache.get_seq_length()
        draft_length = draft_cache.get_seq_length() if draft_cache is not None else 0
        draft_sequence = sequence
        draft_hidden = previous
        proposals, distributions = [], []
        count = min(mtp_tokens, limit - sequence.shape[1])
        for index in range(count):
            draft_hidden, draft_cache = model.mtp(
                draft_hidden, model.get_input_embeddings()(draft_sequence[:, -1:]),
                rope(next_position + index, 1)[-3:], past_key_values=draft_cache,
            )
            distribution = policy.distribution(draft_sequence, model.lm_head(draft_hidden)[:, -1])
            candidate = policy.sample(distribution)
            proposals.append(candidate)
            distributions.append(distribution)
            draft_sequence = torch.cat((draft_sequence, candidate), dim=1)
            if policy.finished(candidate):
                break

        checkpoint = CacheCheckpoint(cache)
        block = draft_sequence[:, prefix_length:]
        verified = target_forward(block, next_position)
        logits = model.lm_head(verified.last_hidden_state)
        metrics["draft_tokens"] += len(proposals)
        metrics["verification_rounds"] += 1
        accepted = 0
        correction = None
        for index, (candidate, distribution) in enumerate(zip(proposals, distributions)):
            target = policy.distribution(sequence, logits[:, index])
            keep, correction = verify_candidate(target, distribution, candidate, do_sample=do_sample)
            if not keep:
                break
            accepted += 1
            sequence = torch.cat((sequence, candidate), dim=1)
            if sequence.shape[1] == limit or policy.finished(candidate):
                metrics["accepted_tokens"] += accepted
                return MTPGenerationOutput(sequence, dict(metrics, acceptance_rate=
                    metrics["accepted_tokens"] / metrics["draft_tokens"]))
        metrics["accepted_tokens"] += accepted
        if accepted == len(proposals):
            correction = policy.sample(policy.distribution(sequence, logits[:, -1]))
            committed_hidden = verified.last_hidden_state
        else:
            checkpoint.restore()
            committed = block[:, :accepted + 1]
            committed_hidden = target_forward(committed, next_position).last_hidden_state
            metrics["replay_forward_calls"] += 1
        sequence = torch.cat((sequence, correction), dim=1)
        if sequence.shape[1] == limit or policy.finished(correction):
            break
        # Rebuild only this round's MTP states using actual target states, not
        # speculative hidden states. The accepted prefix remains untouched.
        crop_draft_cache(draft_cache, draft_length)
        committed = block[:, :accepted + 1]
        shifted_hidden = torch.cat((previous, committed_hidden[:, :accepted]), dim=1)
        _, draft_cache = model.mtp(
            shifted_hidden, model.get_input_embeddings()(committed),
            rope(next_position, accepted + 1)[-3:], past_key_values=draft_cache,
        )
        previous = committed_hidden[:, -1:].clone()
        next_position += accepted + 1

    return MTPGenerationOutput(sequence, dict(metrics, acceptance_rate=
        metrics["accepted_tokens"] / metrics["draft_tokens"] if metrics["draft_tokens"] else 0.0))
