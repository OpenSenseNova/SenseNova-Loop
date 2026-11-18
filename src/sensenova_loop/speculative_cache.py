"""Transactions on Loop's independent attention and recurrent execution slots."""

from transformers.cache_utils import CacheLayerMixin, DynamicCache, LinearAttentionCacheLayerMixin


class CacheCheckpoint:
    """Save recurrent state without copying the growing full-attention KV cache.

    Dynamic attention updates append new tensors. Recurrent and convolution
    updates mutate their tensors in place, so their values must be copied before
    verifying a speculative block. Rollback precedes replay of accepted tokens.
    """

    def __init__(self, cache):
        if not isinstance(cache, DynamicCache):
            raise TypeError("MTP requires a DynamicCache")
        self.cache = cache
        self.length = cache.get_seq_length()
        self.recurrent = []
        for layer in cache.layers:
            if isinstance(layer, LinearAttentionCacheLayerMixin):
                states = {
                    name: {key: value.clone() for key, value in getattr(layer, name).items()}
                    for name in ("conv_states", "recurrent_states")
                }
                self.recurrent.append((layer, states))

    def restore(self):
        for layer in self.cache.layers:
            if isinstance(layer, CacheLayerMixin):
                added = layer.get_seq_length() - self.length
                if added:
                    layer.crop(-added)
        for layer, states in self.recurrent:
            for name, values in states.items():
                for key, value in values.items():
                    getattr(layer, name)[key].copy_(value)


def crop_draft_cache(cache, length):
    """Discard speculative full-attention states before teacher-forced catch-up."""
    added = cache.get_seq_length() - length
    if added:
        cache.crop(-added)
