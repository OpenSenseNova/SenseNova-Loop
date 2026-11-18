"""Include separately published MTP weights in the normal HF loading pipeline."""

from dataclasses import replace
from pathlib import Path

from safetensors import safe_open
from transformers.utils.hub import cached_file


def include_mtp_companion(checkpoint_files, load_config):
    """Resolve an unindexed predictor without modifying the checkpoint directory.

    Existing checkpoint MTP tensors take precedence, including incomplete heads
    whose missing tensors must still be reported. HF remains responsible for
    dtype conversion, shape checks, device placement and offload metadata.
    """
    if not checkpoint_files or not all(name.endswith(".safetensors") for name in checkpoint_files):
        return checkpoint_files, load_config
    weight_map = {}
    for filename in checkpoint_files:
        with safe_open(filename, framework="pt", device="cpu") as saved:
            keys = list(saved.keys())
        if any(key.startswith("mtp.") for key in keys):
            return checkpoint_files, load_config
        weight_map.update(dict.fromkeys(keys, str(Path(filename).resolve())))

    source = Path(load_config.pretrained_model_name_or_path)
    if source.is_file():
        sibling = source.parent / "mtp.safetensors"
        companion = str(sibling) if sibling.is_file() else None
    else:
        companion = cached_file(
            load_config.pretrained_model_name_or_path, "mtp.safetensors",
            **load_config.download_kwargs, _raise_exceptions_for_missing_entries=False,
        )
    if companion is None:
        return checkpoint_files, load_config
    with safe_open(companion, framework="pt", device="cpu") as saved:
        keys = list(saved.keys())
    if not keys or any(not key.startswith("mtp.") for key in keys):
        raise ValueError("mtp.safetensors must contain only MTP weights")
    weight_map.update(dict.fromkeys(keys, str(Path(companion).resolve())))
    # Absolute shard paths also work when a Hub companion resolves elsewhere.
    # Supply a complete map even for an otherwise unsharded checkpoint, so disk
    # offload discovers the extra file before any tensors are dispatched.
    metadata = dict(load_config.sharded_metadata or {})
    metadata.update(weight_map=weight_map, all_checkpoint_keys=list(weight_map))
    return [*checkpoint_files, companion], replace(load_config, sharded_metadata=metadata)
