"""Model placement and CPU offload policies for a selected CUDA device."""

import math

import psutil
import torch
from accelerate import cpu_offload


def load_model(model_class, model_path, config, *, device, dtype, attn_backend,
               local_files_only, vram_mode="full", max_memory_gib=None):
    """Load full GPU, budgeted placement, or full CPU weight offload.

    Accelerate owns weight dispatch/hooks, including repeated physical layers.
    mHC parameter precision is preserved by the model implementation.
    """
    if vram_mode not in ("full", "balanced", "low"):
        raise ValueError("vram_mode must be full, balanced or low")
    if vram_mode != "full" and device.type != "cuda":
        raise ValueError("CPU offload modes require a CUDA execution device")
    if max_memory_gib is not None and (
        not math.isfinite(max_memory_gib) or max_memory_gib <= 0 or vram_mode != "balanced"
    ):
        raise ValueError("max_memory_gib must be finite, positive and used with balanced mode")
    options = dict(config=config, trust_remote_code=False, dtype=dtype,
                   attn_implementation=attn_backend, local_files_only=local_files_only)
    if vram_mode == "balanced":
        index = device.index if device.index is not None else torch.cuda.current_device()
        available = torch.cuda.mem_get_info(index)[0]
        budget = int(max_memory_gib * 1024**3) if max_memory_gib is not None else int(available * 0.8)
        model = model_class.from_pretrained(model_path, device_map="auto",
                                            max_memory={index: budget, "cpu": psutil.virtual_memory().available}, **options)
        if model.device.type == "cpu":
            model = cpu_offload(model, execution_device=device, offload_buffers=True)
    elif vram_mode == "low":
        model = model_class.from_pretrained(model_path, device_map={"": "cpu"}, **options)
        model = cpu_offload(model, execution_device=device, offload_buffers=True)
    else:
        model = model_class.from_pretrained(model_path, device_map={"": str(device)}, **options)
    return model.eval()
