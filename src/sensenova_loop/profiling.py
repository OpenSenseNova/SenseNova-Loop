"""Synchronized latency and peak allocated CUDA memory for inference phases."""

from time import perf_counter

import torch


class InferenceProfiler:
    def __init__(self, device, enabled=False):
        self.device = torch.device(device)
        self.enabled = enabled
        self.metrics = {}

    def __enter__(self):
        if self.enabled:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
            self.started = perf_counter()
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.enabled and exc_type is None:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                self.metrics["peak_allocated_gib"] = torch.cuda.max_memory_allocated(self.device) / 1024**3
            self.metrics["elapsed_seconds"] = perf_counter() - self.started
