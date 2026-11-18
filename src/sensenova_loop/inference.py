"""Text and VQA inference using the repository-owned Loop model implementation."""

from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

from . import register
from .loading import load_model
from .profiling import InferenceProfiler
from .request import build_messages, validate_generation_options, validate_messages

__all__ = ["LoopInference", "GenerationResult", "build_messages"]

DEFAULT_MODEL = "sensenova/SenseNova-Loop-9B-Preview"


@dataclass
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    metrics: dict


class LoopInference:
    """Load a checkpoint once and answer single conversations, including history.

    ``messages`` uses the checkpoint's chat template and Transformers content
    blocks. Every call prepares a fresh prompt and autoregressive cache. The
    model classes are provided by this package; checkpoint code is not executed.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        *,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        attn_backend: str = "sdpa",
        local_files_only: bool = False,
        vram_mode: str = "full",
        max_memory_gib: float | None = None,
        profile: bool = False,
    ):
        self.device = torch.device(device)
        register()
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=False,
                                            local_files_only=local_files_only)
        with InferenceProfiler(self.device, profile) as profiler:
            self.processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=False, local_files_only=local_files_only
            )
            self.model = load_model(
                AutoModelForCausalLM, model_path, config, device=self.device, dtype=dtype,
                attn_backend=attn_backend, local_files_only=local_files_only,
                vram_mode=vram_mode, max_memory_gib=max_memory_gib,
            )
        self.load_metrics = profiler.metrics

    def generate(self, messages: list[dict], **kwargs) -> str:
        """Return newly generated text; see generate_result for metrics and options."""
        return self.generate_result(messages, **kwargs).text

    @torch.inference_mode()
    def generate_result(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int = 256,
        enable_thinking: bool = False,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int | None = None,
        repetition_penalty: float = 1.0,
        profile: bool = False,
    ) -> GenerationResult:
        """Return only newly generated text; thinking text is retained if enabled."""
        validate_messages(messages)
        validate_generation_options(max_new_tokens, do_sample, temperature, top_p, top_k, repetition_penalty)

        with InferenceProfiler(self.device, profile) as profiler:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device)
            prompt_length = inputs["input_ids"].shape[1]
            context_length = self.model.config.text_config.max_position_embeddings
            if prompt_length + max_new_tokens > context_length:
                raise ValueError(
                    f"Prompt ({prompt_length}) + max_new_tokens ({max_new_tokens}) "
                    f"exceeds the model context length ({context_length})"
                )

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                # Clear checkpoint sampling defaults for greedy decoding.
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                top_k=top_k if do_sample else None,
                repetition_penalty=repetition_penalty,
                use_cache=True,
            )
            text = self.processor.decode(outputs[0, prompt_length:], skip_special_tokens=True)
        if profile:
            profiler.metrics["output_tokens_per_second"] = (
                (outputs.shape[1] - prompt_length) / profiler.metrics["elapsed_seconds"]
            )
        return GenerationResult(text, prompt_length, outputs.shape[1] - prompt_length, profiler.metrics)
