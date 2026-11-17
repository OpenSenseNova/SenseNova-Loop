"""Canonical Loop/mHC model implementation and Transformers registration."""

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText

from .modeling_sensenova_loop import (
    SenseNovaLoopConfig,
    SenseNovaLoopForConditionalGeneration,
    SenseNovaLoopModel,
)


def register():
    AutoConfig.register("sensenova_loop", SenseNovaLoopConfig, exist_ok=True)
    AutoModel.register(SenseNovaLoopConfig, SenseNovaLoopModel, exist_ok=True)
    AutoModelForCausalLM.register(SenseNovaLoopConfig, SenseNovaLoopForConditionalGeneration, exist_ok=True)
    AutoModelForImageTextToText.register(SenseNovaLoopConfig, SenseNovaLoopForConditionalGeneration, exist_ok=True)
