import copy

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache

from sensenova_loop import register
from sensenova_loop.models.modeling_sensenova_loop import (
    MHCRouter,
    SenseNovaLoopConfig,
    SenseNovaLoopForConditionalGeneration,
    SenseNovaLoopModel,
    execution_plan,
    sinkhorn,
)


def tiny_config():
    return SenseNovaLoopConfig(
        text_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                         num_attention_heads=2, num_key_value_heads=2, head_dim=16,
                         vocab_size=128, layer_types=["full_attention"] * 4,
                         max_position_embeddings=128, partial_rotary_factor=1.0,
                         rope_parameters=dict(rope_type="default", rope_theta=10000,
                                              partial_rotary_factor=1.0, mrope_section=[2, 2, 4])),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                           out_hidden_size=32, num_position_embeddings=16,
                           deepstack_visual_indexes=[]),
        image_token_id=125, video_token_id=126, vision_start_token_id=123, vision_end_token_id=124,
        loop_settings=dict(start=0, end=4, eval_count=2, counts=[2]),
        mhc=dict(start=0, end=4, streams=2, iterations=20, fixed_identity=False, pre_scale=1.0),
    )


def test_registered_model_roundtrip_without_checkpoint_code(tmp_path):
    register()
    config = tiny_config()
    model = SenseNovaLoopForConditionalGeneration(config).eval()
    model.save_pretrained(tmp_path)
    loaded_config = AutoConfig.from_pretrained(tmp_path, trust_remote_code=False)
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, trust_remote_code=False)
    assert isinstance(loaded_config, SenseNovaLoopConfig)
    assert type(loaded) is SenseNovaLoopForConditionalGeneration
    assert execution_plan(loaded.config.to_dict()) == tuple(range(4)) * 2
    assert len(loaded.model.language_model.layers) == 4


@pytest.mark.parametrize("model_class", [SenseNovaLoopModel, SenseNovaLoopForConditionalGeneration])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision_loading_preserves_exported_fp32_parameters(tmp_path, model_class, dtype):
    config = tiny_config()
    config.text_config.layer_types = ["linear_attention"] * 3 + ["full_attention"]
    config.text_config.linear_num_key_heads = 2
    config.text_config.linear_num_value_heads = 2
    config.text_config.linear_key_head_dim = 16
    config.text_config.linear_value_head_dim = 16
    model = model_class(config)
    originals = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".mhc_attn." in name or ".mhc_ffn." in name or name.endswith((".A_log", ".dt_bias")):
                # Values deliberately lose bits if converted to either low precision dtype.
                parameter.copy_(torch.linspace(0.1234567, 0.8765432, parameter.numel()).reshape_as(parameter))
                originals[name] = parameter.detach().clone()
                assert not torch.equal(parameter, parameter.to(dtype).float())
    assert len(originals) == 22  # Four router tensors per layer, plus two per linear layer.
    model.save_pretrained(tmp_path)
    loaded = model_class.from_pretrained(tmp_path, dtype=dtype, local_files_only=True)
    parameters = dict(loaded.named_parameters())
    for name, expected in originals.items():
        assert parameters[name].dtype == torch.float32, name
        torch.testing.assert_close(parameters[name], expected, rtol=0, atol=0)
    embedding = next(value for name, value in parameters.items() if name.endswith("embed_tokens.weight"))
    assert embedding.dtype == dtype


def test_cached_forward_matches_full_sequence_with_shared_layers():
    torch.manual_seed(1)
    model = SenseNovaLoopForConditionalGeneration(tiny_config()).eval()
    with torch.inference_mode():
        prefix = model(input_ids=torch.tensor([[2, 3]]), use_cache=True)
        step = model(input_ids=torch.tensor([[4]]), past_key_values=prefix.past_key_values, use_cache=True)
        full = model(input_ids=torch.tensor([[2, 3, 4]]), use_cache=False)
    assert len(step.past_key_values.layers) == 8
    assert step.past_key_values.get_seq_length() == 3
    torch.testing.assert_close(step.logits[:, -1], full.logits[:, -1], rtol=1e-4, atol=1e-5)


def test_router_precision_and_sinkhorn_constraints():
    settings = tiny_config().mhc
    router = MHCRouter(32, settings)
    before = router.bias.detach().clone()
    router.bfloat16()
    assert router.weight.dtype == router.bias.dtype == torch.float32
    torch.testing.assert_close(router.bias, before, rtol=0, atol=0)
    matrix = sinkhorn(torch.randn(2, 2), 20)
    torch.testing.assert_close(matrix.sum(-1), torch.ones(2), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(matrix.sum(-2), torch.ones(2), rtol=1e-4, atol=1e-4)


def test_future_checkpoint_version_fails_before_allocation():
    with pytest.raises(RuntimeError, match="requires sensenova-loop"):
        SenseNovaLoopConfig(sensenova_loop_min_version="999.0.0")


def test_invalid_loop_plan_is_rejected():
    config = copy.deepcopy(tiny_config().to_dict())
    config["loop_settings"]["eval_count"] = 0
    with pytest.raises(ValueError, match="Invalid Loop"):
        execution_plan(config)


def test_generate_and_empty_foreign_cache_for_full_attention():
    model = SenseNovaLoopForConditionalGeneration(tiny_config()).eval()
    inputs = torch.tensor([[2, 3]])
    with torch.inference_mode():
        generated = model.generate(input_ids=inputs, max_new_tokens=2, do_sample=False)
        fresh = model(input_ids=inputs, past_key_values=DynamicCache(), use_cache=True)
        full = model(input_ids=inputs, use_cache=False)
    assert generated.shape == (1, 4)
    assert len(fresh.past_key_values.layers) == 8
    torch.testing.assert_close(fresh.logits, full.logits)


def test_populated_foreign_cache_is_rejected():
    model = SenseNovaLoopForConditionalGeneration(tiny_config()).eval()
    foreign = DynamicCache()
    foreign.update(torch.ones(1, 2, 1, 16), torch.ones(1, 2, 1, 16), 0)
    with torch.inference_mode(), pytest.raises(ValueError, match="populated cache"):
        model(input_ids=torch.tensor([[2]]), past_key_values=foreign, use_cache=True)


@pytest.mark.parametrize("state_kind", ["kv", "recurrent"])
def test_hybrid_foreign_cache_checks_both_state_kinds(state_kind):
    cache_config = tiny_config().text_config
    cache_config.layer_types = ["hybrid"] * 4
    foreign = DynamicCache(config=cache_config)
    if state_kind == "kv":
        foreign.update(torch.ones(1, 2, 1, 16), torch.ones(1, 2, 1, 16), 0)
        assert not foreign.has_previous_state()
    else:
        foreign.update_recurrent_state(torch.ones(1, 2, 8, 8), 0)
        assert foreign.get_seq_length() == 0
    model = SenseNovaLoopForConditionalGeneration(tiny_config()).eval()
    with torch.inference_mode(), pytest.raises(ValueError, match="populated cache"):
        model(input_ids=torch.tensor([[2]]), past_key_values=foreign, use_cache=True)
