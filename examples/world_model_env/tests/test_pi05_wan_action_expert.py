from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from train_pi05_wan_action_expert import (
    Args,
    _aggregate_action_loss,
    _allclose_or_mismatch,
    _effective_task_cvar_weight,
    _validate_init_checkpoint_action_normalization,
    _validate_init_checkpoint_model_kwargs,
    run_train_eval,
)
from world_model.pi05_wan_action_expert import (
    FUTURE_LEAKAGE_KEYS,
    ActionDenoisingContext,
    CachedWanPrefixActionDataset,
    SinusoidalTimeEmbedding,
    WanPi05ActionExpert,
    flow_matching_loss,
    flow_matching_loss_per_sample_parts,
    load_cached_prefix_dataset,
    load_wan_pi05_action_expert_checkpoint,
    masked_action_mse,
    masked_action_mse_per_sample,
    masked_action_mse_per_sample_parts,
    pi05_flow_targets,
    predict_denormalized_action_chunk,
    sample_actions,
    write_fake_prefix_cache,
)


def _tiny_model(
    conditioning_mode: str = "wan_prefix_state",
    timestep_conditioning: str = "additive",
    timestep_embedding_style: str = "diffusion",
    decoder_arch: str = "encoder",
) -> WanPi05ActionExpert:
    return WanPi05ActionExpert(
        prefix_dim=8,
        state_dim=5,
        action_dim=3,
        action_horizon=4,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        conditioning_mode=conditioning_mode,
        timestep_conditioning=timestep_conditioning,
        timestep_embedding_style=timestep_embedding_style,
        decoder_arch=decoder_arch,
    )


def _tiny_model_kwargs(decoder_arch: str = "encoder") -> dict[str, object]:
    return {
        "prefix_dim": 8,
        "state_dim": 5,
        "action_dim": 3,
        "action_horizon": 4,
        "hidden_dim": 16,
        "num_layers": 1,
        "num_heads": 4,
        "dropout": 0.0,
        "conditioning_mode": "wan_prefix_state",
        "timestep_conditioning": "additive",
        "timestep_embedding_style": "diffusion",
        "decoder_arch": decoder_arch,
    }


def _tiny_batch(batch_size: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(11)
    prefix_tokens = torch.randn(batch_size, 3, 8, generator=generator)
    state = torch.randn(batch_size, 5, generator=generator)
    actions = torch.randn(batch_size, 4, 3, generator=generator)
    return prefix_tokens, state, actions


class _MaskApplyingSpyEncoder(torch.nn.Module):
    def __init__(self, context_len: int) -> None:
        super().__init__()
        self.context_len = context_len
        self.context_outputs: list[torch.Tensor] = []
        self.masks: list[torch.Tensor] = []

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            weights = torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device, dtype=tokens.dtype)
        else:
            self.masks.append(mask.detach().clone())
            weights = (~mask).to(device=tokens.device, dtype=tokens.dtype)
        encoded = torch.einsum("ij,bjh->bih", weights, tokens)
        self.context_outputs.append(encoded[:, : self.context_len].detach().clone())
        return encoded


class _RecordingContextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[torch.Tensor] = []
        self.outputs: list[torch.Tensor] = []

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        encoded = context * 2.0 + 0.5
        self.inputs.append(context.detach().clone())
        self.outputs.append(encoded.detach().clone())
        return encoded


class _RecordingDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.memories: list[torch.Tensor] = []

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        self.memories.append(memory.detach().clone())
        return tgt + memory.mean(dim=1, keepdim=True)


def _set_cache_actions(cache_dir, value: float) -> None:
    for row_path in sorted(cache_dir.glob("*.pt")):
        row = torch.load(row_path, weights_only=False)
        row["actions"] = torch.full_like(row["actions"], value)
        torch.save(row, row_path)


def _set_first_cache_row_actions_and_mask(cache_dir, *, value: float, action_mask: torch.Tensor) -> None:
    row_path = sorted(cache_dir.glob("*.pt"))[0]
    row = torch.load(row_path, weights_only=False)
    row["actions"] = torch.full_like(row["actions"], value)
    row["action_mask"] = action_mask.to(dtype=torch.float32)
    torch.save(row, row_path)


def _set_cache_wan_action_modes(cache_dir, modes: list[str | None]) -> None:
    row_paths = sorted(cache_dir.glob("*.pt"))
    assert len(row_paths) == len(modes)
    for row_path, mode in zip(row_paths, modes, strict=True):
        row = torch.load(row_path, weights_only=False)
        metadata = dict(row.get("metadata") or {})
        if mode is None:
            metadata.pop("wan_action_mode", None)
        else:
            metadata["wan_action_mode"] = mode
        row["metadata"] = metadata
        torch.save(row, row_path)


def _manual_full_joint_softmax_layer_output(
    layer: torch.nn.Module,
    prefix_tokens: torch.Tensor,
    action_tokens: torch.Tensor,
) -> torch.Tensor:
    normalized_prefix = layer.prefix_self_norm(prefix_tokens)
    attended_prefix, _ = layer.prefix_self_attention(normalized_prefix, normalized_prefix, normalized_prefix)
    prefix_tokens = prefix_tokens + layer.prefix_self_dropout(attended_prefix)
    prefix_tokens = prefix_tokens + layer.prefix_ff_dropout(
        layer.prefix_feedforward(layer.prefix_ff_norm(prefix_tokens))
    )

    normalized_prefix = layer.action_joint_norm(prefix_tokens)
    normalized_actions = layer.action_joint_norm(action_tokens)
    full_tokens = torch.cat([normalized_prefix, normalized_actions], dim=1)
    attention = layer.action_joint_attention
    prefix_length = normalized_prefix.shape[1]
    query = attention._split_heads(attention.query_projection(full_tokens))[:, :, prefix_length:]
    keys = attention._split_heads(attention.key_projection(full_tokens))
    values = attention._split_heads(attention.value_projection(full_tokens))
    scores = torch.matmul(query, keys.transpose(-2, -1)) / (attention.head_dim**0.5)
    weights = torch.softmax(scores, dim=-1)
    attended = torch.matmul(weights, values)
    attended = attention.output_projection(attention._merge_heads(attended))
    action_tokens = action_tokens + layer.action_joint_dropout(attended)
    return action_tokens + layer.action_ff_dropout(layer.action_feedforward(layer.action_ff_norm(action_tokens)))


def test_wan_pi05_action_expert_output_shape() -> None:
    model = _tiny_model()
    prefix_tokens, state, actions = _tiny_batch()
    time = torch.tensor([1.0, 0.25])

    velocity = model(prefix_tokens, state, actions, time)

    assert velocity.shape == actions.shape
    assert model.decoder_arch == "encoder"


def test_default_decoder_arch_is_encoder() -> None:
    torch.manual_seed(107)
    default_model = _tiny_model()
    torch.manual_seed(107)
    explicit_model = _tiny_model(decoder_arch="encoder")
    default_model.eval()
    explicit_model.eval()
    prefix_tokens, state, actions = _tiny_batch()
    time = torch.tensor([1.0, 0.25])

    default_velocity = default_model(prefix_tokens, state, actions, time)
    explicit_velocity = explicit_model(prefix_tokens, state, actions, time)

    assert default_model.decoder_arch == "encoder"
    assert torch.allclose(default_velocity, explicit_velocity)


def test_old_model_kwargs_without_decoder_arch_load_encoder_state_dict() -> None:
    torch.manual_seed(109)
    encoder_model = _tiny_model(decoder_arch="encoder")
    encoder_model.eval()
    old_model_kwargs = {
        "prefix_dim": 8,
        "state_dim": 5,
        "action_dim": 3,
        "action_horizon": 4,
        "hidden_dim": 16,
        "num_layers": 1,
        "num_heads": 4,
        "dropout": 0.0,
        "conditioning_mode": "wan_prefix_state",
        "timestep_conditioning": "additive",
    }
    prefix_tokens, state, actions = _tiny_batch()
    time = torch.tensor([1.0, 0.25])

    restored_model = WanPi05ActionExpert(**old_model_kwargs)
    restored_model.load_state_dict(encoder_model.state_dict())
    restored_model.eval()
    encoder_velocity = encoder_model(prefix_tokens, state, actions, time)
    restored_velocity = restored_model(prefix_tokens, state, actions, time)

    assert "decoder_arch" not in old_model_kwargs
    assert restored_model.decoder_arch == "encoder"
    assert torch.allclose(restored_velocity, encoder_velocity)


def test_load_checkpoint_predicts_denormalized_action_chunk(tmp_path) -> None:
    model = _tiny_model()
    for parameter in model.parameters():
        parameter.data.zero_()
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_kwargs": _tiny_model_kwargs(),
            "args": {"wan_action_mode": "current_wan_prefix_action_expert"},
            "metrics": {"wan_action_mode": "current_wan_prefix_action_expert"},
            "action_normalization": {
                "enabled": True,
                "mean": torch.tensor([10.0, 20.0, 30.0]),
                "std": torch.tensor([2.0, 3.0, 4.0]),
            },
        },
        checkpoint_path,
    )
    loaded = load_wan_pi05_action_expert_checkpoint(checkpoint_path, device="cpu")
    noise = torch.ones(4, 3)

    action_chunk = predict_denormalized_action_chunk(
        loaded,
        torch.zeros(3, 8),
        torch.zeros(5),
        num_steps=2,
        noise=noise,
    )

    expected = torch.tensor([12.0, 23.0, 34.0]).view(1, 3).expand(4, 3)
    assert action_chunk.shape == (4, 3)
    assert torch.allclose(action_chunk, expected)
    assert loaded.action_normalization["enabled"] is True
    assert loaded.wan_action_mode == "current_wan_prefix_action_expert"


def test_suffix_prefix_cache_checkpoint_predicts_denormalized_action_chunk(tmp_path) -> None:
    model = _tiny_model(decoder_arch="suffix_prefix_cache")
    for parameter in model.parameters():
        parameter.data.zero_()
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_kwargs": _tiny_model_kwargs(decoder_arch="suffix_prefix_cache"),
            "args": {"decoder_arch": "suffix_prefix_cache"},
            "metrics": {"decoder_arch": "suffix_prefix_cache"},
            "action_normalization": {"enabled": False},
        },
        checkpoint_path,
    )

    loaded = load_wan_pi05_action_expert_checkpoint(checkpoint_path, device="cpu")
    action_chunk = predict_denormalized_action_chunk(
        loaded,
        torch.zeros(3, 8),
        torch.zeros(5),
        num_steps=2,
        noise=torch.ones(4, 3),
    )

    assert loaded.model.decoder_arch == "suffix_prefix_cache"
    assert loaded.args["decoder_arch"] == "suffix_prefix_cache"
    assert loaded.metrics["decoder_arch"] == "suffix_prefix_cache"
    assert action_chunk.shape == (4, 3)


def test_load_checkpoint_accepts_old_kwargs_without_decoder_arch(tmp_path) -> None:
    model = _tiny_model(decoder_arch="encoder")
    model_kwargs = _tiny_model_kwargs()
    model_kwargs.pop("decoder_arch")
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_kwargs": model_kwargs,
            "args": {},
            "metrics": {},
            "action_normalization": {"enabled": False},
        },
        checkpoint_path,
    )

    loaded = load_wan_pi05_action_expert_checkpoint(checkpoint_path, device="cpu")

    assert loaded.model.decoder_arch == "encoder"
    assert loaded.action_norm_mean is None
    assert loaded.action_norm_std is None


def test_load_checkpoint_rejects_malformed_checkpoint(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": {}}, checkpoint_path)

    with pytest.raises(ValueError, match="missing required key 'model_kwargs'"):
        load_wan_pi05_action_expert_checkpoint(checkpoint_path, device="cpu")


def test_load_checkpoint_rejects_malformed_action_normalization(tmp_path) -> None:
    model = _tiny_model()
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_kwargs": _tiny_model_kwargs(),
            "args": {},
            "metrics": {},
            "action_normalization": {"enabled": True, "mean": torch.zeros(3)},
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="missing 'std'"):
        load_wan_pi05_action_expert_checkpoint(checkpoint_path, device="cpu")


def test_wan_pi05_action_expert_cross_attention_output_shape() -> None:
    model = _tiny_model(decoder_arch="context_cross_attention")
    prefix_tokens, state, actions = _tiny_batch()
    time = torch.tensor([1.0, 0.25])

    velocity = model(prefix_tokens, state, actions, time)

    assert velocity.shape == actions.shape
    assert model.decoder_arch == "context_cross_attention"


def test_wan_pi05_action_expert_suffix_prefix_cache_output_shape() -> None:
    model = _tiny_model(decoder_arch="suffix_prefix_cache")
    prefix_tokens, state, actions = _tiny_batch()
    time = torch.tensor([1.0, 0.25])

    velocity = model(prefix_tokens, state, actions, time)

    assert velocity.shape == actions.shape
    assert model.decoder_arch == "suffix_prefix_cache"


def test_wan_pi05_action_expert_joint_softmax_prefix_cache_output_shape() -> None:
    model = _tiny_model(decoder_arch="joint_softmax_prefix_cache")
    prefix_tokens, state, actions = _tiny_batch()
    time = torch.tensor([1.0, 0.25])

    velocity = model(prefix_tokens, state, actions, time)
    memory = model.encode_prefix_memory(prefix_tokens, state)

    assert velocity.shape == actions.shape
    assert model.decoder_arch == "joint_softmax_prefix_cache"
    assert len(memory.keys) == 1
    assert memory.keys[0].shape == (2, 4, 4, 4)
    assert memory.values[0].shape == (2, 4, 4, 4)


def test_suffix_prefix_cache_full_forward_equals_cached_memory_forward() -> None:
    model = _tiny_model(decoder_arch="suffix_prefix_cache")
    model.eval()
    prefix_tokens, state, actions = _tiny_batch()
    time = torch.tensor([0.75, 0.25])

    full_velocity = model(prefix_tokens, state, actions, time)
    memory = model.encode_prefix_memory(prefix_tokens, state)
    cached_velocity = model.forward_with_prefix_memory(memory, actions, time)

    assert torch.allclose(full_velocity, cached_velocity, atol=1e-6)


def test_joint_softmax_prefix_cache_full_forward_equals_cached_memory_forward() -> None:
    model = _tiny_model(decoder_arch="joint_softmax_prefix_cache")
    model.eval()
    prefix_tokens, state, actions = _tiny_batch()
    time = torch.tensor([0.75, 0.25])

    full_velocity = model(prefix_tokens, state, actions, time)
    memory = model.encode_prefix_memory(prefix_tokens, state)
    cached_velocity = model.forward_with_prefix_memory(memory, actions, time)

    assert torch.allclose(full_velocity, cached_velocity, atol=1e-6)


def test_joint_softmax_prefix_cache_layer_matches_full_joint_attention_path() -> None:
    torch.manual_seed(113)
    model = _tiny_model(decoder_arch="joint_softmax_prefix_cache")
    model.eval()
    decoder = model.suffix_prefix_decoder
    layer = decoder.layers[0]
    prefix_tokens = torch.randn(1, 2, model.hidden_dim)
    action_tokens = torch.randn(1, 3, model.hidden_dim)

    _, memory_keys, memory_values = layer.encode_prefix(prefix_tokens)
    cached_output = layer(action_tokens, memory_keys, memory_values)
    full_joint_output = _manual_full_joint_softmax_layer_output(layer, prefix_tokens, action_tokens)

    assert torch.allclose(cached_output, full_joint_output, atol=1e-6)


def test_suffix_prefix_cache_memory_tracks_prefix_and_state_not_action_time() -> None:
    model = _tiny_model(decoder_arch="suffix_prefix_cache")
    model.eval()
    prefix_tokens, state, actions = _tiny_batch()
    time = torch.tensor([0.75, 0.25])

    memory = model.encode_prefix_memory(prefix_tokens, state)
    memory_key_snapshots = [key.detach().clone() for key in memory.keys]
    memory_value_snapshots = [value.detach().clone() for value in memory.values]
    output = model.forward_with_prefix_memory(memory, actions, time)
    changed_prefix_memory = model.encode_prefix_memory(prefix_tokens + 0.5, state)
    changed_state_memory = model.encode_prefix_memory(prefix_tokens, state + 0.5)
    changed_prefix_output = model.forward_with_prefix_memory(changed_prefix_memory, actions, time)
    changed_state_output = model.forward_with_prefix_memory(changed_state_memory, actions, time)

    changed_action_time_output = model.forward_with_prefix_memory(
        memory,
        actions + 0.5,
        torch.tensor([0.125, 0.875]),
    )

    assert any(
        not torch.allclose(key, changed_key) for key, changed_key in zip(memory.keys, changed_prefix_memory.keys)
    )
    assert any(not torch.allclose(key, changed_key) for key, changed_key in zip(memory.keys, changed_state_memory.keys))
    assert not torch.allclose(output, changed_prefix_output)
    assert not torch.allclose(output, changed_state_output)
    assert not torch.allclose(output, changed_action_time_output)
    for actual, expected in zip(memory.keys, memory_key_snapshots, strict=True):
        assert torch.allclose(actual, expected)
    for actual, expected in zip(memory.values, memory_value_snapshots, strict=True):
        assert torch.allclose(actual, expected)


def test_encoder_context_outputs_do_not_depend_on_action_tokens() -> None:
    model = _tiny_model(decoder_arch="encoder")
    context_tokens = [
        torch.tensor([[[1.0, 0.0], [0.0, 2.0]]]),
        torch.tensor([[[3.0, 4.0]]]),
    ]
    context_len = sum(tokens.shape[1] for tokens in context_tokens)
    encoder = _MaskApplyingSpyEncoder(context_len=context_len)
    model.encoder = encoder
    action_tokens = torch.tensor([[[5.0, 7.0], [11.0, 13.0]]])
    changed_action_tokens = action_tokens + 100.0

    model._encode_with_prefix_encoder(context_tokens, action_tokens, device=action_tokens.device)
    model._encode_with_prefix_encoder(context_tokens, changed_action_tokens, device=action_tokens.device)

    assert bool(torch.all(encoder.masks[0][:context_len, context_len:]))
    assert torch.allclose(encoder.context_outputs[0], encoder.context_outputs[1])


def test_cross_attention_encoded_context_is_independent_from_action_and_time_inputs() -> None:
    model = _tiny_model(decoder_arch="context_cross_attention")
    model.eval()
    context_encoder = _RecordingContextEncoder()
    decoder = _RecordingDecoder()
    model.context_encoder = context_encoder
    model.decoder = decoder
    prefix_tokens, state, actions = _tiny_batch()

    model(prefix_tokens, state, actions, torch.tensor([1.0, 0.25]))
    model(prefix_tokens, state, actions + 100.0, torch.tensor([0.0, 0.75]))

    assert torch.allclose(context_encoder.inputs[0], context_encoder.inputs[1])
    assert torch.allclose(context_encoder.outputs[0], context_encoder.outputs[1])
    assert torch.allclose(decoder.memories[0], decoder.memories[1])


def test_default_timestep_conditioning_is_additive() -> None:
    torch.manual_seed(101)
    default_model = _tiny_model()
    torch.manual_seed(101)
    explicit_model = _tiny_model(timestep_conditioning="additive")
    default_model.eval()
    explicit_model.eval()
    prefix_tokens, state, actions = _tiny_batch()
    time = torch.tensor([1.0, 0.25])

    default_velocity = default_model(prefix_tokens, state, actions, time)
    explicit_velocity = explicit_model(prefix_tokens, state, actions, time)

    assert default_model.timestep_conditioning == "additive"
    assert "timestep_film.weight" not in default_model.state_dict()
    assert torch.allclose(default_velocity, explicit_velocity)


def test_wan_pi05_action_expert_film_timestep_output_shape() -> None:
    model = _tiny_model(timestep_conditioning="film", decoder_arch="context_cross_attention")
    prefix_tokens, state, actions = _tiny_batch()
    time = torch.tensor([1.0, 0.25])

    velocity = model(prefix_tokens, state, actions, time)

    assert velocity.shape == actions.shape
    assert model.timestep_conditioning == "film"
    assert model.decoder_arch == "context_cross_attention"


def test_film_timestep_conditioning_changes_output_vs_additive() -> None:
    torch.manual_seed(103)
    additive_model = _tiny_model(timestep_conditioning="additive")
    torch.manual_seed(103)
    film_model = _tiny_model(timestep_conditioning="film")
    additive_model.eval()
    film_model.eval()
    prefix_tokens, state, actions = _tiny_batch()
    time = torch.tensor([0.75, 0.25])

    for name, value in additive_model.state_dict().items():
        assert torch.allclose(value, film_model.state_dict()[name])
    additive_velocity = additive_model(prefix_tokens, state, actions, time)
    film_velocity = film_model(prefix_tokens, state, actions, time)

    assert not torch.allclose(additive_velocity, film_velocity)


def test_default_timestep_embedding_style_is_diffusion() -> None:
    torch.manual_seed(211)
    default_model = _tiny_model()
    torch.manual_seed(211)
    explicit_model = _tiny_model(timestep_embedding_style="diffusion")
    default_model.eval()
    explicit_model.eval()
    prefix_tokens, state, actions = _tiny_batch()
    time = torch.tensor([1.0, 0.25])

    default_velocity = default_model(prefix_tokens, state, actions, time)
    explicit_velocity = explicit_model(prefix_tokens, state, actions, time)

    assert default_model.timestep_embedding_style == "diffusion"
    assert default_model.time_mlp[0].style == "diffusion"
    assert torch.allclose(default_velocity, explicit_velocity)


def test_invalid_timestep_embedding_style_errors_clearly() -> None:
    with pytest.raises(ValueError, match="timestep_embedding_style must be 'diffusion' or 'pi05'.*bogus"):
        _tiny_model(timestep_embedding_style="bogus")
    with pytest.raises(ValueError, match="timestep_embedding_style must be 'diffusion' or 'pi05'.*bogus"):
        SinusoidalTimeEmbedding(16, style="bogus")


def test_pi05_timestep_embedding_requires_even_dim() -> None:
    with pytest.raises(ValueError, match="pi05.*even dim.*15"):
        SinusoidalTimeEmbedding(15, style="pi05")


def test_pi05_timestep_embedding_matches_openpi_posemb_and_is_deterministic() -> None:
    dim = 16
    embedding = SinusoidalTimeEmbedding(dim, style="pi05")
    time = torch.tensor([0.0, 0.1, 0.5, 1.0])

    out = embedding(time)
    repeat = embedding(time)

    fraction = torch.linspace(0.0, 1.0, dim // 2)
    period = 4e-3 * (4.0 / 4e-3) ** fraction
    sinusoid_input = (2.0 * math.pi / period)[None, :] * time[:, None]
    expected = torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1)

    assert out.shape == (4, dim)
    assert torch.allclose(out, expected, atol=1e-6)
    assert torch.equal(out, repeat)


def test_pi05_timestep_embedding_low_precision_uses_float32_angles() -> None:
    embedding = SinusoidalTimeEmbedding(16, style="pi05")
    time = torch.tensor([0.1, 0.5, 1.0], dtype=torch.bfloat16)

    actual = embedding(time)
    expected = embedding(time.to(dtype=torch.float32)).to(dtype=torch.bfloat16)

    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual, expected)


def test_pi05_timestep_embedding_changes_faster_over_unit_interval() -> None:
    dim = 16
    diffusion = SinusoidalTimeEmbedding(dim, style="diffusion")
    pi05 = SinusoidalTimeEmbedding(dim, style="pi05")
    # Sample finely so the high-frequency pi0.5 components are not aliased.
    time = torch.linspace(0.0, 1.0, 4096)

    diffusion_variation = diffusion(time).diff(dim=0).abs().sum()
    pi05_variation = pi05(time).diff(dim=0).abs().sum()

    assert pi05_variation > 10.0 * diffusion_variation


def test_pi05_timestep_embedding_style_changes_model_output_vs_diffusion() -> None:
    torch.manual_seed(213)
    diffusion_model = _tiny_model(timestep_embedding_style="diffusion")
    torch.manual_seed(213)
    pi05_model = _tiny_model(timestep_embedding_style="pi05")
    diffusion_model.eval()
    pi05_model.eval()
    prefix_tokens, state, actions = _tiny_batch()
    time = torch.tensor([0.75, 0.25])

    # Same trainable weights: only the (non-persistent) time-embedding frequencies differ.
    for name, value in diffusion_model.state_dict().items():
        assert torch.allclose(value, pi05_model.state_dict()[name])
    diffusion_velocity = diffusion_model(prefix_tokens, state, actions, time)
    pi05_velocity = pi05_model(prefix_tokens, state, actions, time)

    assert pi05_model.timestep_embedding_style == "pi05"
    assert not torch.allclose(diffusion_velocity, pi05_velocity)


def test_old_model_kwargs_without_timestep_embedding_style_load_diffusion() -> None:
    torch.manual_seed(217)
    diffusion_model = _tiny_model(timestep_embedding_style="diffusion")
    diffusion_model.eval()
    old_model_kwargs = _tiny_model_kwargs()
    old_model_kwargs.pop("timestep_embedding_style")
    prefix_tokens, state, actions = _tiny_batch()
    time = torch.tensor([1.0, 0.25])

    restored_model = WanPi05ActionExpert(**old_model_kwargs)
    restored_model.load_state_dict(diffusion_model.state_dict())
    restored_model.eval()
    restored_velocity = restored_model(prefix_tokens, state, actions, time)
    diffusion_velocity = diffusion_model(prefix_tokens, state, actions, time)

    assert "timestep_embedding_style" not in old_model_kwargs
    assert restored_model.timestep_embedding_style == "diffusion"
    assert torch.allclose(restored_velocity, diffusion_velocity)


def test_wan_pi05_action_expert_prefix_only_output_shape() -> None:
    model = _tiny_model(conditioning_mode="wan_prefix", decoder_arch="context_cross_attention")
    prefix_tokens, state, actions = _tiny_batch()
    time = torch.tensor([1.0, 0.25])

    velocity = model(prefix_tokens, state, actions, time)

    assert velocity.shape == actions.shape
    assert model.conditioning_mode == "wan_prefix"
    assert model.condition_on_state is False
    assert model.decoder_arch == "context_cross_attention"


def test_invalid_timestep_conditioning_errors_clearly() -> None:
    with pytest.raises(ValueError, match="timestep_conditioning must be 'additive' or 'film'.*bogus"):
        _tiny_model(timestep_conditioning="bogus")


def test_invalid_decoder_arch_errors_clearly() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "decoder_arch must be one of 'encoder', 'context_cross_attention', 'suffix_prefix_cache', "
            "'joint_softmax_prefix_cache'.*bogus"
        ),
    ):
        _tiny_model(decoder_arch="bogus")


def test_cross_attention_decoder_validates_input_shapes() -> None:
    model = _tiny_model(decoder_arch="context_cross_attention")
    prefix_tokens, state, actions = _tiny_batch()
    time = torch.tensor([1.0, 0.25])

    with pytest.raises(ValueError, match="prefix_tokens last dim"):
        model(prefix_tokens[..., :7], state, actions, time)
    with pytest.raises(ValueError, match="state must have shape"):
        model(prefix_tokens, state[:1], actions, time)
    with pytest.raises(ValueError, match="noisy_actions must have shape"):
        model(prefix_tokens, state, actions[:, :3], time)
    with pytest.raises(ValueError, match="time must have shape"):
        model(prefix_tokens, state, actions, time.view(2, 1))


def test_prefix_only_mode_ignores_state_values() -> None:
    model = _tiny_model(conditioning_mode="wan_prefix")
    model.eval()
    prefix_tokens, state, actions = _tiny_batch()
    changed_state = state + torch.randn_like(state) * 100.0
    time = torch.tensor([0.75, 0.25])

    velocity_a = model(prefix_tokens, state, actions, time)
    velocity_b = model(prefix_tokens, changed_state, actions, time)

    assert torch.allclose(velocity_a, velocity_b)


def test_default_conditioning_mode_uses_state_values() -> None:
    model = _tiny_model()
    model.eval()
    prefix_tokens, state, actions = _tiny_batch()
    changed_state = state + torch.randn_like(state) * 100.0
    time = torch.tensor([0.75, 0.25])

    velocity_a = model(prefix_tokens, state, actions, time)
    velocity_b = model(prefix_tokens, changed_state, actions, time)

    assert model.conditioning_mode == "wan_prefix_state"
    assert model.condition_on_state is True
    assert not torch.allclose(velocity_a, velocity_b)


def test_flow_matching_loss_is_finite_and_has_gradients() -> None:
    model = _tiny_model()
    prefix_tokens, state, actions = _tiny_batch()
    action_mask = torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]])
    generator = torch.Generator().manual_seed(17)

    loss = flow_matching_loss(model, prefix_tokens, state, actions, action_mask, generator=generator)
    loss.backward()

    assert torch.isfinite(loss)
    grad_norm = sum(
        parameter.grad.detach().abs().sum() for parameter in model.parameters() if parameter.grad is not None
    )
    assert grad_norm > 0


def test_masked_action_mse_default_matches_unweighted_loss() -> None:
    predicted = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [5.0, 7.0, 11.0]],
            [[13.0, 17.0, 19.0], [23.0, 29.0, 31.0]],
        ]
    )
    target = torch.ones_like(predicted)
    action_mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])

    loss = masked_action_mse(predicted, target, action_mask)

    mask = action_mask.unsqueeze(-1).expand_as(predicted)
    expected = ((predicted - target).square() * mask).sum() / mask.sum()
    assert float(loss) == pytest.approx(float(expected))


def test_masked_action_mse_applies_per_action_dimension_weights() -> None:
    predicted = torch.tensor([[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]])
    target = torch.zeros_like(predicted)
    action_mask = torch.tensor([[1.0, 0.0]])
    action_weights = torch.tensor([2.0, 3.0, 5.0])

    loss = masked_action_mse(predicted, target, action_mask, action_weights=action_weights)

    expected = (1.0**2 * 2.0 + 2.0**2 * 3.0 + 3.0**2 * 5.0) / 3.0
    assert float(loss) == pytest.approx(expected)


def test_masked_action_mse_per_sample_reports_means_but_scalar_stays_element_weighted() -> None:
    predicted = torch.tensor(
        [
            [[1.0, 3.0], [10.0, 10.0], [10.0, 10.0]],
            [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]],
        ]
    )
    target = torch.zeros_like(predicted)
    action_mask = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    action_weights = torch.tensor([1.0, 2.0])

    per_sample = masked_action_mse_per_sample(predicted, target, action_mask, action_weights=action_weights)
    numerator, count = masked_action_mse_per_sample_parts(
        predicted,
        target,
        action_mask,
        action_weights=action_weights,
    )
    scalar = masked_action_mse(predicted, target, action_mask, action_weights=action_weights)

    expected_numerator = torch.tensor(
        [
            1.0**2 + 3.0**2 * 2.0,
            2.0**2 + 4.0**2 * 2.0 + 6.0**2 + 8.0**2 * 2.0 + 10.0**2 + 12.0**2 * 2.0,
        ]
    )
    expected_count = torch.tensor([2.0, 6.0])
    assert torch.allclose(numerator, expected_numerator)
    assert torch.allclose(count, expected_count)
    assert torch.allclose(per_sample, expected_numerator / expected_count)
    assert float(scalar) == pytest.approx(float(expected_numerator.sum() / expected_count.sum()))
    assert float(scalar) != pytest.approx(float(per_sample.mean()))


def test_flow_matching_loss_scalar_matches_total_valid_element_mean_with_uneven_masks() -> None:
    model = _tiny_model()
    prefix_tokens, state, actions = _tiny_batch()
    action_mask = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]])
    action_weights = torch.tensor([1.0, 3.0, 5.0])

    scalar = flow_matching_loss(
        model,
        prefix_tokens,
        state,
        actions,
        action_mask,
        generator=torch.Generator().manual_seed(123),
        action_weights=action_weights,
    )
    per_sample, numerator, count = flow_matching_loss_per_sample_parts(
        model,
        prefix_tokens,
        state,
        actions,
        action_mask,
        generator=torch.Generator().manual_seed(123),
        action_weights=action_weights,
    )

    assert torch.allclose(per_sample, numerator / count)
    assert float(scalar) == pytest.approx(float(numerator.sum() / count.sum()))


def test_task_balanced_action_loss_uses_task_valid_element_means() -> None:
    numerator = torch.tensor([10.0, 100.0, 30.0])
    count = torch.tensor([2.0, 10.0, 3.0])
    per_sample = numerator / count
    tasks = ["a", "a", "b"]

    loss = _aggregate_action_loss(
        per_sample,
        numerator,
        count,
        tasks=tasks,
        aggregation="task_balanced",
        task_cvar_fraction=0.25,
        task_cvar_weight=0.5,
    )

    task_a = (10.0 + 100.0) / (2.0 + 10.0)
    task_b = 30.0 / 3.0
    assert float(loss) == pytest.approx((task_a + task_b) / 2.0)
    assert float(loss) != pytest.approx(float(per_sample.mean()))


def test_task_cvar_action_loss_uses_top_task_valid_element_means() -> None:
    numerator = torch.tensor([9.0, 30.0, 100.0])
    count = torch.tensor([3.0, 5.0, 10.0])
    per_sample = numerator / count
    tasks = ["a", "b", "c"]

    loss = _aggregate_action_loss(
        per_sample,
        numerator,
        count,
        tasks=tasks,
        aggregation="task_cvar",
        task_cvar_fraction=0.5,
        task_cvar_weight=0.25,
    )

    task_means = torch.tensor([3.0, 6.0, 10.0])
    expected = task_means.mean() + 0.25 * torch.tensor([10.0, 6.0]).mean()
    assert float(loss) == pytest.approx(float(expected))


def test_effective_task_cvar_weight_uses_final_when_schedule_disabled() -> None:
    assert _effective_task_cvar_weight(
        0,
        start_weight=None,
        final_weight=0.25,
        warmup_epochs=10,
    ) == pytest.approx(0.25)
    assert _effective_task_cvar_weight(
        8,
        start_weight=None,
        final_weight=0.25,
        warmup_epochs=10,
    ) == pytest.approx(0.25)


def test_effective_task_cvar_weight_linearly_warms_to_final() -> None:
    kwargs = {"start_weight": 0.0, "final_weight": 0.25, "warmup_epochs": 10}

    assert _effective_task_cvar_weight(0, **kwargs) == pytest.approx(0.0)
    assert _effective_task_cvar_weight(5, **kwargs) == pytest.approx(0.125)
    assert _effective_task_cvar_weight(10, **kwargs) == pytest.approx(0.25)
    assert _effective_task_cvar_weight(11, **kwargs) == pytest.approx(0.25)
    assert _effective_task_cvar_weight(
        0,
        start_weight=0.0,
        final_weight=0.25,
        warmup_epochs=0,
    ) == pytest.approx(0.25)


def test_task_tail_action_loss_rejects_missing_tasks_and_zero_valid_task() -> None:
    per_sample = torch.tensor([1.0, 2.0])
    numerator = torch.tensor([1.0, 0.0])
    count = torch.tensor([1.0, 0.0])

    with pytest.raises(ValueError, match="exactly one task label"):
        _aggregate_action_loss(
            per_sample,
            numerator,
            count,
            tasks=["a"],
            aggregation="task_balanced",
            task_cvar_fraction=0.25,
            task_cvar_weight=0.5,
        )
    with pytest.raises(ValueError, match="zero valid elements"):
        _aggregate_action_loss(
            per_sample,
            numerator,
            count,
            tasks=["a", "b"],
            aggregation="task_cvar",
            task_cvar_fraction=0.25,
            task_cvar_weight=0.5,
        )
    with pytest.raises(ValueError, match="non-empty task labels"):
        _aggregate_action_loss(
            per_sample,
            numerator + 1.0,
            count + 1.0,
            tasks=["a", ""],
            aggregation="task_balanced",
            task_cvar_fraction=0.25,
            task_cvar_weight=0.5,
        )


def test_sample_actions_is_deterministic_with_explicit_noise() -> None:
    model = WanPi05ActionExpert(
        prefix_dim=8,
        state_dim=5,
        action_dim=3,
        action_horizon=4,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.5,
    )
    model.train()
    prefix_tokens, state, actions = _tiny_batch()
    explicit_noise = torch.full_like(actions, 0.125)

    sample_a = sample_actions(model, prefix_tokens, state, num_steps=3, noise=explicit_noise)
    sample_b = sample_actions(model, prefix_tokens, state, num_steps=3, noise=explicit_noise)

    assert torch.allclose(sample_a, sample_b)
    assert model.training


@pytest.mark.parametrize(
    "decoder_arch",
    ["encoder", "context_cross_attention", "suffix_prefix_cache", "joint_softmax_prefix_cache"],
)
def test_sample_actions_prepares_context_once_and_steps_many(
    monkeypatch: pytest.MonkeyPatch,
    decoder_arch: str,
) -> None:
    model = _tiny_model(decoder_arch=decoder_arch)
    model.train()
    prefix_tokens, state, actions = _tiny_batch()
    explicit_noise = torch.full_like(actions, 0.125)
    original_prepare_action_context = model.prepare_action_context
    original_forward_with_action_context = model.forward_with_action_context
    prepare_calls = 0
    step_calls = 0

    def recording_prepare_action_context(prefix: torch.Tensor, current_state: torch.Tensor):
        nonlocal prepare_calls
        prepare_calls += 1
        return original_prepare_action_context(prefix, current_state)

    def recording_forward_with_action_context(
        context: ActionDenoisingContext, noisy_actions: torch.Tensor, time: torch.Tensor
    ):
        nonlocal step_calls
        step_calls += 1
        return original_forward_with_action_context(context, noisy_actions, time)

    monkeypatch.setattr(model, "prepare_action_context", recording_prepare_action_context)
    monkeypatch.setattr(model, "forward_with_action_context", recording_forward_with_action_context)

    sample = sample_actions(model, prefix_tokens, state, num_steps=3, noise=explicit_noise)

    assert sample.shape == actions.shape
    assert prepare_calls == 1
    assert step_calls == 3
    assert model.training


@pytest.mark.parametrize(
    "decoder_arch",
    ["encoder", "context_cross_attention", "suffix_prefix_cache", "joint_softmax_prefix_cache"],
)
def test_forward_with_prepared_action_context_matches_forward(decoder_arch: str) -> None:
    model = _tiny_model(decoder_arch=decoder_arch).eval()
    prefix_tokens, state, actions = _tiny_batch()
    time = torch.full((actions.shape[0],), 0.5)

    direct = model(prefix_tokens, state, actions, time)
    context = model.prepare_action_context(prefix_tokens, state)
    with_context = model.forward_with_action_context(context, actions, time)

    assert torch.allclose(with_context, direct)
    assert context.decoder_arch == decoder_arch
    assert context.batch_size == actions.shape[0]


def test_context_cross_attention_prepare_context_encodes_context_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny_model(decoder_arch="context_cross_attention")
    prefix_tokens, state, actions = _tiny_batch()
    explicit_noise = torch.full_like(actions, 0.125)
    original_forward = model.context_encoder.forward
    encode_calls = 0

    def recording_forward(*args, **kwargs):
        nonlocal encode_calls
        encode_calls += 1
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model.context_encoder, "forward", recording_forward)

    sample_actions(model, prefix_tokens, state, num_steps=4, noise=explicit_noise)

    assert encode_calls == 1


def test_action_context_rejects_decoder_arch_mismatch() -> None:
    encoder_model = _tiny_model(decoder_arch="encoder")
    prefix_model = _tiny_model(decoder_arch="suffix_prefix_cache")
    prefix_tokens, state, actions = _tiny_batch()
    time = torch.full((actions.shape[0],), 0.5)
    context = encoder_model.prepare_action_context(prefix_tokens, state)

    with pytest.raises(ValueError, match="decoder_arch"):
        prefix_model.forward_with_action_context(context, actions, time)


def test_action_denoising_context_is_not_native_wan_kv_contract() -> None:
    assert "not native Wan attention KV" in (ActionDenoisingContext.__doc__ or "")
    assert "keys" not in ActionDenoisingContext.__dataclass_fields__
    assert "values" not in ActionDenoisingContext.__dataclass_fields__


@pytest.mark.parametrize("decoder_arch", ["suffix_prefix_cache", "joint_softmax_prefix_cache"])
def test_forward_with_prefix_memory_remains_compatible(decoder_arch: str) -> None:
    model = _tiny_model(decoder_arch=decoder_arch).eval()
    prefix_tokens, state, actions = _tiny_batch()
    time = torch.full((actions.shape[0],), 0.5)
    memory = model.encode_prefix_memory(prefix_tokens, state)

    with_memory = model.forward_with_prefix_memory(memory, actions, time)
    context = model.prepare_action_context(prefix_tokens, state)
    with_context = model.forward_with_action_context(context, actions, time)

    assert torch.allclose(with_memory, with_context)


def test_pi05_time_convention_interpolates_noise_at_one_and_data_at_zero() -> None:
    actions = torch.tensor([[[1.0, 2.0]], [[3.0, 5.0]], [[7.0, 11.0]]])
    noise = torch.tensor([[[13.0, 17.0]], [[19.0, 23.0]], [[29.0, 31.0]]])
    time = torch.tensor([0.0, 1.0, 0.25])

    noisy_actions, target_velocity = pi05_flow_targets(actions, noise, time)

    assert torch.allclose(noisy_actions[0], actions[0])
    assert torch.allclose(noisy_actions[1], noise[1])
    assert torch.allclose(noisy_actions[2], 0.25 * noise[2] + 0.75 * actions[2])
    assert torch.allclose(target_velocity, noise - actions)


def test_fake_prefix_cache_is_current_only(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    paths = write_fake_prefix_cache(
        cache_dir,
        num_rows=3,
        prefix_tokens=2,
        prefix_dim=8,
        state_dim=5,
        action_horizon=4,
        action_dim=3,
        seed=23,
    )

    dataset = load_cached_prefix_dataset(cache_dir)
    sample = dataset[0]
    raw_row = torch.load(paths[0], weights_only=False)
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))

    assert len(dataset) == 3
    assert sample["prefix_tokens"].shape == (2, 8)
    assert sample["state"].shape == (5,)
    assert sample["actions"].shape == (4, 3)
    assert sample["action_mask"].shape == (4,)
    assert not (set(raw_row) & FUTURE_LEAKAGE_KEYS)
    assert manifest["contains_future_images"] is False
    assert manifest["contains_future_latents"] is False


def test_cache_row_rejects_future_leakage_keys(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    torch.save(
        {
            "prefix_tokens": torch.zeros(2, 8),
            "state": torch.zeros(5),
            "actions": torch.zeros(4, 3),
            "future_images": torch.zeros(1, 3, 16, 16),
        },
        cache_dir / "bad.pt",
    )
    dataset = CachedWanPrefixActionDataset(cache_dir)

    with pytest.raises(ValueError, match="current-only.*future_images"):
        _ = dataset[0]


def test_missing_real_wan_prefix_cache_errors_clearly(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Real Wan prefix caching.*does not run Wan2.2 5B inference"):
        load_cached_prefix_dataset(tmp_path / "missing", real_wan_prefix_cache=True)


def test_train_eval_smoke_saves_metrics_and_checkpoint(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    write_fake_prefix_cache(
        cache_dir,
        num_rows=6,
        prefix_tokens=2,
        prefix_dim=8,
        state_dim=5,
        action_horizon=4,
        action_dim=3,
        seed=31,
    )
    args = Args(
        cache_path=str(cache_dir),
        output_dir=str(output_dir),
        epochs=1,
        batch_size=2,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        sample_steps=2,
        seed=31,
    )

    metrics = run_train_eval(args)

    assert (output_dir / "checkpoint.pt").exists()
    assert (output_dir / "metrics.json").exists()
    assert metrics["num_train"] == 4
    assert metrics["num_val"] == 2
    assert "eval_cache_path" not in metrics
    assert metrics["val_model_sample_mse"] >= 0.0
    assert metrics["val_model_zero_noise_mse"] == pytest.approx(metrics["val_model_sample_mse"])
    assert metrics["val_mean_action_mse"] >= 0.0
    assert metrics["normalize_actions"] is False
    assert metrics["action_loss_weighting"] == "none"
    assert metrics["action_loss_weights"] == pytest.approx([1.0, 1.0, 1.0])
    assert metrics["action_loss_weights_source"] == "ones"
    assert metrics["action_loss_aggregation"] == "mean"
    assert metrics["task_cvar_fraction"] == 0.25
    assert metrics["task_cvar_weight"] == 0.5
    assert metrics["task_cvar_start_weight"] is None
    assert metrics["task_cvar_warmup_epochs"] == 0
    assert metrics["task_cvar_schedule_enabled"] is False
    assert metrics["task_cvar_final_effective_weight"] == pytest.approx(0.5)
    assert metrics["conditioning_mode"] == "wan_prefix_state"
    assert metrics["timestep_conditioning"] == "additive"
    assert metrics["timestep_embedding_style"] == "diffusion"
    assert metrics["decoder_arch"] == "encoder"
    assert len(metrics["val_model_zero_noise_mse_per_action_dim"]) == 3
    assert len(metrics["val_mean_action_mse_per_action_dim"]) == 3

    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert checkpoint["action_normalization"] == {"enabled": False, "scope": "global"}
    assert checkpoint["action_loss"]["weighting"] == "none"
    assert torch.allclose(checkpoint["action_loss"]["weights"], torch.ones(3))
    assert checkpoint["action_loss"]["weights_source"] == "ones"
    assert checkpoint["action_loss"]["aggregation"] == "mean"
    assert checkpoint["action_loss"]["task_cvar_fraction"] == 0.25
    assert checkpoint["action_loss"]["task_cvar_weight"] == 0.5
    assert checkpoint["action_loss"]["task_cvar_start_weight"] is None
    assert checkpoint["action_loss"]["task_cvar_warmup_epochs"] == 0
    assert checkpoint["action_loss"]["task_cvar_schedule_enabled"] is False
    assert checkpoint["action_loss"]["task_cvar_final_effective_weight"] == pytest.approx(0.5)
    assert checkpoint["args"]["action_loss_weighting"] == "none"
    assert checkpoint["args"]["action_loss_aggregation"] == "mean"
    assert checkpoint["args"]["task_cvar_start_weight"] is None
    assert checkpoint["args"]["task_cvar_warmup_epochs"] == 0
    assert checkpoint["args"]["decoder_arch"] == "encoder"
    assert checkpoint["model_kwargs"]["conditioning_mode"] == "wan_prefix_state"
    assert checkpoint["model_kwargs"]["timestep_conditioning"] == "additive"
    assert checkpoint["model_kwargs"]["timestep_embedding_style"] == "diffusion"
    assert checkpoint["model_kwargs"]["decoder_arch"] == "encoder"


def test_train_eval_init_checkpoint_strict_loads_source_weights_and_records_metadata(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    write_fake_prefix_cache(
        cache_dir,
        num_rows=6,
        prefix_tokens=2,
        prefix_dim=8,
        state_dim=5,
        action_horizon=4,
        action_dim=3,
        seed=231,
    )
    source_args = Args(
        cache_path=str(cache_dir),
        output_dir=str(source_dir),
        epochs=0,
        batch_size=2,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        sample_steps=2,
        device="cpu",
        seed=231,
    )
    source_metrics = run_train_eval(source_args)
    source_checkpoint_path = source_dir / "checkpoint.pt"
    source_checkpoint = torch.load(source_checkpoint_path, map_location="cpu", weights_only=False)

    target_args = Args(
        cache_path=str(cache_dir),
        output_dir=str(target_dir),
        init_checkpoint=str(source_checkpoint_path),
        epochs=0,
        batch_size=2,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        sample_steps=2,
        device="cpu",
        seed=999,
    )

    target_metrics = run_train_eval(target_args)

    target_checkpoint = torch.load(target_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert source_metrics["init_checkpoint"] is None
    assert target_metrics["init_checkpoint"] == str(source_checkpoint_path)
    assert target_checkpoint["args"]["init_checkpoint"] == str(source_checkpoint_path)
    assert target_checkpoint["metrics"]["init_checkpoint"] == str(source_checkpoint_path)
    assert target_checkpoint["init_checkpoint"] == str(source_checkpoint_path)
    assert torch.allclose(
        target_checkpoint["model_state_dict"]["prefix_projection.0.weight"],
        source_checkpoint["model_state_dict"]["prefix_projection.0.weight"],
    )


def test_train_eval_init_checkpoint_rejects_architecture_mismatch(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    write_fake_prefix_cache(
        cache_dir,
        num_rows=6,
        prefix_tokens=2,
        prefix_dim=8,
        state_dim=5,
        action_horizon=4,
        action_dim=3,
        seed=232,
    )
    run_train_eval(
        Args(
            cache_path=str(cache_dir),
            output_dir=str(source_dir),
            epochs=0,
            batch_size=2,
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
            sample_steps=2,
            device="cpu",
            seed=232,
        )
    )

    with pytest.raises(ValueError, match="architecture/model_kwargs mismatch.*hidden_dim"):
        run_train_eval(
            Args(
                cache_path=str(cache_dir),
                output_dir=str(target_dir),
                init_checkpoint=str(source_dir / "checkpoint.pt"),
                epochs=0,
                batch_size=2,
                hidden_dim=32,
                num_layers=1,
                num_heads=4,
                sample_steps=2,
                device="cpu",
                seed=233,
            )
        )


def test_train_eval_init_checkpoint_rejects_action_normalization_mismatch(tmp_path) -> None:
    source_cache_dir = tmp_path / "source_cache"
    target_cache_dir = tmp_path / "target_cache"
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    for cache_dir, value, seed in ((source_cache_dir, 1.0, 241), (target_cache_dir, 2.0, 242)):
        write_fake_prefix_cache(
            cache_dir,
            num_rows=6,
            prefix_tokens=2,
            prefix_dim=8,
            state_dim=5,
            action_horizon=4,
            action_dim=3,
            seed=seed,
        )
        _set_cache_actions(cache_dir, value)
    run_train_eval(
        Args(
            cache_path=str(source_cache_dir),
            output_dir=str(source_dir),
            epochs=0,
            batch_size=2,
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
            sample_steps=2,
            normalize_actions=True,
            device="cpu",
            seed=241,
        )
    )

    with pytest.raises(ValueError, match="action_normalization mismatch for mean"):
        run_train_eval(
            Args(
                cache_path=str(target_cache_dir),
                output_dir=str(target_dir),
                init_checkpoint=str(source_dir / "checkpoint.pt"),
                epochs=0,
                batch_size=2,
                hidden_dim=16,
                num_layers=1,
                num_heads=4,
                sample_steps=2,
                normalize_actions=True,
                device="cpu",
                seed=242,
            )
        )


def test_init_checkpoint_normalization_compare_accepts_tensor_metadata_on_any_device(tmp_path) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    checkpoint_value = torch.tensor([1.0, 2.0, 3.0], device=device)
    current_value = torch.tensor([1.0, 2.0, 3.0], device="cpu")

    _allclose_or_mismatch(
        checkpoint_value,
        current_value,
        field="mean",
        checkpoint_path=tmp_path / "checkpoint.pt",
    )


def test_validate_init_checkpoint_action_normalization_defaults_missing_scope_to_global(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    metadata = {
        "enabled": True,
        "mean": torch.tensor([0.0, 1.0, 2.0]),
        "std": torch.tensor([1.0, 2.0, 3.0]),
    }

    _validate_init_checkpoint_action_normalization(
        {"action_normalization": metadata},
        checkpoint_path=checkpoint_path,
        normalize_actions=True,
        action_normalization_scope="global",
        action_norm_mean=torch.tensor([0.0, 1.0, 2.0]),
        action_norm_std=torch.tensor([1.0, 2.0, 3.0]),
        action_norm_by_task=None,
    )

    with pytest.raises(ValueError, match="action_normalization.scope mismatch"):
        _validate_init_checkpoint_action_normalization(
            {"action_normalization": metadata},
            checkpoint_path=checkpoint_path,
            normalize_actions=True,
            action_normalization_scope="per_task",
            action_norm_mean=None,
            action_norm_std=None,
            action_norm_by_task={"task": (torch.zeros(3), torch.ones(3))},
        )


def test_validate_init_checkpoint_missing_timestep_embedding_style_is_diffusion(tmp_path) -> None:
    """Old checkpoints predate timestep_embedding_style; absence must read as 'diffusion'."""
    checkpoint_path = tmp_path / "checkpoint.pt"
    old_model_kwargs = _tiny_model_kwargs()
    old_model_kwargs.pop("timestep_embedding_style")
    assert "timestep_embedding_style" not in old_model_kwargs

    diffusion_expected = _tiny_model_kwargs()
    pi05_expected = _tiny_model_kwargs()
    pi05_expected["timestep_embedding_style"] = "pi05"

    # Missing field is compatible with a current diffusion run (no error raised).
    _validate_init_checkpoint_model_kwargs(
        {"model_kwargs": old_model_kwargs},
        checkpoint_path=checkpoint_path,
        expected=diffusion_expected,
    )

    # Missing field defaults to diffusion, so a current pi05 run is a real mismatch.
    with pytest.raises(ValueError, match="architecture/model_kwargs mismatch.*timestep_embedding_style"):
        _validate_init_checkpoint_model_kwargs(
            {"model_kwargs": old_model_kwargs},
            checkpoint_path=checkpoint_path,
            expected=pi05_expected,
        )

    # An explicit value that disagrees still errors (here checkpoint pi05 vs current diffusion).
    explicit_pi05_kwargs = _tiny_model_kwargs()
    explicit_pi05_kwargs["timestep_embedding_style"] = "pi05"
    with pytest.raises(ValueError, match="architecture/model_kwargs mismatch.*timestep_embedding_style"):
        _validate_init_checkpoint_model_kwargs(
            {"model_kwargs": explicit_pi05_kwargs},
            checkpoint_path=checkpoint_path,
            expected=diffusion_expected,
        )


def test_train_eval_smoke_records_task_cvar_action_loss_settings(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    write_fake_prefix_cache(
        cache_dir,
        num_rows=9,
        prefix_tokens=2,
        prefix_dim=8,
        state_dim=5,
        action_horizon=4,
        action_dim=3,
        seed=131,
    )
    args = Args(
        cache_path=str(cache_dir),
        output_dir=str(output_dir),
        epochs=1,
        batch_size=6,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        sample_steps=2,
        action_loss_aggregation="task_cvar",
        task_cvar_fraction=0.5,
        task_cvar_weight=0.75,
        task_cvar_start_weight=0.25,
        task_cvar_warmup_epochs=4,
        device="cpu",
        seed=131,
    )

    metrics = run_train_eval(args)

    assert metrics["action_loss_aggregation"] == "task_cvar"
    assert metrics["task_cvar_fraction"] == 0.5
    assert metrics["task_cvar_weight"] == 0.75
    assert metrics["task_cvar_start_weight"] == 0.25
    assert metrics["task_cvar_warmup_epochs"] == 4
    assert metrics["task_cvar_schedule_enabled"] is True
    assert metrics["task_cvar_final_effective_weight"] == pytest.approx(0.25)
    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert checkpoint["args"]["action_loss_aggregation"] == "task_cvar"
    assert checkpoint["args"]["task_cvar_fraction"] == 0.5
    assert checkpoint["args"]["task_cvar_weight"] == 0.75
    assert checkpoint["args"]["task_cvar_start_weight"] == 0.25
    assert checkpoint["args"]["task_cvar_warmup_epochs"] == 4
    assert checkpoint["metrics"]["action_loss_aggregation"] == "task_cvar"
    assert checkpoint["action_loss"]["aggregation"] == "task_cvar"
    assert checkpoint["action_loss"]["task_cvar_fraction"] == 0.5
    assert checkpoint["action_loss"]["task_cvar_weight"] == 0.75
    assert checkpoint["action_loss"]["task_cvar_start_weight"] == 0.25
    assert checkpoint["action_loss"]["task_cvar_warmup_epochs"] == 4
    assert checkpoint["action_loss"]["task_cvar_schedule_enabled"] is True
    assert checkpoint["action_loss"]["task_cvar_final_effective_weight"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"action_loss_aggregation": "bogus"}, "action_loss_aggregation"),
        ({"task_cvar_fraction": 0.0}, "task_cvar_fraction"),
        ({"task_cvar_fraction": 1.5}, "task_cvar_fraction"),
        ({"task_cvar_weight": -0.1}, "task_cvar_weight"),
        ({"task_cvar_start_weight": -0.1}, "task_cvar_start_weight"),
        ({"task_cvar_warmup_epochs": -1}, "task_cvar_warmup_epochs"),
        ({"task_cvar_warmup_epochs": 1.5}, "task_cvar_warmup_epochs"),
    ],
)
def test_train_eval_rejects_invalid_task_tail_loss_settings(tmp_path, overrides, match) -> None:
    args = Args(
        cache_path=str(tmp_path / "cache"),
        output_dir=str(tmp_path / "output"),
        fake_cache=True,
        epochs=0,
        device="cpu",
        **overrides,
    )

    with pytest.raises(ValueError, match=match):
        run_train_eval(args)


def test_train_eval_propagates_consistent_wan_action_mode(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    write_fake_prefix_cache(
        cache_dir,
        num_rows=6,
        prefix_tokens=2,
        prefix_dim=8,
        state_dim=5,
        action_horizon=4,
        action_dim=3,
        seed=32,
    )
    _set_cache_wan_action_modes(cache_dir, ["current_wan_prefix_action_expert"] * 6)
    args = Args(
        cache_path=str(cache_dir),
        output_dir=str(output_dir),
        epochs=0,
        batch_size=2,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        sample_steps=2,
        seed=32,
    )

    metrics = run_train_eval(args)

    assert metrics["wan_action_mode"] == "current_wan_prefix_action_expert"
    saved_metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved_metrics["wan_action_mode"] == "current_wan_prefix_action_expert"
    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert checkpoint["args"]["wan_action_mode"] == "current_wan_prefix_action_expert"


def test_train_eval_rejects_disagreeing_wan_action_modes(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    write_fake_prefix_cache(
        cache_dir,
        num_rows=6,
        prefix_tokens=2,
        prefix_dim=8,
        state_dim=5,
        action_horizon=4,
        action_dim=3,
        seed=34,
    )
    _set_cache_wan_action_modes(
        cache_dir,
        [
            "current_wan_prefix_action_expert",
            "partial_wan_prefix_action_expert",
            "current_wan_prefix_action_expert",
            "current_wan_prefix_action_expert",
            "current_wan_prefix_action_expert",
            "current_wan_prefix_action_expert",
        ],
    )
    args = Args(
        cache_path=str(cache_dir),
        output_dir=str(output_dir),
        epochs=0,
        batch_size=2,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        sample_steps=2,
        seed=34,
    )

    with pytest.raises(ValueError, match="disagree on wan_action_mode"):
        run_train_eval(args)


def test_train_eval_rejects_partially_missing_wan_action_modes(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    write_fake_prefix_cache(
        cache_dir,
        num_rows=6,
        prefix_tokens=2,
        prefix_dim=8,
        state_dim=5,
        action_horizon=4,
        action_dim=3,
        seed=38,
    )
    _set_cache_wan_action_modes(
        cache_dir,
        [
            "current_wan_prefix_action_expert",
            None,
            "current_wan_prefix_action_expert",
            "current_wan_prefix_action_expert",
            "current_wan_prefix_action_expert",
            "current_wan_prefix_action_expert",
        ],
    )
    args = Args(
        cache_path=str(cache_dir),
        output_dir=str(output_dir),
        epochs=0,
        batch_size=2,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        sample_steps=2,
        seed=38,
    )

    with pytest.raises(ValueError, match="missing wan_action_mode"):
        run_train_eval(args)


def test_train_cli_fake_cache_records_original_scale_action_loss_weights(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    examples_dir = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "train_pi05_wan_action_expert.py",
            "--cache-path",
            str(cache_dir),
            "--output-dir",
            str(output_dir),
            "--fake-cache",
            "--fake-cache-rows",
            "6",
            "--epochs",
            "0",
            "--batch-size",
            "2",
            "--hidden-dim",
            "16",
            "--num-layers",
            "1",
            "--num-heads",
            "4",
            "--sample-steps",
            "2",
            "--normalize-actions",
            "--action-loss-weighting",
            "original_scale",
            "--device",
            "cpu",
            "--seed",
            "47",
        ],
        cwd=examples_dir,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    expected_weights = [std**2 for std in metrics["action_normalization_std"]]
    assert metrics["action_loss_weighting"] == "original_scale"
    assert metrics["action_loss_weights"] == pytest.approx(expected_weights)
    assert metrics["action_loss_weights_source"] == "action_norm_std_squared"

    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert checkpoint["args"]["action_loss_weighting"] == "original_scale"
    assert checkpoint["action_loss"]["weighting"] == "original_scale"
    assert torch.allclose(checkpoint["action_loss"]["weights"], torch.tensor(expected_weights))
    assert checkpoint["action_loss"]["weights_source"] == "action_norm_std_squared"


def test_original_scale_action_loss_weighting_requires_action_normalization(tmp_path) -> None:
    args = Args(
        cache_path=str(tmp_path / "missing_cache"),
        output_dir=str(tmp_path / "output"),
        normalize_actions=False,
        action_loss_weighting="original_scale",
    )

    with pytest.raises(ValueError, match="requires normalize_actions=True"):
        run_train_eval(args)


def test_train_eval_prefix_only_smoke_records_conditioning_mode(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    write_fake_prefix_cache(
        cache_dir,
        num_rows=6,
        prefix_tokens=2,
        prefix_dim=8,
        state_dim=5,
        action_horizon=4,
        action_dim=3,
        seed=33,
    )
    args = Args(
        cache_path=str(cache_dir),
        output_dir=str(output_dir),
        epochs=1,
        batch_size=2,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        sample_steps=2,
        conditioning_mode="wan_prefix",
        seed=33,
    )

    metrics = run_train_eval(args)

    assert metrics["conditioning_mode"] == "wan_prefix"
    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert checkpoint["args"]["conditioning_mode"] == "wan_prefix"
    assert checkpoint["model_kwargs"]["conditioning_mode"] == "wan_prefix"


def test_train_eval_smoke_records_film_timestep_conditioning(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    write_fake_prefix_cache(
        cache_dir,
        num_rows=6,
        prefix_tokens=2,
        prefix_dim=8,
        state_dim=5,
        action_horizon=4,
        action_dim=3,
        seed=35,
    )
    args = Args(
        cache_path=str(cache_dir),
        output_dir=str(output_dir),
        epochs=0,
        batch_size=2,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        sample_steps=2,
        timestep_conditioning="film",
        device="cpu",
        seed=35,
    )

    metrics = run_train_eval(args)

    assert metrics["timestep_conditioning"] == "film"
    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert checkpoint["args"]["timestep_conditioning"] == "film"
    assert checkpoint["model_kwargs"]["timestep_conditioning"] == "film"


def test_train_eval_smoke_records_pi05_timestep_embedding_style(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    write_fake_prefix_cache(
        cache_dir,
        num_rows=6,
        prefix_tokens=2,
        prefix_dim=8,
        state_dim=5,
        action_horizon=4,
        action_dim=3,
        seed=53,
    )
    args = Args(
        cache_path=str(cache_dir),
        output_dir=str(output_dir),
        epochs=0,
        batch_size=2,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        sample_steps=2,
        timestep_embedding_style="pi05",
        device="cpu",
        seed=53,
    )

    metrics = run_train_eval(args)

    assert metrics["timestep_embedding_style"] == "pi05"
    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert checkpoint["args"]["timestep_embedding_style"] == "pi05"
    assert checkpoint["model_kwargs"]["timestep_embedding_style"] == "pi05"
    assert checkpoint["metrics"]["timestep_embedding_style"] == "pi05"
    # The style round-trips through the inference loader so reloaded checkpoints rebuild the same embedding.
    loaded = load_wan_pi05_action_expert_checkpoint(output_dir / "checkpoint.pt", device="cpu")
    assert loaded.model.timestep_embedding_style == "pi05"
    assert loaded.model.time_mlp[0].style == "pi05"


def test_train_eval_smoke_records_cross_attention_decoder_arch(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    write_fake_prefix_cache(
        cache_dir,
        num_rows=6,
        prefix_tokens=2,
        prefix_dim=8,
        state_dim=5,
        action_horizon=4,
        action_dim=3,
        seed=36,
    )
    args = Args(
        cache_path=str(cache_dir),
        output_dir=str(output_dir),
        epochs=0,
        batch_size=2,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        sample_steps=2,
        decoder_arch="context_cross_attention",
        device="cpu",
        seed=36,
    )

    metrics = run_train_eval(args)

    assert (output_dir / "checkpoint.pt").exists()
    assert metrics["decoder_arch"] == "context_cross_attention"
    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert checkpoint["args"]["decoder_arch"] == "context_cross_attention"
    assert checkpoint["model_kwargs"]["decoder_arch"] == "context_cross_attention"


def test_train_eval_fake_cache_records_suffix_prefix_cache_decoder_arch(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    args = Args(
        cache_path=str(cache_dir),
        output_dir=str(output_dir),
        fake_cache=True,
        fake_cache_rows=6,
        epochs=0,
        batch_size=2,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        sample_steps=2,
        decoder_arch="suffix_prefix_cache",
        device="cpu",
        seed=40,
    )

    metrics = run_train_eval(args)

    assert (output_dir / "checkpoint.pt").exists()
    assert (output_dir / "metrics.json").exists()
    assert metrics["num_train"] == 4
    assert metrics["num_val"] == 2
    assert metrics["decoder_arch"] == "suffix_prefix_cache"
    assert metrics["val_model_sample_mse"] >= 0.0
    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert checkpoint["args"]["decoder_arch"] == "suffix_prefix_cache"
    assert checkpoint["model_kwargs"]["decoder_arch"] == "suffix_prefix_cache"
    assert checkpoint["metrics"]["decoder_arch"] == "suffix_prefix_cache"


def test_train_eval_fake_cache_records_joint_softmax_prefix_cache_decoder_arch(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    args = Args(
        cache_path=str(cache_dir),
        output_dir=str(output_dir),
        fake_cache=True,
        fake_cache_rows=6,
        epochs=0,
        batch_size=2,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        sample_steps=2,
        decoder_arch="joint_softmax_prefix_cache",
        device="cpu",
        seed=41,
    )

    metrics = run_train_eval(args)

    assert (output_dir / "checkpoint.pt").exists()
    assert (output_dir / "metrics.json").exists()
    assert metrics["num_train"] == 4
    assert metrics["num_val"] == 2
    assert metrics["decoder_arch"] == "joint_softmax_prefix_cache"
    assert metrics["val_model_sample_mse"] >= 0.0
    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert checkpoint["args"]["decoder_arch"] == "joint_softmax_prefix_cache"
    assert checkpoint["model_kwargs"]["decoder_arch"] == "joint_softmax_prefix_cache"
    assert checkpoint["metrics"]["decoder_arch"] == "joint_softmax_prefix_cache"


def test_train_eval_normalizes_actions_and_reports_original_unit_metrics(tmp_path) -> None:
    train_cache_dir = tmp_path / "train_cache"
    eval_cache_dir = tmp_path / "eval_cache"
    output_dir = tmp_path / "output"
    write_fake_prefix_cache(
        train_cache_dir,
        num_rows=5,
        prefix_tokens=2,
        prefix_dim=8,
        state_dim=5,
        action_horizon=4,
        action_dim=3,
        seed=37,
    )
    write_fake_prefix_cache(
        eval_cache_dir,
        num_rows=2,
        prefix_tokens=2,
        prefix_dim=8,
        state_dim=5,
        action_horizon=4,
        action_dim=3,
        seed=39,
    )
    _set_cache_actions(train_cache_dir, 100.0)
    _set_first_cache_row_actions_and_mask(train_cache_dir, value=1000.0, action_mask=torch.zeros(4))
    _set_cache_actions(eval_cache_dir, 101.0)
    args = Args(
        cache_path=str(train_cache_dir),
        eval_cache_path=str(eval_cache_dir),
        output_dir=str(output_dir),
        epochs=0,
        batch_size=2,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        sample_steps=2,
        normalize_actions=True,
        device="cpu",
        seed=37,
    )

    metrics = run_train_eval(args)

    assert metrics["normalize_actions"] is True
    assert metrics["action_normalization_mean"] == pytest.approx([100.0, 100.0, 100.0])
    assert metrics["action_normalization_std"] == pytest.approx([1e-6, 1e-6, 1e-6])
    assert metrics["val_mean_action_mse"] == pytest.approx(1.0)
    assert metrics["val_model_zero_noise_mse"] == pytest.approx(metrics["val_model_sample_mse"])
    assert metrics["val_model_sample_mse"] < 10.0

    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    normalization = checkpoint["action_normalization"]
    assert normalization["enabled"] is True
    assert torch.allclose(normalization["mean"], torch.full((3,), 100.0))
    assert torch.allclose(normalization["std"], torch.full((3,), 1e-6))
    assert checkpoint["args"]["normalize_actions"] is True


def test_train_eval_random_noise_eval_metrics(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    write_fake_prefix_cache(
        cache_dir,
        num_rows=6,
        prefix_tokens=2,
        prefix_dim=8,
        state_dim=5,
        action_horizon=4,
        action_dim=3,
        seed=51,
    )
    args = Args(
        cache_path=str(cache_dir),
        output_dir=str(output_dir),
        epochs=0,
        batch_size=2,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        sample_steps=2,
        eval_random_samples=2,
        eval_random_seed=123,
        device="cpu",
        seed=51,
    )

    metrics = run_train_eval(args)

    assert metrics["eval_random_samples"] == 2
    assert metrics["eval_random_seed"] == 123
    assert metrics["val_model_random_noise_mse_mean"] >= 0.0
    assert metrics["val_model_random_noise_mse_std"] >= 0.0


def test_train_eval_uses_held_out_eval_cache_for_counts_and_baseline(tmp_path) -> None:
    train_cache_dir = tmp_path / "train_cache"
    eval_cache_dir = tmp_path / "eval_cache"
    output_dir = tmp_path / "output"
    write_fake_prefix_cache(
        train_cache_dir,
        num_rows=5,
        prefix_tokens=2,
        prefix_dim=8,
        state_dim=5,
        action_horizon=4,
        action_dim=3,
        seed=41,
    )
    write_fake_prefix_cache(
        eval_cache_dir,
        num_rows=3,
        prefix_tokens=2,
        prefix_dim=8,
        state_dim=5,
        action_horizon=4,
        action_dim=3,
        seed=43,
    )
    _set_cache_actions(train_cache_dir, 1.0)
    _set_cache_actions(eval_cache_dir, 11.0)
    args = Args(
        cache_path=str(train_cache_dir),
        eval_cache_path=str(eval_cache_dir),
        output_dir=str(output_dir),
        epochs=0,
        batch_size=2,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        sample_steps=2,
        device="cpu",
        seed=41,
    )

    metrics = run_train_eval(args)

    assert metrics["cache_path"] == str(train_cache_dir)
    assert metrics["eval_cache_path"] == str(eval_cache_dir)
    assert metrics["num_train"] == 5
    assert metrics["num_val"] == 3
    assert metrics["val_mean_action_mse"] == pytest.approx(100.0)
    assert metrics["val_model_sample_mse"] >= 0.0


def _write_standard_prefix_cache(cache_dir, *, num_rows: int, seed: int, action_dim: int = 3) -> None:
    write_fake_prefix_cache(
        cache_dir,
        num_rows=num_rows,
        prefix_tokens=2,
        prefix_dim=8,
        state_dim=5,
        action_horizon=4,
        action_dim=action_dim,
        seed=seed,
    )


def _train_args(**overrides) -> Args:
    base = {
        "epochs": 0,
        "batch_size": 2,
        "hidden_dim": 16,
        "num_layers": 1,
        "num_heads": 4,
        "sample_steps": 2,
        "device": "cpu",
    }
    base.update(overrides)
    return Args(**base)


def test_train_eval_single_cache_records_train_cache_metadata(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    _write_standard_prefix_cache(cache_dir, num_rows=6, seed=61)

    metrics = run_train_eval(_train_args(cache_path=str(cache_dir), output_dir=str(output_dir), seed=61))

    assert metrics["cache_path"] == str(cache_dir)
    assert metrics["train_cache_paths"] == [str(cache_dir)]
    assert metrics["train_cache_sample_counts"] == [6]
    assert metrics["num_train"] + metrics["num_val"] == 6

    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert checkpoint["train_caches"] == [{"path": str(cache_dir), "num_samples": 6}]
    assert checkpoint["metrics"]["train_cache_paths"] == [str(cache_dir)]
    assert list(checkpoint["args"]["extra_cache_path"]) == []


def test_train_eval_concatenates_extra_train_caches_and_records_counts(tmp_path) -> None:
    primary_dir = tmp_path / "primary"
    extra_dir = tmp_path / "extra"
    eval_dir = tmp_path / "eval"
    output_dir = tmp_path / "output"
    _write_standard_prefix_cache(primary_dir, num_rows=4, seed=71)
    _write_standard_prefix_cache(extra_dir, num_rows=3, seed=72)
    _write_standard_prefix_cache(eval_dir, num_rows=2, seed=73)

    metrics = run_train_eval(
        _train_args(
            cache_path=str(primary_dir),
            extra_cache_path=(str(extra_dir),),
            eval_cache_path=str(eval_dir),
            output_dir=str(output_dir),
            seed=71,
        )
    )

    # eval cache is held out separately, so the full primary+extra concatenation is the train set.
    assert metrics["train_cache_paths"] == [str(primary_dir), str(extra_dir)]
    assert metrics["train_cache_sample_counts"] == [4, 3]
    assert metrics["num_train"] == 7
    assert metrics["num_val"] == 2
    assert metrics["cache_path"] == str(primary_dir)
    assert metrics["eval_cache_path"] == str(eval_dir)

    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert checkpoint["train_caches"] == [
        {"path": str(primary_dir), "num_samples": 4},
        {"path": str(extra_dir), "num_samples": 3},
    ]
    assert list(checkpoint["args"]["extra_cache_path"]) == [str(extra_dir)]


def test_train_eval_extra_train_caches_split_when_no_eval_cache(tmp_path) -> None:
    primary_dir = tmp_path / "primary"
    extra_dir = tmp_path / "extra"
    output_dir = tmp_path / "output"
    _write_standard_prefix_cache(primary_dir, num_rows=4, seed=81)
    _write_standard_prefix_cache(extra_dir, num_rows=4, seed=82)

    metrics = run_train_eval(
        _train_args(
            cache_path=str(primary_dir),
            extra_cache_path=(str(extra_dir),),
            output_dir=str(output_dir),
            val_fraction=0.25,
            seed=81,
        )
    )

    assert metrics["train_cache_paths"] == [str(primary_dir), str(extra_dir)]
    assert metrics["train_cache_sample_counts"] == [4, 4]
    assert metrics["num_train"] + metrics["num_val"] == 8
    assert "eval_cache_path" not in metrics


def test_train_eval_normalizes_actions_across_concatenated_train_caches(tmp_path) -> None:
    primary_dir = tmp_path / "primary"
    extra_dir = tmp_path / "extra"
    eval_dir = tmp_path / "eval"
    output_dir = tmp_path / "output"
    _write_standard_prefix_cache(primary_dir, num_rows=2, seed=91)
    _write_standard_prefix_cache(extra_dir, num_rows=2, seed=92)
    _write_standard_prefix_cache(eval_dir, num_rows=2, seed=93)
    _set_cache_actions(primary_dir, 100.0)
    _set_cache_actions(extra_dir, 102.0)
    _set_cache_actions(eval_dir, 50.0)

    metrics = run_train_eval(
        _train_args(
            cache_path=str(primary_dir),
            extra_cache_path=(str(extra_dir),),
            eval_cache_path=str(eval_dir),
            output_dir=str(output_dir),
            batch_size=4,
            normalize_actions=True,
            seed=91,
        )
    )

    # Stats must span both train caches: mean = (2*100 + 2*102)/4 = 101, std = 1.0 per dim.
    # Using only the primary cache would yield mean 100 and std ~0, so this pins concatenation.
    assert metrics["action_normalization_mean"] == pytest.approx([101.0, 101.0, 101.0])
    assert metrics["action_normalization_std"] == pytest.approx([1.0, 1.0, 1.0])


def test_train_eval_rejects_incompatible_extra_cache_shapes(tmp_path) -> None:
    primary_dir = tmp_path / "primary"
    extra_dir = tmp_path / "extra"
    output_dir = tmp_path / "output"
    _write_standard_prefix_cache(primary_dir, num_rows=4, seed=101, action_dim=3)
    _write_standard_prefix_cache(extra_dir, num_rows=4, seed=102, action_dim=4)

    with pytest.raises(ValueError, match="Incompatible Wan prefix cache shapes"):
        run_train_eval(
            _train_args(
                cache_path=str(primary_dir),
                extra_cache_path=(str(extra_dir),),
                output_dir=str(output_dir),
                seed=101,
            )
        )


def test_train_eval_rejects_extra_cache_with_disagreeing_wan_action_mode(tmp_path) -> None:
    primary_dir = tmp_path / "primary"
    extra_dir = tmp_path / "extra"
    output_dir = tmp_path / "output"
    _write_standard_prefix_cache(primary_dir, num_rows=3, seed=111)
    _write_standard_prefix_cache(extra_dir, num_rows=3, seed=112)
    _set_cache_wan_action_modes(primary_dir, ["current_wan_prefix_action_expert"] * 3)
    _set_cache_wan_action_modes(extra_dir, ["partial_wan_prefix_action_expert"] * 3)

    with pytest.raises(ValueError, match="disagree on wan_action_mode"):
        run_train_eval(
            _train_args(
                cache_path=str(primary_dir),
                extra_cache_path=(str(extra_dir),),
                output_dir=str(output_dir),
                seed=111,
            )
        )
