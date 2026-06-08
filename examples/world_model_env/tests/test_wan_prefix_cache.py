from __future__ import annotations

import inspect
import json

import pytest
import torch

import cache_pi05_wan_prefix_tokens as cache_module
from cache_pi05_wan_prefix_tokens import Args as CacheArgs
from cache_pi05_wan_prefix_tokens import precompute_pi05_wan_prefix_tokens
from train_pi05_wan_action_expert import Args as TrainArgs
from train_pi05_wan_action_expert import _resolve_action_loss_weights, run_train_eval
from world_model.pi05_wan_action_expert import FUTURE_LEAKAGE_KEYS, CachedWanPrefixActionDataset
from world_model.wan_dit_prefix_encoder import (
    DEFAULT_WAN_DIT_LAYERS,
    DIFFSYNTH_WAN22_TI2V_DIT_SOURCE,
    WAN_DIT_FUTURE_LATENT_NOISE_SEED_STRATEGY,
    WAN_DIT_HIDDEN_POOL_DESCRIPTION,
    WAN_DIT_HIDDEN_POOL_TOKEN_POOL,
    wan_dit_future_latent_noise_seed,
)
from world_model.wan_prefix_encoder import FakeWanCurrentPrefixEncoder, FrozenDiffSynthWanCurrentPrefixEncoder

CURRENT_WAN_ACTION_MODE = "current_wan_prefix_action_expert"
NON_WAN_ACTION_MODE = "non_wan_current_prefix_baseline"
PARTIAL_WAN_ACTION_MODE = "partial_wan_prefix_action_expert"


class _InjectedPrefixEncoder:
    def __init__(self, *, prefix_dim: int, token_count: int = 3) -> None:
        self.prefix_dim = prefix_dim
        self.token_count = token_count

    def encode_prefix(self, current_images: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        batch_size = current_images.shape[0]
        base = torch.arange(self.token_count * self.prefix_dim, dtype=torch.float32, device=current_images.device)
        tokens = base.reshape(1, self.token_count, self.prefix_dim).repeat(batch_size, 1, 1)
        prompt_offsets = torch.tensor(
            [len(prompt) for prompt in prompts], dtype=torch.float32, device=current_images.device
        )
        return tokens + prompt_offsets.view(batch_size, 1, 1) * 0.001


class _IndexAwarePrefixEncoder(_InjectedPrefixEncoder):
    def __init__(self, *, prefix_dim: int, token_count: int = 3) -> None:
        super().__init__(prefix_dim=prefix_dim, token_count=token_count)
        self.seen_sample_indices: list[list[int]] = []

    def encode_prefix(
        self,
        current_images: torch.Tensor,
        prompts: list[str],
        *,
        sample_indices: list[int] | None = None,
    ) -> torch.Tensor:
        if sample_indices is None:
            raise AssertionError("sample_indices must be passed for noisy DiT prefix caching")
        self.seen_sample_indices.append(list(sample_indices))
        tokens = super().encode_prefix(current_images, prompts)
        offsets = torch.tensor(sample_indices, dtype=torch.float32, device=current_images.device)
        return tokens + offsets.view(-1, 1, 1)


class _FutureLatentAwarePrefixEncoder(_InjectedPrefixEncoder):
    def __init__(self, *, prefix_dim: int, token_count: int = 3) -> None:
        super().__init__(prefix_dim=prefix_dim, token_count=token_count)
        self.seen_future_latents: list[torch.Tensor] = []

    def encode_prefix(
        self,
        current_images: torch.Tensor,
        prompts: list[str],
        *,
        future_latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if future_latents is None:
            raise AssertionError("future_latents must be passed when future_latent_cache_dir is configured")
        self.seen_future_latents.append(future_latents.detach().cpu().clone())
        tokens = super().encode_prefix(current_images, prompts)
        future_signal = future_latents.float().mean(dim=(1, 2, 3, 4))
        return tokens + future_signal.view(-1, 1, 1)


def _cache_args(cache_dir, **overrides) -> CacheArgs:
    values = {
        "dataset_source": "synthetic",
        "image_key": "corner4.image",
        "output_dir": str(cache_dir),
        "synthetic_samples": 6,
        "image_size": 16,
        "action_horizon": 4,
        "batch_size": 2,
        "device": "cpu",
        "prefix_dim": 8,
        "fake_encoder": True,
        "fake_spatial_stride": 16,
        "seed": 13,
    }
    values.update(overrides)
    return CacheArgs(**values)


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _future_cache_dataset_config(*, synthetic_samples: int = 6) -> dict:
    return {
        "source": "synthetic",
        "repo_id": "brandonyang/metaworld_ml45",
        "image_keys": ["corner4.image"],
        "image_size": 16,
        "frame_delta": 1,
        "num_future_frames": 4,
        "action_horizon": 4,
        "idm_history_length": 0,
        "max_samples": None,
        "samples_per_episode": None,
        "synthetic_samples": synthetic_samples,
        "episodes": None,
        "seed": 13,
    }


def _write_future_latent_cache(
    cache_dir,
    *,
    indices=(0, 1, 2, 3, 4, 5),
    duplicate_index: int | None = None,
    generated: bool = False,
    synthetic_samples: int = 6,
) -> None:
    cache_dir.mkdir(parents=True)
    latents_dir = cache_dir / "latents"
    latents_dir.mkdir()
    latent_shape = (4, 3, 2, 2)
    metadata = {
        "version": 1,
        "repo_id": "brandonyang/metaworld_ml45",
        "image_keys": ["corner4.image"],
        "image_size": 16,
        "frame_delta": 1,
        "num_future_frames": 4,
        "idm_history_length": 0,
        "wan_vae_latent_channels": 4,
        "wan_vae_spatial_stride": 8,
        "num_samples": len(indices) + (1 if duplicate_index is not None else 0),
        "dataset_config": _future_cache_dataset_config(synthetic_samples=synthetic_samples),
    }
    if generated:
        metadata.update(
            {
                "cache_schema": "generated_wan_latents",
                "generator": {
                    "source": "diffsynth_wan_lora",
                    "denoise_mode": "partial",
                    "num_inference_steps": 8,
                    "stop_after_steps": 4,
                },
            }
        )
    else:
        metadata.update(
            {
                "wan_vae_checkpoint_path": "fake-wan-vae.ckpt",
                "wan_vae_dtype": "float32",
            }
        )
    rows = []
    for dataset_index in list(indices) + ([] if duplicate_index is None else [duplicate_index]):
        tensor = torch.full(latent_shape, float(dataset_index), dtype=torch.float32)
        tensor[:, 1:] += 0.25
        relative_path = f"latents/sample_{dataset_index:06d}.pt"
        torch.save(tensor, cache_dir / relative_path)
        row = {
            "dataset_index": dataset_index,
            "latent_tensor": relative_path,
            "latent_shape": list(latent_shape),
        }
        if generated:
            row["generator_metadata"] = metadata["generator"]
            row["seed"] = 1000 + dataset_index
        rows.append(row)
    (cache_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (cache_dir / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def _forbidden_paths(value, prefix=""):
    paths = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if key_text in FUTURE_LEAKAGE_KEYS:
                paths.append(path)
            paths.extend(_forbidden_paths(child, path))
    elif isinstance(value, list | tuple):
        for index, child in enumerate(value):
            paths.extend(_forbidden_paths(child, f"{prefix}[{index}]"))
    return paths


def test_fake_wan_prefix_cache_writes_rows_loadable_by_pi05_dataset(tmp_path) -> None:
    cache_dir = tmp_path / "cache"

    result = precompute_pi05_wan_prefix_tokens(_cache_args(cache_dir))
    dataset = CachedWanPrefixActionDataset(cache_dir)
    sample = dataset[0]
    row = torch.load(cache_dir / "sample_000000.pt", map_location="cpu", weights_only=False)
    config = json.loads((cache_dir / "config.json").read_text(encoding="utf-8"))

    assert result["written"] == 6
    assert len(dataset) == 6
    assert set(row) == {"prefix_tokens", "state", "actions", "action_mask", "task", "metadata"}
    assert sample["prefix_tokens"].shape == (2, 8)
    assert sample["state"].shape == (4,)
    assert sample["actions"].shape == (4, 4)
    assert sample["action_mask"].shape == (4,)
    assert config["cache_kind"] == "pi05_wan_current_prefix"
    assert config["prefix_token_count"] == 2
    assert config["source"] == "fake"
    assert result["wan_action_mode"] == NON_WAN_ACTION_MODE
    assert config["wan_action_mode"] == NON_WAN_ACTION_MODE
    assert row["metadata"]["wan_action_mode"] == NON_WAN_ACTION_MODE


def test_raw_current_prefix_cache_writes_loadable_rows_without_wan_encoder_instantiation(
    monkeypatch,
    tmp_path,
) -> None:
    cache_dir = tmp_path / "cache"

    def _forbidden_wan_encoder(**kwargs):
        raise AssertionError(f"raw_current backend must not instantiate Wan encoders: {kwargs}")

    monkeypatch.setattr(cache_module, "FrozenDiffSynthWanCurrentPrefixEncoder", _forbidden_wan_encoder)
    monkeypatch.setattr(cache_module, "FrozenDiffSynthWanDiTCurrentPrefixEncoder", _forbidden_wan_encoder)

    result = precompute_pi05_wan_prefix_tokens(
        _cache_args(
            cache_dir,
            fake_encoder=False,
            prefix_backend="raw_current",
            prefix_dim=3072,
            synthetic_samples=3,
        )
    )
    dataset = CachedWanPrefixActionDataset(cache_dir)
    sample = dataset[0]
    row = torch.load(cache_dir / "sample_000000.pt", map_location="cpu", weights_only=False)
    manifest_row = _jsonl(cache_dir / "manifest.jsonl")[0]
    config = json.loads((cache_dir / "config.json").read_text(encoding="utf-8"))
    compression = {
        "description": (
            "deterministic non-Wan current RGB image + prompt baseline; exactly three current-only tokens, "
            "with each token fit to prefix_dim"
        ),
        "token_order": [
            "downsampled_current_image",
            "current_image_statistics",
            "hashed_char_ngram_prompt",
        ],
        "image": "current RGB image clamped to [0, 1], adaptive-average-pooled, flattened, then fit to prefix_dim",
        "image_statistics": (
            "current RGB per-channel/global mean/std/min/max plus 8-bin per-channel histogram, then fit to prefix_dim"
        ),
        "text": "deterministic signed blake2b feature hashing over prompt character 1-4grams to prefix_dim",
        "uses_wan": False,
        "uses_diffsynth": False,
        "uses_future_images": False,
        "uses_future_latents": False,
    }

    assert result["written"] == 3
    assert len(dataset) == 3
    assert sample["prefix_tokens"].shape == (3, 3072)
    assert row["prefix_tokens"].shape == (3, 3072)
    assert config["cache_kind"] == "pi05_wan_current_prefix"
    assert config["prefix_dim"] == 3072
    assert config["prefix_token_count"] == 3
    assert config["source"] == "raw_current_image_prompt"
    assert config["prefix_compression"] == compression
    assert result["source"] == "raw_current_image_prompt"
    assert result["wan_action_mode"] == NON_WAN_ACTION_MODE
    assert result["prefix_compression"] == compression
    assert row["metadata"]["source"] == "raw_current_image_prompt"
    assert row["metadata"]["wan_action_mode"] == NON_WAN_ACTION_MODE
    assert row["metadata"]["prefix_compression"] == compression
    assert manifest_row["source"] == "raw_current_image_prompt"
    assert manifest_row["wan_action_mode"] == NON_WAN_ACTION_MODE
    assert manifest_row["prefix_compression"] == compression
    assert _forbidden_paths(row) == []
    for metadata in (result, config, row["metadata"], manifest_row):
        assert metadata["contains_future_images"] is False
        assert metadata["contains_future_latents"] is False


def test_raw_current_prefix_encoder_is_deterministic_current_only() -> None:
    encoder = cache_module.RawCurrentImagePromptPrefixEncoder(prefix_dim=3072)
    current_images = torch.linspace(0.0, 1.0, 2 * 3 * 16 * 16).reshape(2, 3, 16, 16)
    prompts = ["pick object", "open drawer"]

    prefix_a = encoder.encode_prefix(current_images, prompts)
    prefix_b = encoder.encode_prefix(current_images, prompts)
    changed_prompt = encoder.encode_prefix(current_images, ["pick object carefully", prompts[1]])

    assert prefix_a.shape == (2, 3, 3072)
    assert torch.allclose(prefix_a, prefix_b)
    assert torch.allclose(prefix_a[:, 0], changed_prompt[:, 0])
    assert torch.allclose(prefix_a[:, 1], changed_prompt[:, 1])
    assert not torch.allclose(prefix_a[0, 2], changed_prompt[0, 2])
    assert "future_images" not in inspect.signature(encoder.encode_prefix).parameters
    with pytest.raises(TypeError):
        encoder.encode_prefix(current_images, prompts, future_images=torch.zeros(1))


def test_raw_current_prefix_cache_records_batch_source_provenance(tmp_path) -> None:
    cache_dir = tmp_path / "cache"

    precompute_pi05_wan_prefix_tokens(
        _cache_args(
            cache_dir,
            fake_encoder=False,
            prefix_backend="raw_current",
            prefix_dim=32,
            synthetic_samples=18,
            batch_size=5,
        )
    )
    row = torch.load(cache_dir / "sample_000017.pt", map_location="cpu", weights_only=False)
    manifest_row = _jsonl(cache_dir / "manifest.jsonl")[17]
    expected_provenance = {
        "cache_index": 17,
        "source_dataset_index": 17,
        "episode_index": 1,
        "frame_index": 1,
        "task_index": 1,
    }

    assert row["metadata"]["source"] == "raw_current_image_prompt"
    assert manifest_row["source"] == "raw_current_image_prompt"
    assert row["metadata"]["dataset_index"] == 17
    assert manifest_row["dataset_index"] == 17
    for key, value in expected_provenance.items():
        assert row["metadata"][key] == value
        assert manifest_row[key] == value


def test_raw_current_prefix_cache_rejects_existing_source_config_mismatch(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    precompute_pi05_wan_prefix_tokens(
        _cache_args(cache_dir, fake_encoder=False, prefix_backend="raw_current", prefix_dim=8)
    )

    with pytest.raises(ValueError, match="metadata mismatch for source"):
        precompute_pi05_wan_prefix_tokens(
            _cache_args(cache_dir, fake_encoder=False, prefix_backend="vae_text", prefix_dim=8),
            encoder=_InjectedPrefixEncoder(prefix_dim=8),
        )


def test_real_backend_selection_changes_source_and_prefix_metadata_without_loading_weights(tmp_path) -> None:
    vae_cache_dir = tmp_path / "vae_cache"
    dit_cache_dir = tmp_path / "dit_cache"

    vae_result = precompute_pi05_wan_prefix_tokens(
        _cache_args(vae_cache_dir, fake_encoder=False, prefix_backend="vae_text"),
        encoder=_InjectedPrefixEncoder(prefix_dim=8),
    )
    dit_result = precompute_pi05_wan_prefix_tokens(
        _cache_args(dit_cache_dir, fake_encoder=False, prefix_backend="dit_hidden"),
        encoder=_InjectedPrefixEncoder(prefix_dim=8),
    )
    vae_config = json.loads((vae_cache_dir / "config.json").read_text(encoding="utf-8"))
    dit_config = json.loads((dit_cache_dir / "config.json").read_text(encoding="utf-8"))
    dit_row = torch.load(dit_cache_dir / "sample_000000.pt", map_location="cpu", weights_only=False)
    dit_manifest_row = _jsonl(dit_cache_dir / "manifest.jsonl")[0]

    assert vae_result["source"] == "diffsynth_wan2.2_ti2v_5b_current_vae_text"
    assert vae_config["source"] == vae_result["source"]
    assert vae_result["wan_action_mode"] == CURRENT_WAN_ACTION_MODE
    assert vae_config["wan_action_mode"] == CURRENT_WAN_ACTION_MODE
    assert "text" in vae_config["prefix_compression"]
    assert vae_config["prefix_compression"]["image"] == (
        "wan_vae_latents_flattened_T_H_W_to_tokens_with_last_dim_fit_to_prefix_dim"
    )

    assert dit_result["source"] == DIFFSYNTH_WAN22_TI2V_DIT_SOURCE
    assert dit_config["source"] == DIFFSYNTH_WAN22_TI2V_DIT_SOURCE
    assert dit_row["metadata"]["source"] == DIFFSYNTH_WAN22_TI2V_DIT_SOURCE
    assert dit_manifest_row["source"] == DIFFSYNTH_WAN22_TI2V_DIT_SOURCE
    for metadata in (dit_result, dit_config, dit_row["metadata"], dit_manifest_row):
        assert metadata["wan_action_mode"] == CURRENT_WAN_ACTION_MODE
        assert metadata["contains_future_ground_truth_latents"] is False
        assert metadata["uses_future_latent_slots"] is False
        assert metadata["future_latent_slot_count"] == 0
        assert metadata["wan_backbone_runs_per_observation"] == 1
        assert metadata["native_wan_attention_kv_cache"] is False
        compression = metadata["prefix_compression"]
        assert "current-only fused latents" in compression["description"]
        assert compression["hidden_pool"] == WAN_DIT_HIDDEN_POOL_DESCRIPTION
        assert compression["tokens_per_layer"] == 1
        assert compression["selected_layers"] == list(DEFAULT_WAN_DIT_LAYERS)
        assert compression["num_latent_frames"] == 1
        assert compression["timestep"] == 500.0
        assert compression["configured_timestep"] == 500.0
        assert compression["timestep_shape"] == [1]
        assert compression["timestep_applies_to_future_latents_only"] is True
        assert compression["future_latent_frames"] == 0
        assert compression["effective_timestep"] == 0.0
        assert compression["future_latent_fill"] == "zeros"
        assert compression["future_latent_seed"] == 0
        assert compression["fuse_vae_embedding_in_latents"] is True


def test_dit_backend_prefix_metadata_records_future_latent_timestep(tmp_path) -> None:
    cache_dir = tmp_path / "cache"

    result = precompute_pi05_wan_prefix_tokens(
        _cache_args(
            cache_dir,
            fake_encoder=False,
            prefix_backend="dit_hidden",
            dit_num_latent_frames=3,
            dit_timestep=250.0,
            dit_future_latent_fill="noise",
            dit_future_latent_seed=123,
        ),
        encoder=_InjectedPrefixEncoder(prefix_dim=8),
    )
    config = json.loads((cache_dir / "config.json").read_text(encoding="utf-8"))
    row = torch.load(cache_dir / "sample_000000.pt", map_location="cpu", weights_only=False)
    manifest_row = _jsonl(cache_dir / "manifest.jsonl")[0]

    for metadata in (result, config, row["metadata"], manifest_row):
        assert metadata["wan_action_mode"] == PARTIAL_WAN_ACTION_MODE
        assert metadata["contains_future_ground_truth_latents"] is False
        assert metadata["uses_future_latent_slots"] is True
        assert metadata["future_latent_slot_count"] == 2
        assert metadata["wan_backbone_runs_per_observation"] == 1
        assert metadata["native_wan_attention_kv_cache"] is False
        compression = metadata["prefix_compression"]
        assert "current-only fused latents" not in compression["description"]
        assert "deterministic per-sample noise future latent slot(s)" in compression["description"]
        assert compression["num_latent_frames"] == 3
        assert compression["timestep"] == 250.0
        assert compression["configured_timestep"] == 250.0
        assert compression["timestep_shape"] == [1]
        assert compression["timestep_applies_to_future_latents_only"] is True
        assert compression["future_latent_frames"] == 2
        assert compression["effective_timestep"] == 250.0
        assert compression["future_latent_fill"] == "noise"
        assert compression["future_latent_seed"] == 123
        assert compression["future_slot_conditioning"] == "deterministic_per_sample_noise_placeholders"
        assert compression["future_slot_noise_seed_key"] == "dataset_index"
        assert compression["future_slot_noise_seed_strategy"] == WAN_DIT_FUTURE_LATENT_NOISE_SEED_STRATEGY
        assert compression["uses_future_ground_truth_latents"] is False
        assert compression["stores_future_ground_truth_latents"] is False


def test_dit_noise_prefix_cache_passes_sample_indices_and_records_row_seeds(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    encoder = _IndexAwarePrefixEncoder(prefix_dim=8)

    precompute_pi05_wan_prefix_tokens(
        _cache_args(
            cache_dir,
            fake_encoder=False,
            prefix_backend="dit_hidden",
            synthetic_samples=5,
            batch_size=2,
            dit_num_latent_frames=3,
            dit_future_latent_fill="noise",
            dit_future_latent_seed=123,
        ),
        encoder=encoder,
    )
    config = json.loads((cache_dir / "config.json").read_text(encoding="utf-8"))
    row_0 = torch.load(cache_dir / "sample_000000.pt", map_location="cpu", weights_only=False)
    row_3 = torch.load(cache_dir / "sample_000003.pt", map_location="cpu", weights_only=False)
    manifest_rows = _jsonl(cache_dir / "manifest.jsonl")

    assert encoder.seen_sample_indices == [[0, 1], [2, 3], [4]]
    compression = config["prefix_compression"]
    assert compression["future_slot_noise_seed_key"] == "dataset_index"
    assert compression["future_slot_noise_seed_strategy"] == WAN_DIT_FUTURE_LATENT_NOISE_SEED_STRATEGY
    assert row_0["metadata"]["dit_future_slot_noise_seed"] == wan_dit_future_latent_noise_seed(123, 0)
    assert row_3["metadata"]["dit_future_slot_noise_seed"] == wan_dit_future_latent_noise_seed(123, 3)
    assert row_0["metadata"]["dit_future_slot_noise_seed"] != row_3["metadata"]["dit_future_slot_noise_seed"]
    assert manifest_rows[3]["dit_future_slot_noise_seed"] == wan_dit_future_latent_noise_seed(123, 3)
    assert row_3["prefix_tokens"][0, 0] > row_0["prefix_tokens"][0, 0]


def test_dit_noise_prefix_cache_outputs_are_batch_size_independent(tmp_path) -> None:
    cache_dir_a = tmp_path / "cache_a"
    cache_dir_b = tmp_path / "cache_b"
    shared_overrides = {
        "fake_encoder": False,
        "prefix_backend": "dit_hidden",
        "synthetic_samples": 5,
        "dit_num_latent_frames": 3,
        "dit_future_latent_fill": "noise",
        "dit_future_latent_seed": 123,
    }

    precompute_pi05_wan_prefix_tokens(
        _cache_args(cache_dir_a, batch_size=2, **shared_overrides),
        encoder=_IndexAwarePrefixEncoder(prefix_dim=8),
    )
    precompute_pi05_wan_prefix_tokens(
        _cache_args(cache_dir_b, batch_size=4, **shared_overrides),
        encoder=_IndexAwarePrefixEncoder(prefix_dim=8),
    )

    for dataset_index in range(5):
        row_a = torch.load(cache_dir_a / f"sample_{dataset_index:06d}.pt", map_location="cpu", weights_only=False)
        row_b = torch.load(cache_dir_b / f"sample_{dataset_index:06d}.pt", map_location="cpu", weights_only=False)
        assert torch.allclose(row_a["prefix_tokens"], row_b["prefix_tokens"])
        assert row_a["metadata"]["dit_future_slot_noise_seed"] == row_b["metadata"]["dit_future_slot_noise_seed"]


def test_dit_backend_can_join_ground_truth_future_latent_cache(tmp_path) -> None:
    cache_dir = tmp_path / "prefix_cache"
    future_cache_dir = tmp_path / "future_latents"
    _write_future_latent_cache(future_cache_dir)
    encoder = _FutureLatentAwarePrefixEncoder(prefix_dim=8)

    result = precompute_pi05_wan_prefix_tokens(
        _cache_args(
            cache_dir,
            fake_encoder=False,
            prefix_backend="dit_hidden",
            dit_num_latent_frames=3,
            future_latent_cache_dir=str(future_cache_dir),
        ),
        encoder=encoder,
    )
    config = json.loads((cache_dir / "config.json").read_text(encoding="utf-8"))
    row = torch.load(cache_dir / "sample_000002.pt", map_location="cpu", weights_only=False)
    manifest_row = _jsonl(cache_dir / "manifest.jsonl")[2]

    assert len(encoder.seen_future_latents) == 3
    assert encoder.seen_future_latents[1].shape == (2, 4, 2, 2, 2)
    assert torch.allclose(encoder.seen_future_latents[1][0], torch.full((4, 2, 2, 2), 2.25))
    for metadata in (result, config, row["metadata"], manifest_row):
        assert metadata["wan_action_mode"] == PARTIAL_WAN_ACTION_MODE
        assert metadata["contains_future_latents"] is True
        assert metadata["contains_future_ground_truth_latents"] is True
        assert metadata["uses_future_latent_slots"] is True
        assert metadata["future_latent_slot_count"] == 2
        assert metadata["native_wan_attention_kv_cache"] is False
        assert metadata["future_slot_cache_source"] == "wan_vae_ground_truth_latents"
        assert metadata["future_slot_cache_dir"] == str(future_cache_dir.resolve())
        compression = metadata["prefix_compression"]
        assert compression["future_slot_conditioning"] == "cached_future_latents"
        assert compression["future_latent_fill"] == "cached"
        assert compression["uses_future_ground_truth_latents"] is True
    assert row["metadata"]["future_slot_cache_tensor"] == "latents/sample_000002.pt"
    assert manifest_row["future_slot_cache_dataset_index"] == 2


def test_dit_backend_records_generated_future_latent_cache_metadata(tmp_path) -> None:
    cache_dir = tmp_path / "prefix_cache"
    future_cache_dir = tmp_path / "generated_latents"
    _write_future_latent_cache(future_cache_dir, generated=True)

    precompute_pi05_wan_prefix_tokens(
        _cache_args(
            cache_dir,
            fake_encoder=False,
            prefix_backend="dit_hidden",
            dit_num_latent_frames=3,
            future_latent_cache_dir=str(future_cache_dir),
        ),
        encoder=_FutureLatentAwarePrefixEncoder(prefix_dim=8),
    )
    row = torch.load(cache_dir / "sample_000003.pt", map_location="cpu", weights_only=False)
    config = json.loads((cache_dir / "config.json").read_text(encoding="utf-8"))

    assert config["contains_future_latents"] is True
    assert config["contains_future_ground_truth_latents"] is False
    assert config["future_slot_cache_source"] == "generated_wan_latents"
    assert config["prefix_compression"]["future_slot_generator_metadata"]["denoise_mode"] == "partial"
    assert row["metadata"]["future_slot_generation_seed"] == 1003
    assert row["metadata"]["future_slot_generator_metadata"]["stop_after_steps"] == 4


def test_dit_backend_accepts_legacy_generated_future_latent_dataset_metadata(tmp_path) -> None:
    cache_dir = tmp_path / "prefix_cache"
    future_cache_dir = tmp_path / "legacy_generated_latents"
    _write_future_latent_cache(future_cache_dir, generated=True)
    config_path = future_cache_dir / "config.json"
    metadata = json.loads(config_path.read_text(encoding="utf-8"))
    dataset_config = metadata["dataset_config"]
    for key in (
        "repo_id",
        "image_keys",
        "image_size",
        "frame_delta",
        "max_samples",
        "samples_per_episode",
        "synthetic_samples",
        "episodes",
        "seed",
    ):
        metadata[key] = dataset_config[key]
    metadata["dataset_config"] = {"source": dataset_config["source"]}
    config_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    precompute_pi05_wan_prefix_tokens(
        _cache_args(
            cache_dir,
            fake_encoder=False,
            prefix_backend="dit_hidden",
            dit_num_latent_frames=3,
            future_latent_cache_dir=str(future_cache_dir),
        ),
        encoder=_FutureLatentAwarePrefixEncoder(prefix_dim=8),
    )
    config = json.loads((cache_dir / "config.json").read_text(encoding="utf-8"))
    manifest_row = _jsonl(cache_dir / "manifest.jsonl")[0]

    assert config["future_slot_cache_source"] == "generated_wan_latents"
    assert manifest_row["future_slot_cache_source"] == "generated_wan_latents"
    assert manifest_row["future_slot_generation_seed"] == 1000


def test_future_latent_cache_rejects_missing_and_duplicate_rows(tmp_path) -> None:
    cache_dir = tmp_path / "prefix_cache"
    missing_future_cache = tmp_path / "missing_future_latents"
    duplicate_future_cache = tmp_path / "duplicate_future_latents"
    _write_future_latent_cache(missing_future_cache, indices=(0, 1), synthetic_samples=3)
    _write_future_latent_cache(duplicate_future_cache, duplicate_index=1)

    with pytest.raises(ValueError, match="missing dataset_index=2"):
        precompute_pi05_wan_prefix_tokens(
            _cache_args(
                cache_dir,
                fake_encoder=False,
                prefix_backend="dit_hidden",
                synthetic_samples=3,
                dit_num_latent_frames=3,
                future_latent_cache_dir=str(missing_future_cache),
            ),
            encoder=_FutureLatentAwarePrefixEncoder(prefix_dim=8),
        )

    with pytest.raises(ValueError, match="duplicate dataset_index=1"):
        precompute_pi05_wan_prefix_tokens(
            _cache_args(
                tmp_path / "duplicate_prefix_cache",
                fake_encoder=False,
                prefix_backend="dit_hidden",
                dit_num_latent_frames=3,
                future_latent_cache_dir=str(duplicate_future_cache),
            ),
            encoder=_FutureLatentAwarePrefixEncoder(prefix_dim=8),
        )


def test_future_latent_cache_requires_dit_future_slots(tmp_path) -> None:
    future_cache_dir = tmp_path / "future_latents"
    _write_future_latent_cache(future_cache_dir)

    with pytest.raises(ValueError, match="requires prefix_backend='dit_hidden'"):
        precompute_pi05_wan_prefix_tokens(
            _cache_args(tmp_path / "fake_cache", future_latent_cache_dir=str(future_cache_dir))
        )
    with pytest.raises(ValueError, match="requires dit_num_latent_frames > 1"):
        precompute_pi05_wan_prefix_tokens(
            _cache_args(
                tmp_path / "current_only_cache",
                fake_encoder=False,
                prefix_backend="dit_hidden",
                dit_num_latent_frames=1,
                future_latent_cache_dir=str(future_cache_dir),
            ),
            encoder=_FutureLatentAwarePrefixEncoder(prefix_dim=8),
        )


def test_dit_backend_token_pool_metadata_records_tokens_per_layer(tmp_path) -> None:
    cache_dir = tmp_path / "cache"

    result = precompute_pi05_wan_prefix_tokens(
        _cache_args(
            cache_dir,
            fake_encoder=False,
            prefix_backend="dit_hidden",
            dit_selected_layers=(1, 3),
            dit_hidden_pool=WAN_DIT_HIDDEN_POOL_TOKEN_POOL,
            dit_tokens_per_layer=4,
        ),
        encoder=_InjectedPrefixEncoder(prefix_dim=8, token_count=8),
    )
    config = json.loads((cache_dir / "config.json").read_text(encoding="utf-8"))
    row = torch.load(cache_dir / "sample_000000.pt", map_location="cpu", weights_only=False)
    manifest_row = _jsonl(cache_dir / "manifest.jsonl")[0]

    for metadata in (result, config, row["metadata"], manifest_row):
        assert metadata["prefix_token_count"] == 8
        compression = metadata["prefix_compression"]
        assert compression["hidden_pool"] == WAN_DIT_HIDDEN_POOL_TOKEN_POOL
        assert compression["tokens_per_layer"] == 4
        assert compression["selected_layers"] == [1, 3]


def test_dit_backend_rejects_invalid_future_latent_fill(tmp_path) -> None:
    with pytest.raises(ValueError, match="dit_future_latent_fill must be 'zeros' or 'noise'"):
        precompute_pi05_wan_prefix_tokens(
            _cache_args(
                tmp_path / "cache",
                fake_encoder=False,
                prefix_backend="dit_hidden",
                dit_future_latent_fill="blank",
            ),
            encoder=_InjectedPrefixEncoder(prefix_dim=8),
        )


def test_dit_backend_rejects_invalid_tokens_per_layer(tmp_path) -> None:
    with pytest.raises(ValueError, match="dit_tokens_per_layer must be positive"):
        precompute_pi05_wan_prefix_tokens(
            _cache_args(
                tmp_path / "cache",
                fake_encoder=False,
                prefix_backend="dit_hidden",
                dit_tokens_per_layer=0,
            ),
            encoder=_InjectedPrefixEncoder(prefix_dim=8),
        )


@pytest.mark.parametrize(
    "second_overrides",
    [
        {"dit_future_latent_fill": "noise"},
        {"dit_future_latent_seed": 999},
    ],
)
def test_dit_backend_future_latent_fill_seed_changes_reject_existing_cache_config(
    tmp_path,
    second_overrides,
) -> None:
    cache_dir = tmp_path / "cache"
    base_overrides = {
        "fake_encoder": False,
        "prefix_backend": "dit_hidden",
        "dit_num_latent_frames": 3,
        "dit_future_latent_fill": "zeros",
        "dit_future_latent_seed": 0,
    }

    precompute_pi05_wan_prefix_tokens(
        _cache_args(cache_dir, **base_overrides),
        encoder=_InjectedPrefixEncoder(prefix_dim=8),
    )

    requested_overrides = {**base_overrides, **second_overrides}
    with pytest.raises(ValueError, match="metadata mismatch for prefix_compression"):
        precompute_pi05_wan_prefix_tokens(
            _cache_args(cache_dir, **requested_overrides),
            encoder=_InjectedPrefixEncoder(prefix_dim=8),
        )


@pytest.mark.parametrize(
    "second_overrides",
    [
        {"dit_tokens_per_layer": 2},
        {"dit_hidden_pool": "mean"},
    ],
)
def test_dit_backend_token_pool_settings_change_reject_existing_cache_config(
    tmp_path,
    second_overrides,
) -> None:
    cache_dir = tmp_path / "cache"
    base_overrides = {
        "fake_encoder": False,
        "prefix_backend": "dit_hidden",
        "dit_selected_layers": (1, 3),
        "dit_hidden_pool": WAN_DIT_HIDDEN_POOL_TOKEN_POOL,
        "dit_tokens_per_layer": 4,
    }

    precompute_pi05_wan_prefix_tokens(
        _cache_args(cache_dir, **base_overrides),
        encoder=_InjectedPrefixEncoder(prefix_dim=8, token_count=8),
    )

    requested_overrides = {**base_overrides, **second_overrides}
    with pytest.raises(ValueError, match="metadata mismatch for prefix_compression"):
        precompute_pi05_wan_prefix_tokens(
            _cache_args(cache_dir, **requested_overrides),
            encoder=_InjectedPrefixEncoder(prefix_dim=8, token_count=8),
        )


def test_fake_encoder_keeps_fake_source_when_dit_backend_is_requested(tmp_path) -> None:
    cache_dir = tmp_path / "cache"

    result = precompute_pi05_wan_prefix_tokens(_cache_args(cache_dir, prefix_backend="dit_hidden"))
    row = torch.load(cache_dir / "sample_000000.pt", map_location="cpu", weights_only=False)
    manifest_row = _jsonl(cache_dir / "manifest.jsonl")[0]
    config = json.loads((cache_dir / "config.json").read_text(encoding="utf-8"))

    assert result["source"] == "fake"
    assert result["wan_action_mode"] == NON_WAN_ACTION_MODE
    assert config["source"] == "fake"
    assert config["wan_action_mode"] == NON_WAN_ACTION_MODE
    assert row["metadata"]["source"] == "fake"
    assert row["metadata"]["wan_action_mode"] == NON_WAN_ACTION_MODE
    assert manifest_row["source"] == "fake"
    assert manifest_row["wan_action_mode"] == NON_WAN_ACTION_MODE
    assert config["prefix_compression"] == {
        "text": "masked_mean_pool_then_adaptive_avg_pool1d_or_pad_to_prefix_dim",
        "image": "wan_vae_latents_flattened_T_H_W_to_tokens_with_last_dim_fit_to_prefix_dim",
    }


def test_dit_backend_builder_uses_dit_encoder_args_without_loading_weights(monkeypatch, tmp_path) -> None:
    captured_kwargs = {}

    class _FakeDitEncoder:
        def __init__(self, **kwargs) -> None:
            captured_kwargs.update(kwargs)
            self.prefix_dim = kwargs["prefix_dim"]

        def encode_prefix(self, current_images: torch.Tensor, prompts: list[str]) -> torch.Tensor:
            return torch.zeros((current_images.shape[0], 1, self.prefix_dim), dtype=torch.float32)

    monkeypatch.setattr(cache_module, "FrozenDiffSynthWanDiTCurrentPrefixEncoder", _FakeDitEncoder)

    encoder = cache_module._build_encoder(
        _cache_args(
            tmp_path / "cache",
            fake_encoder=False,
            prefix_backend="dit_hidden",
            prefix_dim=12,
            wan_repo_dir="/fake/repo",
            wan_checkpoint_dir="/fake/checkpoint",
            wan_vae_checkpoint_path="/fake/vae.pth",
            wan_text_encoder_checkpoint_path="/fake/t5.pth",
            wan_tokenizer_dir="/fake/tokenizer",
            wan_dtype="float32",
            wan_tiled=True,
            dit_selected_layers=(1, 3),
            dit_hidden_pool=WAN_DIT_HIDDEN_POOL_TOKEN_POOL,
            dit_tokens_per_layer=4,
            dit_num_latent_frames=2,
            dit_timestep=250.0,
            dit_future_latent_fill="noise",
            dit_future_latent_seed=1234,
        )
    )

    assert isinstance(encoder, _FakeDitEncoder)
    assert captured_kwargs == {
        "repo_dir": "/fake/repo",
        "checkpoint_dir": "/fake/checkpoint",
        "vae_checkpoint_path": "/fake/vae.pth",
        "text_encoder_checkpoint_path": "/fake/t5.pth",
        "tokenizer_dir": "/fake/tokenizer",
        "selected_layers": (1, 3),
        "hidden_pool": WAN_DIT_HIDDEN_POOL_TOKEN_POOL,
        "tokens_per_layer": 4,
        "prefix_dim": 12,
        "dtype": "float32",
        "timestep": 250.0,
        "num_latent_frames": 2,
        "future_latent_fill": "noise",
        "future_latent_seed": 1234,
        "tiled": True,
    }


def test_wan_prefix_cache_has_no_future_leakage_keys_and_manifest_is_current_only(tmp_path) -> None:
    cache_dir = tmp_path / "cache"

    precompute_pi05_wan_prefix_tokens(_cache_args(cache_dir))
    row = torch.load(cache_dir / "sample_000001.pt", map_location="cpu", weights_only=False)
    manifest_rows = _jsonl(cache_dir / "manifest.jsonl")
    config = json.loads((cache_dir / "config.json").read_text(encoding="utf-8"))

    assert _forbidden_paths(row) == []
    assert manifest_rows
    assert all(manifest_row["contains_future_images"] is False for manifest_row in manifest_rows)
    assert all(manifest_row["contains_future_latents"] is False for manifest_row in manifest_rows)
    assert all(manifest_row["contains_future_ground_truth_latents"] is False for manifest_row in manifest_rows)
    assert all(manifest_row["uses_future_latent_slots"] is False for manifest_row in manifest_rows)
    assert all(manifest_row["native_wan_attention_kv_cache"] is False for manifest_row in manifest_rows)
    assert config["contains_future_images"] is False
    assert config["contains_future_latents"] is False
    assert config["contains_future_ground_truth_latents"] is False
    assert config["uses_future_latent_slots"] is False
    assert config["native_wan_attention_kv_cache"] is False


def test_wan_prefix_cache_records_batch_source_provenance(tmp_path) -> None:
    cache_dir = tmp_path / "cache"

    precompute_pi05_wan_prefix_tokens(_cache_args(cache_dir, synthetic_samples=18, batch_size=5))
    row = torch.load(cache_dir / "sample_000017.pt", map_location="cpu", weights_only=False)
    manifest_row = _jsonl(cache_dir / "manifest.jsonl")[17]
    expected_provenance = {
        "cache_index": 17,
        "source_dataset_index": 17,
        "episode_index": 1,
        "frame_index": 1,
        "task_index": 1,
    }

    assert row["metadata"]["dataset_index"] == 17
    assert manifest_row["dataset_index"] == 17
    for key, value in expected_provenance.items():
        assert row["metadata"][key] == value
        assert manifest_row[key] == value


def test_wan_prefix_cache_resumes_from_manifest_without_rewriting(tmp_path) -> None:
    cache_dir = tmp_path / "cache"

    first = precompute_pi05_wan_prefix_tokens(_cache_args(cache_dir, synthetic_samples=3))
    first_manifest = (cache_dir / "manifest.jsonl").read_text(encoding="utf-8")
    second = precompute_pi05_wan_prefix_tokens(_cache_args(cache_dir, synthetic_samples=3))
    second_manifest = (cache_dir / "manifest.jsonl").read_text(encoding="utf-8")

    assert first["written"] == 3
    assert second["written"] == 0
    assert first_manifest == second_manifest
    assert len(_jsonl(cache_dir / "manifest.jsonl")) == 3


def test_fake_prefix_encoder_is_current_only_and_accepts_no_future_images() -> None:
    encoder = FakeWanCurrentPrefixEncoder(prefix_dim=8, spatial_stride=16)
    current_images = torch.linspace(0.0, 1.0, 2 * 3 * 16 * 16).reshape(2, 3, 16, 16)
    prompts = ["pick object", "open drawer"]
    future_images_a = torch.zeros(2, 4, 3, 16, 16)
    future_images_b = torch.ones_like(future_images_a)

    prefix_a = encoder.encode_prefix(current_images, prompts)
    del future_images_a
    del future_images_b
    prefix_b = encoder.encode_prefix(current_images, prompts)

    assert torch.allclose(prefix_a, prefix_b)
    assert "future_images" not in inspect.signature(encoder.encode_prefix).parameters
    with pytest.raises(TypeError):
        encoder.encode_prefix(current_images, prompts, future_images=torch.zeros(1))


def test_missing_real_wan_paths_fail_clearly_without_fake_fallback(tmp_path) -> None:
    repo_dir = tmp_path / "DiffSynth-Studio"
    (repo_dir / "diffsynth").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="Wan2.2 VAE checkpoint not found"):
        FrozenDiffSynthWanCurrentPrefixEncoder(
            repo_dir=repo_dir,
            checkpoint_dir=tmp_path / "missing-wan2.2-ti2v-5b",
            prefix_dim=8,
        )


def test_train_pi05_wan_action_expert_consumes_fake_wan_prefix_cache(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "train"
    precompute_pi05_wan_prefix_tokens(_cache_args(cache_dir))

    metrics = run_train_eval(
        TrainArgs(
            cache_path=str(cache_dir),
            output_dir=str(output_dir),
            epochs=1,
            batch_size=2,
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
            sample_steps=2,
            device="cpu",
            seed=17,
        )
    )

    assert metrics["num_train"] == 4
    assert metrics["num_val"] == 2
    assert metrics["val_model_sample_mse"] >= 0.0
    assert metrics["val_mean_action_mse"] >= 0.0
    assert (output_dir / "checkpoint.pt").exists()


def test_pi05_wan_action_loss_weighting_resolves_clipped_and_normalized_weights() -> None:
    action_norm_std = torch.tensor([2.0, 3.0, 5.0, 0.5])

    clipped = _resolve_action_loss_weights(
        weighting="clipped_original_scale",
        normalize_actions=True,
        action_norm_std=action_norm_std,
        action_dim=4,
        device=torch.device("cpu"),
        action_loss_weight_max=4.0,
    )
    normalized = _resolve_action_loss_weights(
        weighting="normalized_original_scale",
        normalize_actions=True,
        action_norm_std=action_norm_std,
        action_dim=4,
        device=torch.device("cpu"),
        action_loss_weight_max=4.0,
    )

    assert torch.allclose(clipped, torch.tensor([4.0, 4.0, 4.0, 0.25]))
    assert torch.allclose(normalized.mean(), torch.tensor(1.0))
    assert torch.allclose(normalized, action_norm_std.square() / action_norm_std.square().mean())


def test_pi05_wan_action_loss_weighting_requires_normalization_and_positive_clip() -> None:
    action_norm_std = torch.ones(4)

    with pytest.raises(ValueError, match="requires normalize_actions=True"):
        _resolve_action_loss_weights(
            weighting="normalized_original_scale",
            normalize_actions=False,
            action_norm_std=None,
            action_dim=4,
            device=torch.device("cpu"),
            action_loss_weight_max=4.0,
        )
    with pytest.raises(ValueError, match="action_loss_weight_max must be positive"):
        _resolve_action_loss_weights(
            weighting="clipped_original_scale",
            normalize_actions=True,
            action_norm_std=action_norm_std,
            action_dim=4,
            device=torch.device("cpu"),
            action_loss_weight_max=0.0,
        )


def test_train_pi05_wan_action_expert_records_clipped_loss_weight_metadata(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "train"
    precompute_pi05_wan_prefix_tokens(_cache_args(cache_dir))

    metrics = run_train_eval(
        TrainArgs(
            cache_path=str(cache_dir),
            output_dir=str(output_dir),
            epochs=1,
            batch_size=2,
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
            sample_steps=2,
            normalize_actions=True,
            action_loss_weighting="clipped_original_scale",
            action_loss_weight_max=2.0,
            device="cpu",
            seed=17,
        )
    )
    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location="cpu", weights_only=False)

    assert metrics["action_loss_weighting"] == "clipped_original_scale"
    assert metrics["action_loss_weights_source"] == "action_norm_std_squared_clipped"
    assert metrics["action_loss_weight_max"] == 2.0
    assert max(metrics["action_loss_weights"]) <= 2.0
    assert checkpoint["action_loss"]["weighting"] == "clipped_original_scale"
    assert checkpoint["action_loss"]["weight_max"] == 2.0


def test_train_pi05_wan_action_expert_records_per_task_action_normalization(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "train"
    precompute_pi05_wan_prefix_tokens(_cache_args(cache_dir, synthetic_samples=9))

    metrics = run_train_eval(
        TrainArgs(
            cache_path=str(cache_dir),
            eval_cache_path=str(cache_dir),
            output_dir=str(output_dir),
            epochs=1,
            batch_size=3,
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
            sample_steps=2,
            normalize_actions=True,
            action_normalization_scope="per_task",
            device="cpu",
            seed=17,
        )
    )
    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location="cpu", weights_only=False)

    assert metrics["action_normalization_scope"] == "per_task"
    assert metrics["action_normalization_task_count"] == 9
    assert len(metrics["action_normalization_tasks"]) == metrics["action_normalization_task_count"]
    assert checkpoint["action_normalization"]["enabled"] is True
    assert checkpoint["action_normalization"]["scope"] == "per_task"
    assert sorted(checkpoint["action_normalization"]["tasks"]) == metrics["action_normalization_tasks"]
