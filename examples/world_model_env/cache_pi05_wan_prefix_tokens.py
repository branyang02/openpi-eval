from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, TextIO

import torch
import torch.nn.functional as F
import tyro
from torch.utils.data import DataLoader

from world_model.config import DatasetConfig, DatasetSource
from world_model.data import create_dataset
from world_model.train_lib import resolve_device
from world_model.wan_dit_prefix_encoder import (
    DEFAULT_WAN_DIT_LAYERS,
    DIFFSYNTH_WAN22_TI2V_DIT_SOURCE,
    WAN_DIT_FUTURE_LATENT_NOISE_SEED_STRATEGY,
    WAN_DIT_HIDDEN_POOL_DESCRIPTION,
    WAN_DIT_HIDDEN_POOL_MEAN,
    WAN_DIT_HIDDEN_POOL_TOKEN_POOL,
    FrozenDiffSynthWanDiTCurrentPrefixEncoder,
    wan_dit_future_latent_noise_seed,
)
from world_model.wan_prefix_encoder import (
    DEFAULT_WAN_CHECKPOINT_DIR,
    DEFAULT_WAN_REPO_DIR,
    DIFFSYNTH_WAN22_TI2V_CURRENT_SOURCE,
    TEXT_COMPRESSION_DESCRIPTION,
    FakeWanCurrentPrefixEncoder,
    FrozenDiffSynthWanCurrentPrefixEncoder,
    WanCurrentPrefixEncoder,
)

PI05_WAN_CURRENT_PREFIX_CACHE_KIND = "pi05_wan_current_prefix"
CURRENT_WAN_PREFIX_ACTION_MODE = "current_wan_prefix_action_expert"
PARTIAL_WAN_PREFIX_ACTION_MODE = "partial_wan_prefix_action_expert"
NON_WAN_CURRENT_PREFIX_ACTION_MODE = "non_wan_current_prefix_baseline"
PrefixBackend = Literal["vae_text", "dit_hidden", "raw_current"]
VAE_TEXT_IMAGE_COMPRESSION_DESCRIPTION = "wan_vae_latents_flattened_T_H_W_to_tokens_with_last_dim_fit_to_prefix_dim"
RAW_CURRENT_SOURCE = "raw_current_image_prompt"
RAW_CURRENT_PREFIX_TOKEN_COUNT = 3
RAW_CURRENT_PREFIX_COMPRESSION = {
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


@dataclasses.dataclass
class Args:
    dataset_source: DatasetSource = "synthetic"
    repo_id: str = "brandonyang/metaworld_ml45"
    image_key: str = "corner4.image"
    output_dir: str = "output/pi05_wan_prefix_cache"
    episodes: tuple[int, ...] | None = None
    max_samples: int | None = None
    samples_per_episode: int | None = None
    synthetic_samples: int = 8
    image_size: int = 64
    action_horizon: int = 8
    batch_size: int = 4
    num_workers: int = 0
    device: str = "auto"
    seed: int = 7
    prefix_dim: int = 48
    fake_encoder: bool = False
    fake_spatial_stride: int = 16
    prefix_backend: PrefixBackend = "vae_text"
    wan_repo_dir: str = DEFAULT_WAN_REPO_DIR
    wan_checkpoint_dir: str = DEFAULT_WAN_CHECKPOINT_DIR
    wan_vae_checkpoint_path: str | None = None
    wan_text_encoder_checkpoint_path: str | None = None
    wan_tokenizer_dir: str | None = None
    wan_dtype: str = "bfloat16"
    wan_tiled: bool = False
    dit_selected_layers: tuple[int, ...] = DEFAULT_WAN_DIT_LAYERS
    dit_hidden_pool: Literal["mean", "token_pool"] = "mean"
    dit_tokens_per_layer: int = 1
    dit_num_latent_frames: int = 1
    dit_timestep: float = 500.0
    dit_future_latent_fill: Literal["zeros", "noise"] = "zeros"
    dit_future_latent_seed: int = 0


@dataclasses.dataclass(frozen=True)
class WanPrefixEncoderConfig:
    """Reusable config for current-image prefix encoders shared by cache and server."""

    prefix_dim: int
    fake_encoder: bool = False
    fake_spatial_stride: int = 16
    prefix_backend: PrefixBackend = "vae_text"
    wan_repo_dir: str = DEFAULT_WAN_REPO_DIR
    wan_checkpoint_dir: str = DEFAULT_WAN_CHECKPOINT_DIR
    wan_vae_checkpoint_path: str | None = None
    wan_text_encoder_checkpoint_path: str | None = None
    wan_tokenizer_dir: str | None = None
    wan_dtype: str = "bfloat16"
    wan_tiled: bool = False
    dit_selected_layers: tuple[int, ...] = DEFAULT_WAN_DIT_LAYERS
    dit_hidden_pool: Literal["mean", "token_pool"] = "mean"
    dit_tokens_per_layer: int = 1
    dit_num_latent_frames: int = 1
    dit_timestep: float = 500.0
    dit_future_latent_fill: Literal["zeros", "noise"] = "zeros"
    dit_future_latent_seed: int = 0

    @classmethod
    def from_args(cls, args: Args) -> WanPrefixEncoderConfig:
        return cls(
            prefix_dim=args.prefix_dim,
            fake_encoder=args.fake_encoder,
            fake_spatial_stride=args.fake_spatial_stride,
            prefix_backend=args.prefix_backend,
            wan_repo_dir=args.wan_repo_dir,
            wan_checkpoint_dir=args.wan_checkpoint_dir,
            wan_vae_checkpoint_path=args.wan_vae_checkpoint_path,
            wan_text_encoder_checkpoint_path=args.wan_text_encoder_checkpoint_path,
            wan_tokenizer_dir=args.wan_tokenizer_dir,
            wan_dtype=args.wan_dtype,
            wan_tiled=args.wan_tiled,
            dit_selected_layers=args.dit_selected_layers,
            dit_hidden_pool=args.dit_hidden_pool,
            dit_tokens_per_layer=args.dit_tokens_per_layer,
            dit_num_latent_frames=args.dit_num_latent_frames,
            dit_timestep=args.dit_timestep,
            dit_future_latent_fill=args.dit_future_latent_fill,
            dit_future_latent_seed=args.dit_future_latent_seed,
        )


def _validate_current_encoder_inputs(current_images: torch.Tensor, prompts: Sequence[str]) -> None:
    if current_images.ndim != 4:
        raise ValueError(f"current_images must have shape (B, 3, H, W), got {tuple(current_images.shape)}.")
    if current_images.shape[1] != 3:
        raise ValueError(f"current_images must have 3 RGB channels, got {current_images.shape[1]}.")
    if len(prompts) != current_images.shape[0]:
        raise ValueError(f"prompts length {len(prompts)} must match batch size {current_images.shape[0]}.")


def _fit_last_dim(values: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}.")
    if values.shape[-1] == width:
        return values
    if values.shape[-1] > width:
        original_shape = values.shape[:-1]
        pooled = F.adaptive_avg_pool1d(values.reshape(-1, 1, values.shape[-1]), width)
        return pooled.reshape(*original_shape, width)
    return F.pad(values, (0, width - values.shape[-1]))


def _current_image_float01(current_images: torch.Tensor) -> torch.Tensor:
    images = current_images.to(dtype=torch.float32)
    if not current_images.is_floating_point():
        images = images.div(255.0)
    return images.clamp(0.0, 1.0)


def _downsampled_current_image_token(images: torch.Tensor, prefix_dim: int) -> torch.Tensor:
    _, _, height, width = images.shape
    target_pixels = max(1, math.ceil(prefix_dim / 3))
    aspect = width / max(height, 1)
    pooled_height = max(1, round(math.sqrt(target_pixels / max(aspect, 1.0e-6))))
    pooled_width = max(1, math.ceil(target_pixels / pooled_height))
    pooled_height = min(height, pooled_height)
    pooled_width = min(width, pooled_width)
    pooled = F.adaptive_avg_pool2d(images, (pooled_height, pooled_width))
    return _fit_last_dim(pooled.flatten(start_dim=1), prefix_dim)


def _current_image_statistics_token(images: torch.Tensor, prefix_dim: int) -> torch.Tensor:
    channel_mean = images.mean(dim=(-2, -1))
    channel_std = images.std(dim=(-2, -1), unbiased=False)
    channel_min = images.amin(dim=(-2, -1))
    channel_max = images.amax(dim=(-2, -1))
    flattened = images.flatten(start_dim=1)
    global_stats = torch.stack(
        [
            flattened.mean(dim=1),
            flattened.std(dim=1, unbiased=False),
            flattened.amin(dim=1),
            flattened.amax(dim=1),
        ],
        dim=1,
    )

    histograms = []
    bin_edges = torch.linspace(0.0, 1.0, 9, device=images.device, dtype=images.dtype)
    for bin_index in range(8):
        lower = bin_edges[bin_index]
        upper = bin_edges[bin_index + 1]
        if bin_index == 7:
            in_bin = images.ge(lower) & images.le(upper)
        else:
            in_bin = images.ge(lower) & images.lt(upper)
        histograms.append(in_bin.to(dtype=images.dtype).mean(dim=(-2, -1)))
    histogram = torch.cat(histograms, dim=1)

    stats = torch.cat([channel_mean, channel_std, channel_min, channel_max, global_stats, histogram], dim=1)
    return _fit_last_dim(stats, prefix_dim)


def _prompt_char_ngram_token(prompt: str, prefix_dim: int, *, device: torch.device) -> torch.Tensor:
    text = f"\x02{str(prompt).casefold().strip()}\x03"
    token = torch.zeros(prefix_dim, dtype=torch.float32, device=device)
    feature_count = 0
    for ngram_width in range(1, 5):
        if len(text) < ngram_width:
            continue
        for offset in range(len(text) - ngram_width + 1):
            ngram = text[offset : offset + ngram_width]
            payload = f"{ngram_width}:{offset % 17}:{ngram}".encode("utf-8")
            digest = hashlib.blake2b(payload, digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % prefix_dim
            sign = 1.0 if digest[4] & 1 else -1.0
            token[index] += sign
            feature_count += 1
    if feature_count > 0:
        token = token / math.sqrt(feature_count)
    return token


class RawCurrentImagePromptPrefixEncoder:
    """Deterministic current-only non-Wan prefix baseline for matched cache rows."""

    token_count = RAW_CURRENT_PREFIX_TOKEN_COUNT

    def __init__(self, *, prefix_dim: int = 3072) -> None:
        if prefix_dim <= 0:
            raise ValueError(f"prefix_dim must be positive, got {prefix_dim}.")
        self.prefix_dim = prefix_dim

    @torch.no_grad()
    def encode_prefix(self, current_images: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
        _validate_current_encoder_inputs(current_images, prompts)
        images = _current_image_float01(current_images)
        image_token = _downsampled_current_image_token(images, self.prefix_dim)
        statistics_token = _current_image_statistics_token(images, self.prefix_dim)
        prompt_token = torch.stack(
            [_prompt_char_ngram_token(str(prompt), self.prefix_dim, device=images.device) for prompt in prompts],
            dim=0,
        )
        return torch.stack([image_token, statistics_token, prompt_token], dim=1).to(dtype=torch.float32)


def _relative_to_output(path: str | Path, output_dir: Path) -> str:
    return str(Path(path).expanduser().resolve().relative_to(output_dir.expanduser().resolve()))


def _append_manifest_row(manifest_file: TextIO, row: dict[str, Any]) -> None:
    manifest_file.write(json.dumps(row) + "\n")
    manifest_file.flush()
    os.fsync(manifest_file.fileno())


def _load_manifest_by_index(manifest_path: Path, output_dir: Path) -> dict[int, dict[str, Any]]:
    if not manifest_path.exists():
        return {}
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest_by_index: dict[int, dict[str, Any]] = {}
    for row in rows:
        dataset_index = int(row["dataset_index"])
        row_file = row.get("row_file")
        if row_file is None or (output_dir / str(row_file)).exists():
            manifest_by_index[dataset_index] = row
    return manifest_by_index


def _batch_scalar_int(batch: dict[str, Any], key: str, local_position: int) -> int | None:
    if key not in batch:
        return None
    value = batch[key]
    if isinstance(value, torch.Tensor):
        selected = value.reshape(1) if value.ndim == 0 else value[local_position]
    elif isinstance(value, list | tuple):
        selected = value[local_position]
    else:
        tensor = torch.as_tensor(value)
        selected = tensor.reshape(1) if tensor.ndim == 0 else tensor[local_position]
    tensor = torch.as_tensor(selected).reshape(-1)
    if tensor.numel() == 0:
        return None
    return int(tensor[0].item())


def _batch_provenance_metadata(
    batch: dict[str, Any],
    *,
    cache_index: int,
    local_position: int,
) -> dict[str, int]:
    metadata = {"cache_index": int(cache_index)}
    source_dataset_index = _batch_scalar_int(batch, "dataset_index", local_position)
    if source_dataset_index is not None:
        metadata["source_dataset_index"] = source_dataset_index
    for key in ("episode_index", "frame_index", "task_index"):
        value = _batch_scalar_int(batch, key, local_position)
        if value is not None:
            metadata[key] = value
    return metadata


def _task_text(dataset: Any, index: int, batch: dict[str, Any] | None = None, local_position: int | None = None) -> str:
    if hasattr(dataset, "task_text"):
        return str(dataset.task_text(index))
    if batch is not None and local_position is not None and "task" in batch:
        task = batch["task"][local_position]
        if isinstance(task, (list, tuple)):
            task = task[0]
        return str(task)
    if batch is not None and local_position is not None and "task_id" in batch:
        task_id = torch.as_tensor(batch["task_id"][local_position]).reshape(-1)[0]
        return f"metaworld task id {int(task_id)}"
    return ""


def _build_dataset_config(args: Args) -> DatasetConfig:
    return DatasetConfig(
        source=args.dataset_source,
        repo_id=args.repo_id,
        image_keys=(args.image_key,),
        image_size=args.image_size,
        frame_delta=1,
        num_future_frames=1,
        action_horizon=args.action_horizon,
        idm_history_length=0,
        max_samples=args.max_samples,
        samples_per_episode=args.samples_per_episode,
        synthetic_samples=args.synthetic_samples,
        episodes=args.episodes,
        seed=args.seed,
    )


def _source_name(args: Args) -> str:
    if args.fake_encoder:
        return "fake"
    if args.prefix_backend == "raw_current":
        return RAW_CURRENT_SOURCE
    if args.prefix_backend == "vae_text":
        return DIFFSYNTH_WAN22_TI2V_CURRENT_SOURCE
    if args.prefix_backend == "dit_hidden":
        return DIFFSYNTH_WAN22_TI2V_DIT_SOURCE
    raise ValueError(f"Unsupported prefix backend: {args.prefix_backend!r}.")


def _wan_action_mode(args: Args) -> str:
    if args.fake_encoder or args.prefix_backend == "raw_current":
        return NON_WAN_CURRENT_PREFIX_ACTION_MODE
    if args.prefix_backend == "vae_text":
        return CURRENT_WAN_PREFIX_ACTION_MODE
    if args.prefix_backend == "dit_hidden":
        if args.dit_num_latent_frames == 1:
            return CURRENT_WAN_PREFIX_ACTION_MODE
        return PARTIAL_WAN_PREFIX_ACTION_MODE
    raise ValueError(f"Unsupported prefix backend: {args.prefix_backend!r}.")


def _validate_dit_metadata_args(args: Args) -> None:
    if not args.dit_selected_layers:
        raise ValueError("dit_selected_layers must contain at least one layer index.")
    for layer in args.dit_selected_layers:
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise ValueError(f"dit_selected_layers must contain integer layer indices, got {layer!r}.")
        if layer < 0:
            raise ValueError(f"dit_selected_layers must be non-negative, got {layer}.")
    if len(set(args.dit_selected_layers)) != len(args.dit_selected_layers):
        raise ValueError(f"dit_selected_layers must not contain duplicates, got {args.dit_selected_layers}.")
    if args.dit_hidden_pool not in (WAN_DIT_HIDDEN_POOL_MEAN, WAN_DIT_HIDDEN_POOL_TOKEN_POOL):
        raise ValueError(f"dit_hidden_pool must be 'mean' or 'token_pool', got {args.dit_hidden_pool!r}.")
    if isinstance(args.dit_tokens_per_layer, bool) or not isinstance(args.dit_tokens_per_layer, int):
        raise ValueError(f"dit_tokens_per_layer must be an integer, got {args.dit_tokens_per_layer!r}.")
    if args.dit_tokens_per_layer <= 0:
        raise ValueError(f"dit_tokens_per_layer must be positive, got {args.dit_tokens_per_layer}.")
    if args.dit_num_latent_frames <= 0:
        raise ValueError(f"dit_num_latent_frames must be positive, got {args.dit_num_latent_frames}.")
    if args.dit_future_latent_fill not in ("zeros", "noise"):
        raise ValueError(f"dit_future_latent_fill must be 'zeros' or 'noise', got {args.dit_future_latent_fill!r}.")
    if isinstance(args.dit_future_latent_seed, bool) or not isinstance(args.dit_future_latent_seed, int):
        raise ValueError(f"dit_future_latent_seed must be an integer, got {args.dit_future_latent_seed!r}.")
    if args.dit_future_latent_seed < 0:
        raise ValueError(f"dit_future_latent_seed must be non-negative, got {args.dit_future_latent_seed}.")


def _dit_timestep_metadata(args: Args) -> dict[str, Any]:
    configured_timestep = float(args.dit_timestep)
    future_latent_frames = max(int(args.dit_num_latent_frames) - 1, 0)
    timestep_applies_to_future_latents_only = True
    effective_timestep = 0.0 if future_latent_frames == 0 else configured_timestep
    return {
        "timestep": configured_timestep,
        "configured_timestep": configured_timestep,
        "timestep_shape": [1],
        "timestep_applies_to_future_latents_only": timestep_applies_to_future_latents_only,
        "future_latent_frames": future_latent_frames,
        "effective_timestep": effective_timestep,
    }


def _dit_future_slot_conditioning_metadata(args: Args) -> dict[str, Any]:
    future_latent_frames = max(int(args.dit_num_latent_frames) - 1, 0)
    if future_latent_frames == 0:
        conditioning = "none_current_latent_slot_only"
    elif args.dit_future_latent_fill == "noise":
        conditioning = "deterministic_per_sample_noise_placeholders"
    else:
        conditioning = "zero_placeholders"

    metadata: dict[str, Any] = {
        "future_slot_conditioning": conditioning,
        "uses_future_ground_truth_latents": False,
        "stores_future_ground_truth_latents": False,
    }
    if future_latent_frames > 0 and args.dit_future_latent_fill == "noise":
        metadata.update(
            {
                "future_slot_noise_seed_key": "dataset_index",
                "future_slot_noise_seed_strategy": WAN_DIT_FUTURE_LATENT_NOISE_SEED_STRATEGY,
            }
        )
    return metadata


def _dit_row_future_slot_conditioning_metadata(args: Args, *, dataset_index: int) -> dict[str, Any]:
    if args.fake_encoder or args.prefix_backend != "dit_hidden":
        return {}
    if args.dit_num_latent_frames <= 1 or args.dit_future_latent_fill != "noise":
        return {}
    return {
        "dit_future_slot_noise_seed": wan_dit_future_latent_noise_seed(
            int(args.dit_future_latent_seed),
            int(dataset_index),
        ),
        "dit_future_slot_noise_seed_key": "dataset_index",
        "dit_future_slot_noise_seed_strategy": WAN_DIT_FUTURE_LATENT_NOISE_SEED_STRATEGY,
        "dit_future_slot_conditioning": "deterministic_per_sample_noise_placeholders",
        "dit_stores_future_ground_truth_latents": False,
    }


def _dit_hidden_pool_metadata(args: Args) -> str:
    if args.dit_hidden_pool == WAN_DIT_HIDDEN_POOL_MEAN:
        return WAN_DIT_HIDDEN_POOL_DESCRIPTION
    return WAN_DIT_HIDDEN_POOL_TOKEN_POOL


def _dit_tokens_per_layer_metadata(args: Args) -> int:
    if args.dit_hidden_pool == WAN_DIT_HIDDEN_POOL_MEAN:
        return 1
    return int(args.dit_tokens_per_layer)


def _prefix_compression(args: Args) -> dict[str, Any]:
    if args.fake_encoder or args.prefix_backend == "vae_text":
        return {
            "text": TEXT_COMPRESSION_DESCRIPTION,
            "image": VAE_TEXT_IMAGE_COMPRESSION_DESCRIPTION,
        }
    if args.prefix_backend == "raw_current":
        return dict(RAW_CURRENT_PREFIX_COMPRESSION)
    if args.prefix_backend == "dit_hidden":
        _validate_dit_metadata_args(args)
        tokens_per_layer = _dit_tokens_per_layer_metadata(args)
        future_latent_frames = max(int(args.dit_num_latent_frames) - 1, 0)
        if future_latent_frames == 0:
            latent_source = "current-only fused latents"
        elif args.dit_future_latent_fill == "noise":
            latent_source = (
                f"the current latent plus {future_latent_frames} deterministic per-sample noise future latent slot(s)"
            )
        else:
            latent_source = f"the current latent plus {future_latent_frames} zero future latent slot(s)"
        description = (
            f"selected Wan DiT hidden states pooled from {latent_source}; "
            "one prefix token per selected DiT layer, with the hidden width fit to prefix_dim"
        )
        if args.dit_hidden_pool == WAN_DIT_HIDDEN_POOL_TOKEN_POOL:
            description = (
                f"selected Wan DiT hidden states adaptive-average-pooled from {latent_source}; "
                f"{tokens_per_layer} prefix token(s) per selected DiT layer, with the hidden width fit to prefix_dim"
            )
        return {
            "description": description,
            "hidden_pool": _dit_hidden_pool_metadata(args),
            "tokens_per_layer": tokens_per_layer,
            "selected_layers": list(args.dit_selected_layers),
            "num_latent_frames": args.dit_num_latent_frames,
            **_dit_timestep_metadata(args),
            "future_latent_fill": args.dit_future_latent_fill,
            "future_latent_seed": args.dit_future_latent_seed,
            **_dit_future_slot_conditioning_metadata(args),
            "fuse_vae_embedding_in_latents": True,
        }
    raise ValueError(f"Unsupported prefix backend: {args.prefix_backend!r}.")


def _json_normalized(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _future_latent_slot_count(args: Args) -> int:
    if args.prefix_backend != "dit_hidden":
        return 0
    return max(int(args.dit_num_latent_frames) - 1, 0)


def _prefix_contract_metadata(args: Args) -> dict[str, Any]:
    future_latent_slot_count = _future_latent_slot_count(args)
    return {
        "contains_future_ground_truth_latents": False,
        "uses_future_latent_slots": future_latent_slot_count > 0,
        "future_latent_slot_count": future_latent_slot_count,
        "wan_backbone_runs_per_observation": 1,
        "native_wan_attention_kv_cache": False,
    }


def _prefix_metadata(
    *,
    args: Args,
    dataset_config: DatasetConfig,
    num_dataset_samples: int,
    prefix_token_count: int,
) -> dict[str, Any]:
    wan_action_mode = _wan_action_mode(args)
    return {
        "cache_kind": PI05_WAN_CURRENT_PREFIX_CACHE_KIND,
        "contains_future_images": False,
        "contains_future_latents": False,
        **_prefix_contract_metadata(args),
        "prefix_dim": args.prefix_dim,
        "prefix_token_count": prefix_token_count,
        "image_key": args.image_key,
        "image_keys": [args.image_key],
        "source": _source_name(args),
        "wan_action_mode": wan_action_mode,
        "num_dataset_samples": num_dataset_samples,
        "dataset_config": dataclasses.asdict(dataset_config),
        "prefix_compression": _prefix_compression(args),
    }


def _validate_existing_config(
    existing: dict[str, Any],
    *,
    output_dir: Path,
    expected: dict[str, Any],
) -> None:
    for key in (
        "cache_kind",
        "contains_future_images",
        "contains_future_latents",
        "prefix_dim",
        "prefix_token_count",
        "image_key",
        "source",
        "wan_action_mode",
        "prefix_compression",
    ):
        if _json_normalized(existing.get(key)) != _json_normalized(expected.get(key)):
            raise ValueError(
                f"Wan prefix cache metadata mismatch for {key} in {output_dir}: "
                f"cached={existing.get(key)!r}, requested={expected.get(key)!r}."
            )
    existing_dataset = existing.get("dataset_config")
    expected_dataset = expected.get("dataset_config")
    if _json_normalized(existing_dataset) != _json_normalized(expected_dataset):
        raise ValueError(f"Wan prefix cache dataset_config mismatch in {output_dir}.")


def _validate_or_write_config(*, output_dir: Path, metadata: dict[str, Any]) -> None:
    config_path = output_dir / "config.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return
    existing = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(existing, dict):
        raise ValueError(f"Wan prefix cache config must be a JSON object: {config_path}")
    _validate_existing_config(existing, output_dir=output_dir, expected=metadata)


def build_wan_prefix_encoder(config: WanPrefixEncoderConfig) -> WanCurrentPrefixEncoder:
    if config.prefix_dim <= 0:
        raise ValueError(f"prefix_dim must be positive, got {config.prefix_dim}.")
    if config.fake_encoder:
        return FakeWanCurrentPrefixEncoder(prefix_dim=config.prefix_dim, spatial_stride=config.fake_spatial_stride)
    if config.prefix_backend == "raw_current":
        return RawCurrentImagePromptPrefixEncoder(prefix_dim=config.prefix_dim)
    if config.prefix_backend == "vae_text":
        return FrozenDiffSynthWanCurrentPrefixEncoder(
            repo_dir=config.wan_repo_dir,
            checkpoint_dir=config.wan_checkpoint_dir,
            vae_checkpoint_path=config.wan_vae_checkpoint_path,
            text_encoder_checkpoint_path=config.wan_text_encoder_checkpoint_path,
            tokenizer_dir=config.wan_tokenizer_dir,
            prefix_dim=config.prefix_dim,
            dtype=config.wan_dtype,
            tiled=config.wan_tiled,
        )
    if config.prefix_backend == "dit_hidden":
        _validate_dit_metadata_args(config)
        return FrozenDiffSynthWanDiTCurrentPrefixEncoder(
            repo_dir=config.wan_repo_dir,
            checkpoint_dir=config.wan_checkpoint_dir,
            vae_checkpoint_path=config.wan_vae_checkpoint_path,
            text_encoder_checkpoint_path=config.wan_text_encoder_checkpoint_path,
            tokenizer_dir=config.wan_tokenizer_dir,
            selected_layers=config.dit_selected_layers,
            prefix_dim=config.prefix_dim,
            dtype=config.wan_dtype,
            timestep=config.dit_timestep,
            num_latent_frames=config.dit_num_latent_frames,
            future_latent_fill=config.dit_future_latent_fill,
            future_latent_seed=config.dit_future_latent_seed,
            hidden_pool=config.dit_hidden_pool,
            tokens_per_layer=config.dit_tokens_per_layer,
            tiled=config.wan_tiled,
        )
    raise ValueError(f"Unsupported prefix backend: {config.prefix_backend!r}.")


def _build_encoder(args: Args) -> WanCurrentPrefixEncoder:
    return build_wan_prefix_encoder(WanPrefixEncoderConfig.from_args(args))


def _encode_prefix_accepts_sample_indices(prefix_encoder: WanCurrentPrefixEncoder) -> bool:
    parameters = inspect.signature(prefix_encoder.encode_prefix).parameters
    return "sample_indices" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def _should_pass_dit_sample_indices(args: Args, prefix_encoder: WanCurrentPrefixEncoder) -> bool:
    if args.fake_encoder or args.prefix_backend != "dit_hidden":
        return False
    if args.dit_num_latent_frames <= 1 or args.dit_future_latent_fill != "noise":
        return False
    return _encode_prefix_accepts_sample_indices(prefix_encoder)


def _manifest_prefix_token_count(manifest_by_index: dict[int, dict[str, Any]]) -> int | None:
    for row in manifest_by_index.values():
        prefix_shape = row.get("prefix_shape")
        if isinstance(prefix_shape, list | tuple) and prefix_shape:
            return int(prefix_shape[0])
    return None


def _existing_config_prefix_token_count(config_path: Path) -> int | None:
    if not config_path.exists():
        return None
    existing = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(existing, dict):
        raise ValueError(f"Wan prefix cache config must be a JSON object: {config_path}")
    value = existing.get("prefix_token_count")
    return None if value is None else int(value)


def _write_sample_row(
    *,
    path: Path,
    prefix_tokens: torch.Tensor,
    state: torch.Tensor,
    actions: torch.Tensor,
    action_mask: torch.Tensor,
    task: str,
    metadata: dict[str, Any],
) -> None:
    row = {
        "prefix_tokens": prefix_tokens.detach().cpu().to(dtype=torch.float32),
        "state": state.detach().cpu().to(dtype=torch.float32),
        "actions": actions.detach().cpu().to(dtype=torch.float32),
        "action_mask": action_mask.detach().cpu().to(dtype=torch.float32),
        "task": task,
        "metadata": metadata,
    }
    if set(row) != {"prefix_tokens", "state", "actions", "action_mask", "task", "metadata"}:
        raise RuntimeError("Internal error: Wan prefix cache row keys drifted.")
    torch.save(row, path)


@torch.no_grad()
def precompute_pi05_wan_prefix_tokens(
    args: Args,
    *,
    encoder: WanCurrentPrefixEncoder | None = None,
) -> dict[str, Any]:
    if args.prefix_dim <= 0:
        raise ValueError(f"prefix_dim must be positive, got {args.prefix_dim}.")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}.")
    if not args.fake_encoder and args.prefix_backend == "dit_hidden":
        _validate_dit_metadata_args(args)

    dataset_config = _build_dataset_config(args)
    dataset = create_dataset(dataset_config)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.jsonl"
    manifest_by_index = _load_manifest_by_index(manifest_path, output_dir)
    config_path = output_dir / "config.json"
    prefix_token_count = _existing_config_prefix_token_count(config_path) or _manifest_prefix_token_count(
        manifest_by_index
    )
    if prefix_token_count is not None:
        metadata = _prefix_metadata(
            args=args,
            dataset_config=dataset_config,
            num_dataset_samples=len(dataset),
            prefix_token_count=prefix_token_count,
        )
        _validate_or_write_config(output_dir=output_dir, metadata=metadata)

    device = resolve_device(args.device)
    prefix_encoder = encoder if encoder is not None else _build_encoder(args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    written = 0
    cursor = 0
    with manifest_path.open("a", encoding="utf-8") as manifest_file:
        for batch in loader:
            batch_size = int(batch["current_images"].shape[0])
            indices = list(range(cursor, cursor + batch_size))
            cursor += batch_size
            uncached_positions = [position for position, index in enumerate(indices) if index not in manifest_by_index]
            if not uncached_positions:
                continue

            current_images = batch["current_images"][uncached_positions]
            if current_images.ndim != 5:
                raise ValueError(
                    f"current_images must have shape (B, num_views, 3, H, W), got {tuple(current_images.shape)}."
                )
            if current_images.shape[1] != 1:
                raise ValueError("Pi0.5 Wan prefix caching requires exactly one image view.")
            current_view = current_images[:, 0].to(device=device, non_blocking=True)
            prompts = [
                _task_text(dataset, indices[position], batch=batch, local_position=position)
                for position in uncached_positions
            ]
            sample_indices = [indices[position] for position in uncached_positions]
            encode_kwargs: dict[str, Any] = {}
            if _should_pass_dit_sample_indices(args, prefix_encoder):
                encode_kwargs["sample_indices"] = sample_indices
            prefix_tokens = (
                prefix_encoder.encode_prefix(current_view, prompts, **encode_kwargs)
                .detach()
                .cpu()
                .to(dtype=torch.float32)
            )
            if prefix_tokens.ndim != 3 or prefix_tokens.shape[-1] != args.prefix_dim:
                raise ValueError(
                    "Wan prefix encoder returned unexpected shape "
                    f"{tuple(prefix_tokens.shape)}; expected (B, N, {args.prefix_dim})."
                )

            if prefix_token_count is None:
                prefix_token_count = int(prefix_tokens.shape[1])
                metadata = _prefix_metadata(
                    args=args,
                    dataset_config=dataset_config,
                    num_dataset_samples=len(dataset),
                    prefix_token_count=prefix_token_count,
                )
                _validate_or_write_config(output_dir=output_dir, metadata=metadata)
            elif int(prefix_tokens.shape[1]) != prefix_token_count:
                raise ValueError(
                    f"Wan prefix token count changed from {prefix_token_count} to {int(prefix_tokens.shape[1])}."
                )

            for local_offset, position in enumerate(uncached_positions):
                dataset_index = indices[position]
                stem = f"sample_{dataset_index:06d}"
                row_path = output_dir / f"{stem}.pt"
                task = prompts[local_offset]
                provenance_metadata = _batch_provenance_metadata(
                    batch,
                    cache_index=dataset_index,
                    local_position=position,
                )
                row_metadata = {
                    "cache_kind": PI05_WAN_CURRENT_PREFIX_CACHE_KIND,
                    "contains_future_images": False,
                    "contains_future_latents": False,
                    **_prefix_contract_metadata(args),
                    "prefix_dim": args.prefix_dim,
                    "prefix_token_count": prefix_token_count,
                    "image_key": args.image_key,
                    "source": _source_name(args),
                    "wan_action_mode": _wan_action_mode(args),
                    "dataset_index": dataset_index,
                    "task": task,
                    **provenance_metadata,
                    **_dit_row_future_slot_conditioning_metadata(args, dataset_index=dataset_index),
                    "prefix_compression": _prefix_compression(args),
                }
                _write_sample_row(
                    path=row_path,
                    prefix_tokens=prefix_tokens[local_offset],
                    state=batch["state"][position],
                    actions=batch["action_chunk"][position],
                    action_mask=batch["action_mask"][position],
                    task=task,
                    metadata=row_metadata,
                )
                manifest_row = {
                    "dataset_index": dataset_index,
                    "row_file": _relative_to_output(row_path, output_dir),
                    "prefix_shape": list(prefix_tokens[local_offset].shape),
                    "state_shape": list(batch["state"][position].shape),
                    "actions_shape": list(batch["action_chunk"][position].shape),
                    "cache_kind": PI05_WAN_CURRENT_PREFIX_CACHE_KIND,
                    "contains_future_images": False,
                    "contains_future_latents": False,
                    **_prefix_contract_metadata(args),
                    "prefix_dim": args.prefix_dim,
                    "prefix_token_count": prefix_token_count,
                    "image_key": args.image_key,
                    "source": _source_name(args),
                    "wan_action_mode": _wan_action_mode(args),
                    "prefix_compression": _prefix_compression(args),
                    **provenance_metadata,
                    **_dit_row_future_slot_conditioning_metadata(args, dataset_index=dataset_index),
                }
                _append_manifest_row(manifest_file, manifest_row)
                manifest_by_index[dataset_index] = manifest_row
                written += 1

    result = {
        "output_dir": str(output_dir),
        "num_samples": len(manifest_by_index),
        "written": written,
        "cache_kind": PI05_WAN_CURRENT_PREFIX_CACHE_KIND,
        "contains_future_images": False,
        "contains_future_latents": False,
        **_prefix_contract_metadata(args),
        "prefix_dim": args.prefix_dim,
        "prefix_token_count": prefix_token_count,
        "image_key": args.image_key,
        "source": _source_name(args),
        "wan_action_mode": _wan_action_mode(args),
        "prefix_compression": _prefix_compression(args),
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main(args: Args) -> None:
    precompute_pi05_wan_prefix_tokens(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
