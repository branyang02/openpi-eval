from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any, TextIO

import torch
import tyro
from torch.utils.data import DataLoader

from world_model.config import DatasetConfig, DatasetSource, ModelConfig
from world_model.data import (
    create_dataset,
    validate_wan_vae_latent_cache_metadata,
    wan_vae_latent_cache_metadata,
)
from world_model.train_lib import resolve_device
from world_model.wan_vae_encoder import build_frozen_wan_vae_encoder


@dataclasses.dataclass
class Args:
    dataset_source: DatasetSource = "synthetic"
    repo_id: str = "brandonyang/metaworld_ml45"
    image_keys: tuple[str, ...] = ("corner4.image",)
    output_dir: str = "output/wan_vae_latent_cache"
    episodes: tuple[int, ...] | None = None
    max_samples: int | None = None
    samples_per_episode: int | None = None
    synthetic_samples: int = 8
    image_size: int = 64
    frame_delta: int = 1
    num_future_frames: int = 4
    action_horizon: int = 8
    idm_history_length: int = 0
    batch_size: int = 4
    num_workers: int = 0
    device: str = "auto"
    seed: int = 7
    wan_vae_repo_dir: str | None = None
    wan_vae_checkpoint_path: str | None = None
    wan_vae_dtype: str = "bfloat16"
    wan_vae_tiled: bool = False
    wan_vae_latent_channels: int = 48
    wan_vae_spatial_stride: int = 16


def _relative_to_output(path: str | Path, output_dir: Path) -> str:
    return str(Path(path).expanduser().resolve().relative_to(output_dir.expanduser().resolve()))


def _append_manifest_row(manifest_file: TextIO, row: dict[str, Any]) -> None:
    manifest_file.write(json.dumps(row) + "\n")
    manifest_file.flush()
    os.fsync(manifest_file.fileno())


def _load_manifest_by_index(manifest_path: Path) -> dict[int, dict[str, Any]]:
    if not manifest_path.exists():
        return {}
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    return {int(row["dataset_index"]): row for row in rows}


def _video_from_batch(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    current_images = batch["current_images"]
    future_images = batch["future_images"]
    if current_images.shape[1] != 1 or future_images.shape[2] != 1:
        raise ValueError("Wan VAE latent caching currently supports exactly one image key/view.")
    current = current_images[:, 0].unsqueeze(1)
    future = future_images[:, :, 0]
    video = torch.cat([current, future], dim=1)
    return video.permute(0, 2, 1, 3, 4).mul(2.0).sub(1.0).contiguous()


def _expected_latent_shape(args: Args) -> tuple[int, int, int, int]:
    total_video_frames = 1 + args.num_future_frames
    latent_frames = (total_video_frames + 3) // 4
    latent_side = args.image_size // args.wan_vae_spatial_stride
    return (args.wan_vae_latent_channels, latent_frames, latent_side, latent_side)


def _build_dataset_config(args: Args) -> DatasetConfig:
    return DatasetConfig(
        source=args.dataset_source,
        repo_id=args.repo_id,
        image_keys=args.image_keys,
        image_size=args.image_size,
        frame_delta=args.frame_delta,
        num_future_frames=args.num_future_frames,
        action_horizon=args.action_horizon,
        idm_history_length=args.idm_history_length,
        max_samples=args.max_samples,
        samples_per_episode=args.samples_per_episode,
        synthetic_samples=args.synthetic_samples,
        episodes=args.episodes,
        seed=args.seed,
    )


def _build_model_config(args: Args) -> ModelConfig:
    return ModelConfig(
        num_views=len(args.image_keys),
        image_size=args.image_size,
        num_future_frames=args.num_future_frames,
        idm_history_length=args.idm_history_length,
        idm_arch="flow_transformer",
        idm_visual_encoder="wan_vae",
        wan_vae_repo_dir=args.wan_vae_repo_dir,
        wan_vae_checkpoint_path=args.wan_vae_checkpoint_path,
        wan_vae_dtype=args.wan_vae_dtype,
        wan_vae_tiled=args.wan_vae_tiled,
        wan_vae_latent_channels=args.wan_vae_latent_channels,
        wan_vae_spatial_stride=args.wan_vae_spatial_stride,
        wan_vae_use_cached_latents=True,
    )


def _validate_or_write_config(
    *,
    output_dir: Path,
    metadata: dict[str, Any],
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
) -> None:
    config_path = output_dir / "config.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(metadata, indent=2) + "\n")
        return
    existing = json.loads(config_path.read_text())
    if not isinstance(existing, dict):
        raise ValueError(f"Wan VAE latent cache config must be a JSON object: {config_path}")
    validate_wan_vae_latent_cache_metadata(
        existing,
        cache_dir=output_dir,
        dataset_config=dataset_config,
        model_config=model_config,
    )


@torch.no_grad()
def precompute_wan_vae_latents(args: Args, *, wan_encoder=None) -> dict[str, Any]:
    if len(args.image_keys) != 1:
        raise ValueError("Wan VAE latent caching currently supports exactly one image key.")
    if args.wan_vae_spatial_stride <= 0:
        raise ValueError(f"wan_vae_spatial_stride must be positive, got {args.wan_vae_spatial_stride}.")
    if args.image_size % args.wan_vae_spatial_stride != 0:
        raise ValueError("image_size must be divisible by wan_vae_spatial_stride.")
    if wan_encoder is None and (args.wan_vae_repo_dir is None or args.wan_vae_checkpoint_path is None):
        raise ValueError("--wan-vae-repo-dir and --wan-vae-checkpoint-path are required for real Wan VAE caching.")

    dataset_config = _build_dataset_config(args)
    model_config = _build_model_config(args)
    dataset = create_dataset(dataset_config)
    metadata = wan_vae_latent_cache_metadata(
        dataset_config=dataset_config,
        wan_vae_checkpoint_path=args.wan_vae_checkpoint_path,
        wan_vae_dtype=args.wan_vae_dtype,
        wan_vae_latent_channels=args.wan_vae_latent_channels,
        wan_vae_spatial_stride=args.wan_vae_spatial_stride,
        num_samples=len(dataset),
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    latents_dir = output_dir / "latents"
    output_dir.mkdir(parents=True, exist_ok=True)
    latents_dir.mkdir(parents=True, exist_ok=True)
    _validate_or_write_config(
        output_dir=output_dir,
        metadata=metadata,
        dataset_config=dataset_config,
        model_config=model_config,
    )

    manifest_path = output_dir / "manifest.jsonl"
    manifest_by_index = _load_manifest_by_index(manifest_path)
    device = resolve_device(args.device)
    encoder = wan_encoder if wan_encoder is not None else build_frozen_wan_vae_encoder(model_config)
    expected_shape = _expected_latent_shape(args)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    written = 0
    cursor = 0
    with manifest_path.open("a") as manifest_file:
        for batch in loader:
            batch_size = int(batch["current_images"].shape[0])
            indices = list(range(cursor, cursor + batch_size))
            cursor += batch_size
            uncached_positions = [position for position, index in enumerate(indices) if index not in manifest_by_index]
            if not uncached_positions:
                continue

            selected_batch = {
                key: value[uncached_positions].to(device, non_blocking=True)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }
            videos = _video_from_batch(selected_batch)
            latents = encoder.encode_videos(videos).detach().cpu().to(dtype=torch.float32)
            if tuple(latents.shape[1:]) != expected_shape:
                raise ValueError(
                    "Wan VAE encoder returned unexpected latent shape per sample: "
                    f"{tuple(latents.shape[1:])} != {expected_shape}."
                )

            for local_position, dataset_index in enumerate(indices[position] for position in uncached_positions):
                stem = f"sample_{dataset_index:06d}"
                latent_path = latents_dir / f"{stem}.pt"
                torch.save(latents[local_position], latent_path)
                row = {
                    "dataset_index": dataset_index,
                    "latent_tensor": _relative_to_output(latent_path, output_dir),
                    "latent_shape": list(latents[local_position].shape),
                }
                _append_manifest_row(manifest_file, row)
                manifest_by_index[dataset_index] = row
                written += 1

    result = {"output_dir": str(output_dir), "num_samples": len(manifest_by_index), "written": written}
    print(json.dumps(result, sort_keys=True))
    return result


def main(args: Args) -> None:
    precompute_wan_vae_latents(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
