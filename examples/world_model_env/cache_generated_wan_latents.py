from __future__ import annotations

import dataclasses
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO

import torch
import tyro
from torch.utils.data import DataLoader

from world_model.config import DatasetConfig, DatasetSource, FutureFrameStrategy, ModelConfig, Wan22Config
from world_model.data import (
    create_dataset,
    generated_wan_latent_cache_metadata,
    validate_generated_wan_latent_cache_metadata,
)
from world_model.diffsynth_wan import (
    WAN_LATENT_STAGE,
    DiffSynthWanLoraConfig,
    DiffSynthWanLoraFutureGenerator,
)


@dataclasses.dataclass
class Args:
    dataset_source: DatasetSource = "synthetic"
    repo_id: str = "brandonyang/metaworld_ml45"
    image_keys: tuple[str, ...] = ("corner4.image",)
    output_dir: str = "output/generated_wan_latent_cache"
    episodes: tuple[int, ...] | None = None
    max_samples: int | None = None
    samples_per_episode: int | None = None
    synthetic_samples: int = 8
    image_size: int = 64
    frame_delta: int = 1
    num_future_frames: int = 16
    action_horizon: int = 32
    idm_history_length: int = 0
    batch_size: int = 1
    num_workers: int = 0
    device: str = "cuda"
    seed: int = 7
    diffsynth_repo_dir: str | None = None
    checkpoint_dir: str | None = None
    lora_path: str | None = None
    height: int = 64
    width: int = 64
    num_frames: int = 17
    num_inference_steps: int = 2
    stop_after_steps: int | None = None
    lora_alpha: float = 1.0
    tiled: bool = True
    base_seed: int = 7
    prompt_template: str = Wan22Config.prompt_template
    future_frame_strategy: FutureFrameStrategy = "first"
    wan_vae_latent_channels: int = 48
    wan_vae_spatial_stride: int = 16


def _json_normalized(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_normalized(item) for item in value]
    if isinstance(value, list):
        return [_json_normalized(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_normalized(item) for key, item in value.items()}
    return value


_ROW_GENERATOR_METADATA_COMPATIBILITY_FIELDS = frozenset(
    {
        "source",
        "latent_stage",
        "height",
        "width",
        "num_frames",
        "num_inference_steps",
        "denoise_steps_run",
        "stop_after_steps",
        "denoise_fraction",
        "denoise_mode",
        "tiled",
        "checkpoint_dir",
        "lora_path",
        "lora_alpha",
        "prompt_template",
    }
)


def _merged_row_generator_metadata(
    *,
    config_generator_metadata: Mapping[str, Any],
    result_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_config = _json_normalized(dict(config_generator_metadata))
    normalized_result = _json_normalized(dict(result_metadata))
    if not isinstance(normalized_config, dict):
        raise ValueError("Generated Wan latent config generator metadata must normalize to a JSON object.")
    if not isinstance(normalized_result, dict):
        raise ValueError("DiffSynth Wan latent result metadata must normalize to a JSON object.")

    for key in sorted(_ROW_GENERATOR_METADATA_COMPATIBILITY_FIELDS):
        if key not in normalized_config or key not in normalized_result:
            continue
        config_value = normalized_config[key]
        result_value = normalized_result[key]
        if result_value != config_value:
            raise ValueError(
                "Generated Wan latent result metadata mismatch for generator_metadata "
                f"key={key!r}: result value {result_value!r} != config value {config_value!r}."
            )

    merged = dict(normalized_result)
    merged.update(normalized_config)
    return merged


def _relative_to_output(path: str | Path, output_dir: Path) -> str:
    return str(Path(path).expanduser().resolve().relative_to(output_dir.expanduser().resolve()))


def _append_manifest_row(manifest_file: TextIO, row: dict[str, Any]) -> None:
    manifest_file.write(json.dumps(row) + "\n")
    manifest_file.flush()
    os.fsync(manifest_file.fileno())


def _perf_counter() -> float:
    return time.perf_counter()


def _load_manifest_by_index(manifest_path: Path) -> dict[int, dict[str, Any]]:
    if not manifest_path.exists():
        return {}
    rows_by_index: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(manifest_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Generated Wan latent manifest row {line_number} must be a JSON object: {row!r}")
        if "dataset_index" not in row:
            raise ValueError(f"Generated Wan latent manifest row {line_number} is missing dataset_index: {row}")
        dataset_index = int(row["dataset_index"])
        if dataset_index in rows_by_index:
            raise ValueError(f"Generated Wan latent manifest has duplicate dataset_index={dataset_index}.")
        rows_by_index[dataset_index] = row
    return rows_by_index


def _seed_for_dataset_index(args: Args, dataset_index: int) -> int:
    return int(args.base_seed) + int(dataset_index)


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
        action_horizon=args.action_horizon,
        idm_history_length=args.idm_history_length,
        idm_arch="flow_transformer",
        idm_visual_encoder="wan_vae",
        wan_vae_latent_channels=args.wan_vae_latent_channels,
        wan_vae_spatial_stride=args.wan_vae_spatial_stride,
        wan_vae_use_cached_latents=True,
    )


def _generator_metadata(args: Args) -> dict[str, Any]:
    denoise_steps_run = args.num_inference_steps if args.stop_after_steps is None else args.stop_after_steps
    return {
        "source": "diffsynth_wan_lora",
        "diffsynth_repo_dir": args.diffsynth_repo_dir,
        "checkpoint_dir": args.checkpoint_dir,
        "lora_path": args.lora_path,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "denoise_steps_run": denoise_steps_run,
        "stop_after_steps": args.stop_after_steps,
        "denoise_fraction": float(denoise_steps_run) / float(args.num_inference_steps),
        "denoise_mode": "partial" if denoise_steps_run < args.num_inference_steps else "full",
        "lora_alpha": args.lora_alpha,
        "tiled": args.tiled,
        "base_seed": args.base_seed,
        "seed_strategy": "base_seed_plus_dataset_index",
        "prompt_template": args.prompt_template,
        "future_frame_strategy": args.future_frame_strategy,
        "latent_stage": WAN_LATENT_STAGE,
    }


def _validate_args(args: Args, *, require_generator_paths: bool) -> None:
    if len(args.image_keys) != 1:
        raise ValueError("Generated Wan latent caching currently supports exactly one image key/view.")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}.")
    if args.num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {args.num_workers}.")
    if args.num_future_frames <= 0:
        raise ValueError(f"num_future_frames must be positive, got {args.num_future_frames}.")
    if args.action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {args.action_horizon}.")
    if args.image_size <= 0:
        raise ValueError(f"image_size must be positive, got {args.image_size}.")
    if args.height <= 0 or args.width <= 0:
        raise ValueError(f"height and width must be positive, got {(args.height, args.width)}.")
    if args.num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {args.num_frames}.")
    if args.num_inference_steps <= 0:
        raise ValueError(f"num_inference_steps must be positive, got {args.num_inference_steps}.")
    if args.stop_after_steps is not None:
        if isinstance(args.stop_after_steps, bool) or not isinstance(args.stop_after_steps, int):
            raise ValueError(f"stop_after_steps must be None or an integer, got {args.stop_after_steps!r}.")
        if args.stop_after_steps <= 0:
            raise ValueError(f"stop_after_steps must be positive when provided, got {args.stop_after_steps}.")
        if args.stop_after_steps > args.num_inference_steps:
            raise ValueError(
                "stop_after_steps must be less than or equal to num_inference_steps "
                f"({args.num_inference_steps}), got {args.stop_after_steps}."
            )
    if args.wan_vae_spatial_stride <= 0:
        raise ValueError(f"wan_vae_spatial_stride must be positive, got {args.wan_vae_spatial_stride}.")
    if args.wan_vae_latent_channels <= 0:
        raise ValueError(f"wan_vae_latent_channels must be positive, got {args.wan_vae_latent_channels}.")
    if args.image_size % args.wan_vae_spatial_stride != 0:
        raise ValueError("image_size must be divisible by wan_vae_spatial_stride.")
    if args.height != args.image_size or args.width != args.image_size:
        raise ValueError(
            "GeneratedWanLatentDataset currently expects square Wan latents aligned to image_size; "
            f"got image_size={args.image_size}, height={args.height}, width={args.width}."
        )
    if args.num_frames != args.num_future_frames + 1:
        raise ValueError(
            "Generated Wan latent caching requires num_frames == num_future_frames + 1 so cached latents "
            f"match the GeneratedWanLatentDataset schema; got num_frames={args.num_frames}, "
            f"num_future_frames={args.num_future_frames}."
        )
    if require_generator_paths and (
        args.diffsynth_repo_dir is None or args.checkpoint_dir is None or args.lora_path is None
    ):
        raise ValueError(
            "--diffsynth-repo-dir, --checkpoint-dir, and --lora-path are required for real DiffSynth caching."
        )


def _expected_latent_shape(args: Args) -> tuple[int, int, int, int]:
    total_video_frames = 1 + args.num_future_frames
    latent_frames = (total_video_frames + 3) // 4
    latent_side = args.image_size // args.wan_vae_spatial_stride
    return (args.wan_vae_latent_channels, latent_frames, latent_side, latent_side)


def _validate_or_write_config(
    *,
    output_dir: Path,
    metadata: dict[str, Any],
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
    generator_metadata: Mapping[str, Any],
) -> None:
    config_path = output_dir / "config.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(metadata, indent=2) + "\n")
        return

    existing = json.loads(config_path.read_text())
    if not isinstance(existing, dict):
        raise ValueError(f"Generated Wan latent cache config must be a JSON object: {config_path}")
    validate_generated_wan_latent_cache_metadata(
        existing,
        cache_dir=output_dir,
        dataset_config=dataset_config,
        model_config=model_config,
        generator_metadata=generator_metadata,
    )
    expected_dataset_config = _json_normalized(dataclasses.asdict(dataset_config))
    cached_dataset_config = _json_normalized(existing.get("dataset_config"))
    if (
        dataset_config.idm_history_length == 0
        and isinstance(cached_dataset_config, dict)
        and cached_dataset_config.get("idm_history_length") is None
    ):
        cached_dataset_config = {**cached_dataset_config, "idm_history_length": 0}
    if cached_dataset_config != expected_dataset_config:
        raise ValueError(
            "Generated Wan latent cache metadata mismatch for dataset_config in "
            f"{output_dir}: cached dataset_config={existing.get('dataset_config')!r}, "
            f"requested dataset_config={expected_dataset_config!r}."
        )
    if int(existing.get("num_samples", -1)) != int(metadata["num_samples"]):
        raise ValueError(
            "Generated Wan latent cache metadata mismatch for num_samples in "
            f"{output_dir}: cached num_samples={existing.get('num_samples')!r}, "
            f"requested num_samples={metadata['num_samples']!r}."
        )


def _build_generator(args: Args) -> DiffSynthWanLoraFutureGenerator:
    return DiffSynthWanLoraFutureGenerator(
        DiffSynthWanLoraConfig(
            diffsynth_repo_dir=str(args.diffsynth_repo_dir),
            checkpoint_dir=str(args.checkpoint_dir),
            lora_path=str(args.lora_path),
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            lora_alpha=args.lora_alpha,
            device=args.device,
            tiled=args.tiled,
            base_seed=args.base_seed,
            prompt_template=args.prompt_template,
            future_frame_strategy=args.future_frame_strategy,
        )
    )


def _task_text_for_index(dataset: Any, index: int, item: Mapping[str, Any]) -> str:
    if hasattr(dataset, "task_text"):
        return str(dataset.task_text(index))
    if "task_id" not in item:
        raise ValueError("Cannot build DiffSynth prompt because dataset has no task_text method or task_id field.")
    return f"metaworld task id {int(item['task_id'])}"


def _latent_without_batch(latents: Any, *, expected_shape: tuple[int, int, int, int]) -> torch.Tensor:
    if not isinstance(latents, torch.Tensor):
        raise ValueError(f"DiffSynth Wan latent result must be a torch.Tensor, got {type(latents).__name__}.")
    shape = tuple(int(dim) for dim in latents.shape)
    if latents.ndim != 5:
        raise ValueError(
            "Expected generated Wan latents with rank 5 shaped (B, C, T, H, W), "
            f"got rank {latents.ndim} and shape {shape}."
        )
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"Generated Wan latents must have only positive dimensions, got {shape}.")
    if latents.shape[0] != 1:
        raise ValueError(f"Expected generated Wan latent batch dimension 1, got shape {shape}.")
    sample_shape = tuple(int(dim) for dim in latents.shape[1:])
    if sample_shape != expected_shape:
        raise ValueError(
            "Generated Wan latent shape does not match GeneratedWanLatentDataset schema: "
            f"{sample_shape} != {expected_shape}."
        )
    return latents[0].detach().cpu().contiguous()


@torch.no_grad()
def precompute_generated_wan_latents(args: Args, *, generator: Any | None = None) -> dict[str, Any]:
    elapsed_start = _perf_counter()
    _validate_args(args, require_generator_paths=generator is None)
    torch.manual_seed(args.seed)

    dataset_config = _build_dataset_config(args)
    model_config = _build_model_config(args)
    dataset = create_dataset(dataset_config)
    config_generator_metadata = _generator_metadata(args)
    metadata = generated_wan_latent_cache_metadata(
        dataset_config=dataset_config,
        wan_vae_latent_channels=args.wan_vae_latent_channels,
        wan_vae_spatial_stride=args.wan_vae_spatial_stride,
        generator_metadata=config_generator_metadata,
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
        generator_metadata=config_generator_metadata,
    )

    manifest_path = output_dir / "manifest.jsonl"
    manifest_by_index = _load_manifest_by_index(manifest_path)
    generator_load_wall_seconds = 0.0
    if generator is None:
        wan_generator = _build_generator(args)
        generator_load_start = _perf_counter()
        _ = wan_generator.pipe
        generator_load_wall_seconds = _perf_counter() - generator_load_start
    else:
        wan_generator = generator
    expected_shape = _expected_latent_shape(args)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    written = 0
    generation_wall_seconds = 0.0
    write_wall_seconds = 0.0
    cursor = 0
    with manifest_path.open("a") as manifest_file:
        for batch in loader:
            current_images = batch["current_images"]
            if current_images.ndim != 5:
                raise ValueError(
                    "Expected batch current_images with shape (B, V, 3, H, W), " f"got {tuple(current_images.shape)}."
                )
            if current_images.shape[1] != 1:
                raise ValueError("Generated Wan latent caching currently supports exactly one image key/view.")

            batch_size = int(current_images.shape[0])
            indices = list(range(cursor, cursor + batch_size))
            cursor += batch_size
            for local_position, dataset_index in enumerate(indices):
                if dataset_index in manifest_by_index:
                    continue

                item = {key: value[local_position] for key, value in batch.items() if isinstance(value, torch.Tensor)}
                task_text = _task_text_for_index(dataset, dataset_index, item)
                seed = _seed_for_dataset_index(args, dataset_index)
                generation_start = _perf_counter()
                result = wan_generator.generate_view_latents(
                    current_images[local_position, 0],
                    task_text=task_text,
                    seed=seed,
                    stop_after_steps=args.stop_after_steps,
                )
                row_generation_wall_seconds = _perf_counter() - generation_start
                generation_wall_seconds += row_generation_wall_seconds
                latent = _latent_without_batch(result.latents, expected_shape=expected_shape)
                result_metadata = getattr(result, "metadata", None)
                if not isinstance(result_metadata, Mapping):
                    raise ValueError("DiffSynth Wan latent result metadata must be a mapping.")
                row_generator_metadata = _merged_row_generator_metadata(
                    config_generator_metadata=config_generator_metadata,
                    result_metadata=result_metadata,
                )

                stem = f"sample_{dataset_index:06d}"
                latent_path = latents_dir / f"{stem}.pt"
                write_start = _perf_counter()
                torch.save(latent, latent_path)
                row = {
                    "dataset_index": dataset_index,
                    "latent_tensor": _relative_to_output(latent_path, output_dir),
                    "latent_shape": list(latent.shape),
                    "seed": int(getattr(result, "seed", seed)),
                    "prompt": str(getattr(result, "prompt")),
                    "generator_metadata": row_generator_metadata,
                    "generation_wall_seconds": row_generation_wall_seconds,
                }
                _append_manifest_row(manifest_file, row)
                write_wall_seconds += _perf_counter() - write_start
                manifest_by_index[dataset_index] = row
                written += 1

    elapsed_wall_seconds = _perf_counter() - elapsed_start
    result = {
        "output_dir": str(output_dir),
        "num_samples": len(manifest_by_index),
        "written": written,
        "elapsed_wall_seconds": elapsed_wall_seconds,
        "generator_load_wall_seconds": generator_load_wall_seconds,
        "generation_wall_seconds": generation_wall_seconds,
        "generation_wall_seconds_mean": generation_wall_seconds / written if written else 0.0,
        "write_wall_seconds": write_wall_seconds,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main(args: Args) -> None:
    precompute_generated_wan_latents(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
