from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any, Literal, TextIO

import torch
import tyro

from world_model.config import (
    DatasetConfig,
    DatasetSource,
    FutureFrameStrategy,
    Wan22Config,
    validate_future_frame_strategy,
)
from world_model.data import (
    create_dataset,
    expected_wan_source_frame_offsets,
    validate_raw_wan_frame_delta,
    validate_wan_selected_frame_indices,
)
from world_model.diffsynth_wan import DiffSynthWanLoraConfig, DiffSynthWanLoraFutureGenerator
from world_model.media import save_png, save_video
from world_model.wan22 import DEFAULT_CONDITIONING_FRAME_MAX_MAE, Wan22FutureGenerator

FutureSource = Literal["dataset_future", "wan2_2", "wan_lora"]


@dataclasses.dataclass
class Args:
    future_source: FutureSource = "dataset_future"
    dataset_source: DatasetSource = "synthetic"
    repo_id: str = "brandonyang/metaworld_ml45"
    image_key: str = "corner4.image"
    output_dir: str = "output/future_cache"
    episodes: tuple[int, ...] | None = None
    max_samples: int | None = None
    samples_per_episode: int | None = None
    synthetic_samples: int = 8
    image_size: int = 64
    frame_delta: int = 1
    num_future_frames: int = 4
    action_horizon: int = 8
    fps: int = 12
    seed: int = 7
    generation_seed: int | None = None
    wan_repo_dir: str | None = None
    wan_checkpoint_dir: str | None = None
    wan_task: str = "ti2v-5B"
    wan_size: str = "1280*704"
    wan_frame_num: int = 17
    wan_sample_steps: int | None = None
    wan_sample_shift: float | None = None
    wan_sample_guide_scale: float | None = None
    wan_offload_model: bool = False
    wan_convert_model_dtype: bool = False
    wan_t5_cpu: bool = False
    wan_python_executable: str = "python"
    wan_future_frame_strategy: FutureFrameStrategy = "first"
    diffsynth_repo_dir: str | None = None
    wan_lora_checkpoint_dir: str | None = None
    wan_lora_path: str | None = None
    wan_lora_height: int | None = None
    wan_lora_width: int | None = None
    wan_lora_num_frames: int = 17
    wan_lora_num_inference_steps: int = 2
    wan_lora_alpha: float = 1.0
    wan_lora_device: str = "cuda"
    wan_lora_tiled: bool = True
    wan_lora_future_frame_strategy: FutureFrameStrategy = "first"
    # Shared across wan2_2 and wan_lora: enforce that generated frame 0 is the conditioning image.
    wan_verify_conditioning_frame: bool = True
    wan_conditioning_frame_max_mae: float = DEFAULT_CONDITIONING_FRAME_MAX_MAE
    prompt_template: str = Wan22Config.prompt_template


def generation_seed_base(args: Args) -> int:
    """Seed used only for stochastic future generation.

    Defaults to the dataset seed for backward compatibility, but can be varied
    independently when comparing multiple Wan samples for the same dataset rows.
    """

    return args.seed if args.generation_seed is None else args.generation_seed


def task_text_for_index(dataset, index: int, fallback_task_id: int) -> str:
    if hasattr(dataset, "task_text"):
        return dataset.task_text(index)
    return f"metaworld task id {fallback_task_id}"


def build_wan_generator(args: Args) -> Wan22FutureGenerator:
    validate_raw_wan_frame_delta(args.frame_delta, context="future_source='wan2_2'")
    if args.wan_repo_dir is None:
        raise ValueError("--wan-repo-dir is required when --future-source wan2_2.")
    if args.wan_checkpoint_dir is None:
        raise ValueError("--wan-checkpoint-dir is required when --future-source wan2_2.")
    return Wan22FutureGenerator(
        Wan22Config(
            repo_dir=args.wan_repo_dir,
            checkpoint_dir=args.wan_checkpoint_dir,
            task=args.wan_task,
            size=args.wan_size,
            frame_num=args.wan_frame_num,
            sample_steps=args.wan_sample_steps,
            sample_shift=args.wan_sample_shift,
            sample_guide_scale=args.wan_sample_guide_scale,
            offload_model=args.wan_offload_model,
            convert_model_dtype=args.wan_convert_model_dtype,
            t5_cpu=args.wan_t5_cpu,
            base_seed=generation_seed_base(args),
            python_executable=args.wan_python_executable,
            frame_delta=args.frame_delta,
            future_frame_strategy=args.wan_future_frame_strategy,
            prompt_template=args.prompt_template,
        ),
        verify_conditioning_frame=args.wan_verify_conditioning_frame,
        conditioning_frame_max_mae=args.wan_conditioning_frame_max_mae,
    )


def build_wan_lora_generator(args: Args) -> DiffSynthWanLoraFutureGenerator:
    if args.diffsynth_repo_dir is None:
        raise ValueError("--diffsynth-repo-dir is required when --future-source wan_lora.")
    if args.wan_lora_checkpoint_dir is None:
        raise ValueError("--wan-lora-checkpoint-dir is required when --future-source wan_lora.")
    if args.wan_lora_path is None:
        raise ValueError("--wan-lora-path is required when --future-source wan_lora.")
    return DiffSynthWanLoraFutureGenerator(
        DiffSynthWanLoraConfig(
            diffsynth_repo_dir=args.diffsynth_repo_dir,
            checkpoint_dir=args.wan_lora_checkpoint_dir,
            lora_path=args.wan_lora_path,
            height=args.wan_lora_height or args.image_size,
            width=args.wan_lora_width or args.image_size,
            num_frames=args.wan_lora_num_frames,
            num_inference_steps=args.wan_lora_num_inference_steps,
            lora_alpha=args.wan_lora_alpha,
            device=args.wan_lora_device,
            tiled=args.wan_lora_tiled,
            fps=args.fps,
            base_seed=generation_seed_base(args),
            prompt_template=args.prompt_template,
            frame_delta=args.frame_delta,
            future_frame_strategy=args.wan_lora_future_frame_strategy,
            verify_conditioning_frame=args.wan_verify_conditioning_frame,
            conditioning_frame_max_mae=args.wan_conditioning_frame_max_mae,
        )
    )


def relative_to_output(path: str | Path, output_dir: Path) -> str:
    return str(Path(path).expanduser().resolve().relative_to(output_dir.expanduser().resolve()))


def load_manifest_by_index(manifest_path: Path) -> dict[int, dict[str, Any]]:
    if not manifest_path.exists():
        return {}
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    return {int(row["dataset_index"]): row for row in rows}


def _json_normalized(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_normalized(item) for item in value]
    if isinstance(value, list):
        return [_json_normalized(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_normalized(item) for key, item in value.items()}
    return value


def validate_existing_cache_config(
    *,
    args: Args,
    output_dir: Path,
    dataset_config: DatasetConfig,
) -> None:
    """Reject resuming an output directory for any other dataset split."""

    config_path = output_dir / "config.json"
    manifest_path = output_dir / "manifest.jsonl"
    if not config_path.exists() and not manifest_path.exists():
        return
    if not config_path.exists():
        raise ValueError(
            f"Cannot resume cache in {output_dir} because manifest.jsonl exists but config.json is missing. "
            "Use a fresh output_dir or regenerate the cache."
        )

    full_config = json.loads(config_path.read_text())
    if not isinstance(full_config, dict):
        raise ValueError(f"Cached future config must be a JSON object: {config_path}")
    if full_config.get("future_source") != args.future_source:
        raise ValueError(
            "Cannot resume cache with a different future_source in the same output_dir: "
            f"existing future_source={full_config.get('future_source')!r}, "
            f"requested future_source={args.future_source!r}. Use a fresh output_dir."
        )
    existing_dataset_config = full_config.get("dataset_config")
    if not isinstance(existing_dataset_config, dict):
        raise ValueError(
            "Cannot resume cache because existing config.json is missing dataset_config. "
            "Use a fresh output_dir or regenerate the cache."
        )

    requested = dataclasses.asdict(dataset_config)
    existing_normalized = _json_normalized(existing_dataset_config)
    requested_normalized = _json_normalized(requested)
    if existing_normalized == requested_normalized:
        return

    missing = object()
    mismatched_keys = [
        key
        for key in sorted(set(existing_normalized) | set(requested_normalized))
        if existing_normalized.get(key, missing) != requested_normalized.get(key, missing)
    ]
    details = {
        key: {
            "existing": existing_dataset_config.get(key),
            "requested": requested.get(key),
        }
        for key in mismatched_keys
    }
    mismatch_summary = (
        f"different {mismatched_keys[0]}" if len(mismatched_keys) == 1 else f"different fields {mismatched_keys}"
    )
    raise ValueError(
        "Cannot resume cache because existing dataset_config does not exactly match the requested "
        f"DatasetConfig ({mismatch_summary}). Mismatched keys: {mismatched_keys}. Details: {details}. "
        "Use a fresh output_dir for a different dataset split."
    )


def validate_existing_generation_seed(
    *,
    args: Args,
    output_dir: Path,
    manifest_by_index: dict[int, dict[str, Any]],
) -> None:
    if not is_wan_future_source(args.future_source) or not manifest_by_index:
        return

    requested_base = generation_seed_base(args)
    requested_strategy = requested_future_frame_strategy(args)
    existing_config_strategy = requested_strategy
    config_path = output_dir / "config.json"
    if config_path.exists():
        full_config = json.loads(config_path.read_text())
        if not isinstance(full_config, dict):
            raise ValueError(f"Cached future config must be a JSON object: {config_path}")
        existing_base = full_config.get("generation_seed")
        if existing_base is not None and int(existing_base) != requested_base:
            raise ValueError(
                "Cannot resume Wan cache with a different generation_seed in the same output_dir: "
                f"existing generation_seed={existing_base}, requested generation_seed={requested_base}. "
                "Use a fresh output_dir for each stochastic Wan sample."
            )
        dataset_config = full_config.get("dataset_config")
        if not isinstance(dataset_config, dict):
            raise ValueError(
                "Cannot resume Wan cache because existing config.json is missing dataset_config. "
                "Use a fresh output_dir or regenerate the cache."
            )
        for key, requested in (
            ("frame_delta", args.frame_delta),
            ("num_future_frames", args.num_future_frames),
        ):
            existing = dataset_config.get(key)
            if existing is None:
                raise ValueError(
                    f"Cannot resume Wan cache because existing config.json is missing dataset_config.{key}. "
                    "Use a fresh output_dir or regenerate the cache."
                )
            if int(existing) != int(requested):
                raise ValueError(
                    f"Cannot resume Wan cache with a different {key} in the same output_dir: "
                    f"existing {key}={existing}, requested {key}={requested}. "
                    "Use a fresh output_dir for a different temporal contract."
                )
        temporal_config = full_config.get("future_frame_selection")
        if isinstance(temporal_config, dict) and temporal_config.get("source_frame_offsets") is not None:
            existing_strategy_value = temporal_config.get("future_frame_strategy", "first")
            if existing_strategy_value is None:
                raise ValueError(
                    "Cannot resume Wan cache because existing future_frame_selection.future_frame_strategy "
                    "is mixed or missing. Use a fresh output_dir or regenerate the cache."
                )
            existing_config_strategy = validate_future_frame_strategy(str(existing_strategy_value))
            if requested_strategy is not None and existing_config_strategy != requested_strategy:
                raise ValueError(
                    "Cannot resume Wan cache with a different future_frame_strategy in the same output_dir: "
                    f"existing future_frame_strategy={existing_config_strategy!r}, "
                    f"requested future_frame_strategy={requested_strategy!r}. "
                    "Use a fresh output_dir for a different temporal selection strategy."
                )
            expected_offsets = expected_wan_source_frame_offsets(args.frame_delta, args.num_future_frames)
            try:
                existing_offsets = [int(offset) for offset in temporal_config["source_frame_offsets"]]
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Cannot resume Wan cache because existing future_frame_selection.source_frame_offsets "
                    f"is invalid: {temporal_config['source_frame_offsets']!r}."
                ) from error
            if existing_offsets != expected_offsets:
                raise ValueError(
                    "Cannot resume Wan cache because existing source_frame_offsets do not match the requested "
                    f"temporal contract: existing {existing_offsets}, requested {expected_offsets}. "
                    "Use a fresh output_dir for a different temporal contract."
                )
            if temporal_config.get("selected_frame_indices") is not None and requested_strategy is not None:
                validate_wan_selected_frame_indices(
                    temporal_config["selected_frame_indices"],
                    frame_delta=args.frame_delta,
                    num_future_frames=args.num_future_frames,
                    strategy=requested_strategy,
                    context="Existing Wan cache config future_frame_selection",
                )

    bad_rows: list[tuple[int, Any, int]] = []
    bad_strategy_rows: list[tuple[int, Any, str]] = []
    for dataset_index, row in sorted(manifest_by_index.items()):
        row_seed = row.get("generation_seed")
        expected_seed = requested_base + dataset_index
        if row_seed != expected_seed:
            bad_rows.append((dataset_index, row_seed, expected_seed))
        if requested_strategy is not None:
            row_strategy = row.get("future_frame_strategy", existing_config_strategy)
            if row_strategy is not None:
                normalized_row_strategy = validate_future_frame_strategy(str(row_strategy))
                if normalized_row_strategy != requested_strategy:
                    bad_strategy_rows.append((dataset_index, row_strategy, requested_strategy))
            if row.get("selected_frame_indices") is not None:
                validate_wan_selected_frame_indices(
                    row["selected_frame_indices"],
                    frame_delta=args.frame_delta,
                    num_future_frames=args.num_future_frames,
                    strategy=requested_strategy,
                    context=f"Existing Wan cache manifest row dataset_index={dataset_index}",
                )

    if bad_rows:
        preview = bad_rows[:5]
        raise ValueError(
            "Cannot resume Wan cache because manifest generation_seed values do not match the requested "
            f"generation_seed base {requested_base}. Mismatched rows: {preview}. "
            "Use a fresh output_dir for each stochastic Wan sample."
        )
    if bad_strategy_rows:
        preview = bad_strategy_rows[:5]
        raise ValueError(
            "Cannot resume Wan cache because manifest future_frame_strategy values do not match the requested "
            f"future_frame_strategy={requested_strategy!r}. Mismatched rows: {preview}. "
            "Use a fresh output_dir for a different temporal selection strategy."
        )


def append_manifest_row(manifest_file: TextIO, row: dict[str, Any]) -> None:
    manifest_file.write(json.dumps(row) + "\n")
    manifest_file.flush()
    os.fsync(manifest_file.fileno())


def is_wan_future_source(source: FutureSource) -> bool:
    return source in {"wan2_2", "wan_lora"}


def requested_future_frame_strategy(args: Args) -> FutureFrameStrategy | None:
    if args.future_source == "wan2_2":
        return args.wan_future_frame_strategy
    if args.future_source == "wan_lora":
        return args.wan_lora_future_frame_strategy
    return None


def build_manifest_row(
    *,
    args: Args,
    index: int,
    prompt: str,
    current_path: Path,
    future_tensor_path: Path,
    video_path: Path,
    output_dir: Path,
    future_shape: torch.Size | tuple[int, ...],
    generation_seed: int | None,
    future_frame_strategy: FutureFrameStrategy | None,
    selected_frame_indices: list[int] | None,
    total_video_frames: int | None,
) -> dict[str, Any]:
    row = {
        "source": args.future_source,
        "dataset_index": index,
        "prompt": prompt,
        "current_image": relative_to_output(current_path, output_dir),
        "future_tensor": relative_to_output(future_tensor_path, output_dir),
        "video": relative_to_output(video_path, output_dir),
        "future_shape": list(future_shape),
        "generation_seed": generation_seed,
        "future_frame_strategy": future_frame_strategy,
        "selected_frame_indices": selected_frame_indices,
        "total_video_frames": total_video_frames,
    }
    if is_wan_future_source(args.future_source):
        row["dataset_frame_delta"] = args.frame_delta
        row["source_frame_offsets"] = expected_wan_source_frame_offsets(
            args.frame_delta,
            args.num_future_frames,
        )
    return row


def build_future_frame_selection_config(
    *,
    args: Args,
    manifest_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not is_wan_future_source(args.future_source):
        return None

    missing_indices = [int(row["dataset_index"]) for row in manifest_rows if row.get("selected_frame_indices") is None]
    if missing_indices:
        raise ValueError(
            "Wan cache temporal metadata is incomplete for dataset_index values "
            f"{missing_indices}. Cannot record selected generated-video frame indices for "
            "existing cached tensors; regenerate those futures with cache_future_rollouts.py."
        )

    selected_by_index = {str(int(row["dataset_index"])): list(row["selected_frame_indices"]) for row in manifest_rows}
    unique_selected = {tuple(indices) for indices in selected_by_index.values()}
    unique_strategies = {
        row.get("future_frame_strategy") for row in manifest_rows if row.get("future_frame_strategy") is not None
    }
    unique_total_frames = {
        int(row["total_video_frames"]) for row in manifest_rows if row.get("total_video_frames") is not None
    }
    source_frame_offsets = expected_wan_source_frame_offsets(args.frame_delta, args.num_future_frames)
    return {
        "future_frame_strategy": next(iter(unique_strategies)) if len(unique_strategies) == 1 else None,
        "selected_frame_indices": list(next(iter(unique_selected))) if len(unique_selected) == 1 else None,
        "selected_frame_indices_by_dataset_index": selected_by_index,
        "total_video_frames": next(iter(unique_total_frames)) if len(unique_total_frames) == 1 else None,
        "dataset_frame_delta": args.frame_delta,
        "frame_delta": args.frame_delta,
        "source_frame_offsets": source_frame_offsets,
        "num_future_frames": args.num_future_frames,
    }


def default_video_path(args: Args, output_dir: Path, videos_dir: Path, stem: str) -> Path:
    if args.future_source == "wan2_2":
        return output_dir / "wan_raw" / stem / "wan22_view0.mp4"
    if args.future_source == "wan_lora":
        return output_dir / "wan_lora_raw" / stem / "wan_lora_view0.mp4"
    return videos_dir / f"{stem}.mp4"


def main(args: Args) -> None:
    dataset_config = DatasetConfig(
        source=args.dataset_source,
        repo_id=args.repo_id,
        image_keys=(args.image_key,),
        image_size=args.image_size,
        frame_delta=args.frame_delta,
        num_future_frames=args.num_future_frames,
        action_horizon=args.action_horizon,
        max_samples=args.max_samples,
        samples_per_episode=args.samples_per_episode,
        synthetic_samples=args.synthetic_samples,
        episodes=args.episodes,
        seed=args.seed,
    )
    output_dir = Path(args.output_dir)
    validate_existing_cache_config(args=args, output_dir=output_dir, dataset_config=dataset_config)
    manifest_path = output_dir / "manifest.jsonl"
    manifest_by_index = load_manifest_by_index(manifest_path)
    validate_existing_generation_seed(args=args, output_dir=output_dir, manifest_by_index=manifest_by_index)
    dataset = create_dataset(dataset_config)
    futures_dir = output_dir / "futures"
    videos_dir = output_dir / "videos"
    images_dir = output_dir / "images"
    futures_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    wan_generator = build_wan_generator(args) if args.future_source == "wan2_2" else None
    wan_lora_generator = build_wan_lora_generator(args) if args.future_source == "wan_lora" else None
    with manifest_path.open("a") as manifest_file:
        for index in range(len(dataset)):
            stem = f"sample_{index:06d}"
            current_path = images_dir / f"{stem}.png"
            future_tensor_path = futures_dir / f"{stem}.pt"
            video_path = default_video_path(args, output_dir, videos_dir, stem)

            if future_tensor_path.exists() and index in manifest_by_index:
                continue

            item = dataset[index]
            task_text = task_text_for_index(dataset, index, int(item["task_id"]))
            prompt = args.prompt_template.format(task=task_text)
            generation_seed = None
            selected_frame_indices = None
            total_video_frames = None
            future_frame_strategy = None

            if future_tensor_path.exists():
                future_images = torch.load(future_tensor_path, map_location="cpu", weights_only=True)
                if not isinstance(future_images, torch.Tensor):
                    raise TypeError(f"Cached future tensor must be a torch.Tensor, got {type(future_images)}.")
                if not current_path.exists():
                    save_png(item["current_images"][0], current_path)
                if args.future_source == "dataset_future" and not video_path.exists():
                    clip = torch.cat([item["current_images"][0].unsqueeze(0), future_images[:, 0]], dim=0)
                    save_video(clip, video_path, args.fps)
                if args.future_source == "wan2_2":
                    generation_seed = generation_seed_base(args) + index
                    future_frame_strategy = args.wan_future_frame_strategy
                elif args.future_source == "wan_lora":
                    generation_seed = generation_seed_base(args) + index
                    future_frame_strategy = args.wan_lora_future_frame_strategy
            else:
                save_png(item["current_images"][0], current_path)
                if args.future_source == "dataset_future":
                    future_images = item["future_images"]
                    clip = torch.cat([item["current_images"][0].unsqueeze(0), future_images[:, 0]], dim=0)
                    save_video(clip, video_path, args.fps)
                elif args.future_source == "wan2_2":
                    assert wan_generator is not None
                    generation_seed = generation_seed_base(args) + index
                    future_frame_strategy = args.wan_future_frame_strategy
                    result = wan_generator.generate_future_stack(
                        item["current_images"],
                        task_text=task_text,
                        output_dir=output_dir / "wan_raw" / stem,
                        image_size=args.image_size,
                        num_future_frames=args.num_future_frames,
                        seed=generation_seed,
                    )
                    future_images = result.future_images
                    video_path = result.video_path
                    selected_frame_indices = list(result.selected_frame_indices)
                    total_video_frames = result.total_video_frames
                elif args.future_source == "wan_lora":
                    assert wan_lora_generator is not None
                    generation_seed = generation_seed_base(args) + index
                    future_frame_strategy = args.wan_lora_future_frame_strategy
                    result = wan_lora_generator.generate_future_stack(
                        item["current_images"],
                        task_text=task_text,
                        output_dir=output_dir / "wan_lora_raw" / stem,
                        image_size=args.image_size,
                        num_future_frames=args.num_future_frames,
                        seed=generation_seed,
                    )
                    future_images = result.future_images
                    video_path = result.video_path
                    selected_frame_indices = list(result.selected_frame_indices)
                    total_video_frames = result.total_video_frames
                else:
                    raise ValueError(f"Unknown future_source: {args.future_source}")

                torch.save(future_images.detach().cpu(), future_tensor_path)

            if tuple(future_images.shape) != tuple(item["future_images"].shape):
                raise ValueError(
                    "Generated future shape does not match dataset future shape: "
                    f"{tuple(future_images.shape)} != {tuple(item['future_images'].shape)}"
                )
            if is_wan_future_source(args.future_source) and selected_frame_indices is None:
                raise ValueError(
                    "Cannot create a Wan cache manifest row without selected generated-video frame indices "
                    f"for dataset_index={index}. The future tensor already exists but its temporal metadata "
                    "is unavailable; regenerate this sample instead of inferring a fallback."
                )
            if is_wan_future_source(args.future_source):
                validate_wan_selected_frame_indices(
                    selected_frame_indices,
                    frame_delta=args.frame_delta,
                    num_future_frames=args.num_future_frames,
                    strategy=future_frame_strategy or "first",
                    context=f"Generated {args.future_source} cache row dataset_index={index}",
                )
            if index not in manifest_by_index:
                row = build_manifest_row(
                    args=args,
                    index=index,
                    prompt=prompt,
                    current_path=current_path,
                    future_tensor_path=future_tensor_path,
                    video_path=video_path,
                    output_dir=output_dir,
                    future_shape=future_images.shape,
                    generation_seed=generation_seed,
                    future_frame_strategy=future_frame_strategy,
                    selected_frame_indices=selected_frame_indices,
                    total_video_frames=total_video_frames,
                )
                append_manifest_row(manifest_file, row)
                manifest_by_index[index] = row

    manifest_rows = sorted(manifest_by_index.values(), key=lambda row: int(row["dataset_index"]))
    future_frame_selection = build_future_frame_selection_config(args=args, manifest_rows=manifest_rows)
    config: dict[str, Any] = {
        "future_source": args.future_source,
        "dataset_config": dataclasses.asdict(dataset_config),
        "num_samples": len(manifest_by_index),
    }
    if future_frame_selection is not None:
        config["future_frame_selection"] = future_frame_selection
        config["generation_seed"] = generation_seed_base(args)
    (output_dir / "config.json").write_text(
        json.dumps(
            config,
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"output_dir": str(output_dir), "num_samples": len(manifest_by_index)}, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))
