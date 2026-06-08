from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Literal

import torch
import tyro
from torch.utils.data import DataLoader

from world_model.config import DatasetConfig, DatasetSource, ModelConfig, Wan22Config
from world_model.data import CachedFutureDataset, create_dataset, infer_batch_spec
from world_model.media import save_png, save_video
from world_model.models import InverseDynamicsModel
from world_model.train_lib import (
    enforce_idm_frame_delta_contract,
    evaluate_idm,
    idm_uses_flow_matching,
    load_idm_checkpoint,
    resolve_device,
    seed_everything,
)

FutureSource = Literal["dataset_future", "cached"]


@dataclasses.dataclass
class Args:
    dataset_source: DatasetSource = "synthetic"
    repo_id: str = "brandonyang/metaworld_ml45"
    image_key: str = "corner4.image"
    output_dir: str = "output/pipeline_eval"
    idm_checkpoint: str | None = None
    cached_future_dir: str | None = None
    episodes: tuple[int, ...] | None = None
    max_samples: int | None = 64
    synthetic_samples: int = 64
    image_size: int = 64
    frame_delta: int = 1
    num_future_frames: int = 4
    action_horizon: int = 8
    batch_size: int = 16
    fps: int = 12
    future_source: FutureSource = "dataset_future"
    device: str = "auto"
    flow_eval_seed: int | None = 0
    seed: int = 7
    prompt_template: str = Wan22Config.prompt_template


def task_text_for_index(dataset, index: int, fallback_task_id: int) -> str:
    if hasattr(dataset, "task_text"):
        return dataset.task_text(index)
    return f"metaworld task id {fallback_task_id}"


def model_config_from_dataset(dataset, dataset_config: DatasetConfig) -> ModelConfig:
    spec = infer_batch_spec(dataset, task_vocab_size=dataset_config.task_vocab_size)
    return ModelConfig(
        num_views=spec.num_views,
        image_size=dataset_config.image_size,
        state_dim=spec.state_dim,
        action_dim=spec.action_dim,
        action_horizon=spec.action_horizon,
        num_future_frames=spec.num_future_frames,
        task_vocab_size=spec.task_vocab_size,
    )


def resolve_dataset_dims(args: Args, checkpoint_model_config: ModelConfig | None) -> tuple[int, int, int, int]:
    """Return (image_size, num_future_frames, action_horizon) for dataset construction.

    When an IDM checkpoint is provided, the dataset must match the dimensions the model was
    trained with, so the checkpoint's model_config takes precedence over the CLI defaults.
    """
    if checkpoint_model_config is None:
        return args.image_size, args.num_future_frames, args.action_horizon, 0
    return (
        checkpoint_model_config.image_size,
        checkpoint_model_config.num_future_frames,
        checkpoint_model_config.action_horizon,
        checkpoint_model_config.idm_history_length,
    )


def resolve_idm(
    *,
    checkpoint: tuple[InverseDynamicsModel, ModelConfig] | None,
    dataset,
    dataset_config: DatasetConfig,
    device: torch.device,
) -> tuple[InverseDynamicsModel, ModelConfig, str]:
    if checkpoint is not None:
        idm, model_config = checkpoint
        return idm, model_config, "checkpoint"
    model_config = model_config_from_dataset(dataset, dataset_config)
    return InverseDynamicsModel(model_config).to(device), model_config, "untrained"


def write_visual_debug_sample(
    dataset,
    output_dir: Path,
    fps: int,
    prompt_template: str,
    future_source: FutureSource,
) -> dict[str, object]:
    item = dataset[0]
    task_text = task_text_for_index(dataset, 0, int(item["task_id"]))
    prompt = prompt_template.format(task=task_text)
    current_path = output_dir / "current_frame.png"
    future_path = output_dir / f"{future_source}_future_debug.mp4"
    save_png(item["current_images"][0], current_path)
    clip = torch.cat([item["current_images"][0].unsqueeze(0), item["future_images"][:, 0]], dim=0)
    save_video(clip, future_path, fps)
    return {
        "prompt": prompt,
        "current_frame": str(current_path),
        "future_debug_video": str(future_path),
        "clip_num_frames": int(clip.shape[0]),
        "current_images_shape": list(item["current_images"].shape),
        "future_images_shape": list(item["future_images"].shape),
    }


def main(args: Args) -> None:
    if args.future_source not in ("dataset_future", "cached"):
        raise ValueError(f"Unknown future_source: {args.future_source}")

    seed_everything(args.seed)
    device = resolve_device(args.device)

    # Load the IDM checkpoint (if any) before building the dataset so its trained
    # dimensions drive dataset construction instead of stale CLI defaults.
    checkpoint: tuple[InverseDynamicsModel, ModelConfig] | None = None
    if args.idm_checkpoint is not None:
        checkpoint = load_idm_checkpoint(args.idm_checkpoint, device)
        enforce_idm_frame_delta_contract(args.idm_checkpoint, args.frame_delta)

    image_size, num_future_frames, action_horizon, idm_history_length = resolve_dataset_dims(
        args, checkpoint[1] if checkpoint is not None else None
    )
    dataset_config = DatasetConfig(
        source=args.dataset_source,
        repo_id=args.repo_id,
        image_keys=(args.image_key,),
        image_size=image_size,
        frame_delta=args.frame_delta,
        num_future_frames=num_future_frames,
        action_horizon=action_horizon,
        max_samples=args.max_samples,
        synthetic_samples=args.synthetic_samples,
        episodes=args.episodes,
        idm_history_length=idm_history_length,
        seed=args.seed,
    )
    base_dataset = create_dataset(dataset_config)
    if args.future_source == "cached":
        if args.cached_future_dir is None:
            raise ValueError("--cached-future-dir is required when --future-source cached.")
        dataset = CachedFutureDataset(base_dataset, args.cached_future_dir)
    else:
        dataset = base_dataset
    if len(dataset) < args.batch_size:
        raise ValueError(f"batch_size ({args.batch_size}) must be <= dataset length ({len(dataset)}).")

    idm, model_config, idm_source = resolve_idm(
        checkpoint=checkpoint,
        dataset=dataset,
        dataset_config=dataset_config,
        device=device,
    )
    idm.eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    metrics = evaluate_idm(idm, loader, device, flow_eval_seed=args.flow_eval_seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    visual = write_visual_debug_sample(dataset, output_dir, args.fps, args.prompt_template, args.future_source)
    output = {
        "future_source": args.future_source,
        "idm_source": idm_source,
        "dataset_config": dataclasses.asdict(dataset_config),
        "model_config": dataclasses.asdict(model_config),
        "flow_eval_seed": args.flow_eval_seed if idm_uses_flow_matching(idm) else None,
        "metrics": metrics,
        "visual_debug": visual,
    }
    (output_dir / "pipeline_eval.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))
