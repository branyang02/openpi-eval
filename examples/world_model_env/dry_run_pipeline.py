from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Literal

import torch
import tyro

from world_model.config import DatasetConfig, DatasetSource, ModelConfig, Wan22Config
from world_model.data import create_dataset, infer_batch_spec
from world_model.media import save_png, save_video
from world_model.models import InverseDynamicsModel
from world_model.train_lib import resolve_device, seed_everything

FutureSource = Literal["dataset_future"]


@dataclasses.dataclass
class Args:
    dataset_source: DatasetSource = "synthetic"
    repo_id: str = "brandonyang/metaworld_ml45"
    image_key: str = "corner4.image"
    output_dir: str = "output/dry_run_pipeline"
    sample_index: int = 0
    episodes: tuple[int, ...] | None = None
    max_samples: int | None = None
    synthetic_samples: int = 16
    image_size: int = 64
    frame_delta: int = 1
    num_future_frames: int = 4
    action_horizon: int = 8
    fps: int = 12
    future_source: FutureSource = "dataset_future"
    device: str = "auto"
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


def main(args: Args) -> None:
    if args.future_source != "dataset_future":
        raise ValueError(f"Unknown future_source: {args.future_source}")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    dataset_config = DatasetConfig(
        source=args.dataset_source,
        repo_id=args.repo_id,
        image_keys=(args.image_key,),
        image_size=args.image_size,
        frame_delta=args.frame_delta,
        num_future_frames=args.num_future_frames,
        action_horizon=args.action_horizon,
        max_samples=args.max_samples,
        synthetic_samples=args.synthetic_samples,
        episodes=args.episodes,
        seed=args.seed,
    )
    dataset = create_dataset(dataset_config)
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(args.sample_index)

    item = dataset[args.sample_index]
    task_text = task_text_for_index(dataset, args.sample_index, int(item["task_id"]))
    prompt = args.prompt_template.format(task=task_text)
    model_config = model_config_from_dataset(dataset, dataset_config)
    idm = InverseDynamicsModel(model_config).to(device).eval()

    current_images = item["current_images"].unsqueeze(0).to(device)
    future_images = item["future_images"].unsqueeze(0).to(device)
    state = item["state"].unsqueeze(0).to(device)
    task_id = item["task_id"].unsqueeze(0).to(device)
    with torch.no_grad():
        action_chunk = idm(current_images, future_images, state, task_id)[0].detach().cpu()

    output_dir = Path(args.output_dir)
    current_path = output_dir / "current_frame.png"
    wan_like_video_path = output_dir / "wan_like_future.mp4"
    save_png(item["current_images"][0], current_path)
    wan_like_clip = torch.cat([item["current_images"][0].unsqueeze(0), item["future_images"][:, 0]], dim=0)
    save_video(wan_like_clip, wan_like_video_path, args.fps)

    output = {
        "future_source": args.future_source,
        "prompt": prompt,
        "current_frame": str(current_path),
        "wan_like_video": str(wan_like_video_path),
        "current_images_shape": list(item["current_images"].shape),
        "future_images_shape": list(item["future_images"].shape),
        "action_chunk_shape": list(action_chunk.shape),
        "action_chunk": action_chunk.tolist(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dry_run.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))
