from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import tyro
from torch.utils.data import DataLoader

from world_model.config import DatasetConfig, DatasetSource, TrainConfig
from world_model.data import create_dataset
from world_model.train_lib import evaluate, load_checkpoint, resolve_device, save_prediction_grid


@dataclasses.dataclass
class Args:
    checkpoint: str
    dataset_source: DatasetSource = "synthetic"
    repo_id: str = "brandonyang/metaworld_ml45"
    image_keys: tuple[str, ...] = ("corner.image", "corner4.image", "gripperPOV.image")
    output_dir: str = "output/eval"
    image_size: int = 64
    frame_delta: int = 4
    num_future_frames: int = 1
    action_horizon: int = 32
    max_samples: int | None = None
    episodes: tuple[int, ...] | None = None
    synthetic_samples: int = 128
    batch_size: int = 16
    device: str = "auto"
    seed: int = 7


def main(args: Args) -> None:
    device = resolve_device(args.device)
    world_model, idm, _ = load_checkpoint(args.checkpoint, device)
    dataset_config = DatasetConfig(
        source=args.dataset_source,
        repo_id=args.repo_id,
        image_keys=args.image_keys,
        frame_delta=args.frame_delta,
        num_future_frames=args.num_future_frames,
        action_horizon=args.action_horizon,
        image_size=args.image_size,
        max_samples=args.max_samples,
        episodes=args.episodes,
        synthetic_samples=args.synthetic_samples,
        seed=args.seed,
    )
    dataset = create_dataset(dataset_config)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    train_config = TrainConfig(dataset=dataset_config, batch_size=args.batch_size, device=args.device, seed=args.seed)
    metrics = evaluate(world_model, idm, loader, train_config, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    save_prediction_grid(world_model, loader, output_dir / "prediction_grid.png", device)
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))
