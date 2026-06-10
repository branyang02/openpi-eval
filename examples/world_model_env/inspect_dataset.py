from __future__ import annotations

import dataclasses
import json

import tyro

from world_model.config import DatasetConfig, DatasetSource
from world_model.data import create_dataset


@dataclasses.dataclass
class Args:
    dataset_source: DatasetSource = "lerobot"
    repo_id: str = "brandonyang/metaworld_ml45"
    image_keys: tuple[str, ...] = ("corner.image", "corner4.image", "gripperPOV.image")
    image_size: int = 64
    frame_delta: int = 4
    num_future_frames: int = 1
    action_horizon: int = 32
    max_samples: int | None = None
    samples_per_episode: int | None = None
    episodes: tuple[int, ...] | None = None
    synthetic_samples: int = 8
    seed: int = 7


def main(args: Args) -> None:
    config = DatasetConfig(
        source=args.dataset_source,
        repo_id=args.repo_id,
        image_keys=args.image_keys,
        image_size=args.image_size,
        frame_delta=args.frame_delta,
        num_future_frames=args.num_future_frames,
        action_horizon=args.action_horizon,
        max_samples=args.max_samples,
        samples_per_episode=args.samples_per_episode,
        episodes=args.episodes,
        synthetic_samples=args.synthetic_samples,
        seed=args.seed,
    )
    dataset = create_dataset(config)
    sample = dataset[0]
    summary = {
        "length": len(dataset),
        "sample": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "min": float(value.min()) if value.numel() else None,
                "max": float(value.max()) if value.numel() else None,
            }
            for key, value in sample.items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))
