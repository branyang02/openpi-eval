from __future__ import annotations

import dataclasses

import tyro

from world_model.config import DatasetConfig, DatasetSource, IdmTargetSource, TrainConfig
from world_model.train_lib import run_training


@dataclasses.dataclass
class Args:
    dataset_source: DatasetSource = "synthetic"
    repo_id: str = "brandonyang/metaworld_ml45"
    image_keys: tuple[str, ...] = ("corner.image", "corner4.image", "gripperPOV.image")
    output_dir: str = "output/smoke"
    epochs: int = 2
    batch_size: int = 16
    learning_rate: float = 1e-4
    image_size: int = 64
    frame_delta: int = 4
    num_future_frames: int = 1
    action_horizon: int = 32
    max_samples: int | None = None
    episodes: tuple[int, ...] | None = None
    synthetic_samples: int = 128
    idm_target_source: IdmTargetSource = "ground_truth"
    num_workers: int = 0
    device: str = "auto"
    data_parallel: bool = False
    seed: int = 7


def main(args: Args) -> None:
    dataset = DatasetConfig(
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
    config = TrainConfig(
        dataset=dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        idm_target_source=args.idm_target_source,
        num_workers=args.num_workers,
        device=args.device,
        data_parallel=args.data_parallel,
        seed=args.seed,
    )
    run_training(config)


if __name__ == "__main__":
    main(tyro.cli(Args))
