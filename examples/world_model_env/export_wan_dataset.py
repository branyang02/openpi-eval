from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path

import torch
import tyro

from world_model.config import DatasetConfig, DatasetSource, Wan22Config
from world_model.data import create_dataset, expected_wan_selected_frame_indices, expected_wan_source_frame_offsets
from world_model.media import save_png, save_video


@dataclasses.dataclass
class Args:
    dataset_source: DatasetSource = "lerobot"
    repo_id: str = "brandonyang/metaworld_ml45"
    image_key: str = "corner4.image"
    output_dir: str = "output/wan_dataset"
    episodes: tuple[int, ...] | None = None
    max_samples: int = 256
    synthetic_samples: int = 256
    image_size: int = 256
    frame_delta: int = 1
    num_future_frames: int = 16
    action_horizon: int = 32
    fps: int = 12
    seed: int = 7
    prompt_template: str = Wan22Config.prompt_template


def task_text_for_index(dataset, index: int, fallback_task_id: int) -> str:
    if hasattr(dataset, "task_text"):
        return dataset.task_text(index)
    return f"metaworld task id {fallback_task_id}"


def validate_wan_frame_count(num_future_frames: int) -> None:
    total_frames = num_future_frames + 1
    if total_frames % 4 != 1:
        raise ValueError(
            "Wan/DiffSynth video training expects total clip frames to be 4n+1. "
            f"Got current + future = {total_frames}; choose num_future_frames divisible by 4."
        )


def main(args: Args) -> None:
    validate_wan_frame_count(args.num_future_frames)
    dataset = create_dataset(
        DatasetConfig(
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
    )

    output_dir = Path(args.output_dir)
    manifest = []
    metadata_rows = []
    selected_frame_indices = expected_wan_selected_frame_indices(args.frame_delta, args.num_future_frames)
    source_frame_offsets = expected_wan_source_frame_offsets(args.frame_delta, args.num_future_frames)
    for index in range(len(dataset)):
        item = dataset[index]
        task_text = task_text_for_index(dataset, index, int(item["task_id"]))
        prompt = args.prompt_template.format(task=task_text)
        stem = f"sample_{index:06d}"
        image_path = output_dir / "images" / f"{stem}.png"
        video_path = output_dir / "videos" / f"{stem}.mp4"
        caption_path = output_dir / "captions" / f"{stem}.txt"
        relative_video_path = video_path.relative_to(output_dir)

        save_png(item["current_images"][0], image_path)
        wan_clip = torch.cat([item["current_images"][0].unsqueeze(0), item["future_images"][:, 0]], dim=0)
        save_video(wan_clip, video_path, args.fps)
        caption_path.parent.mkdir(parents=True, exist_ok=True)
        caption_path.write_text(prompt + "\n")
        manifest.append(
            {
                "image": str(image_path.relative_to(output_dir)),
                "video": str(relative_video_path),
                "caption": str(caption_path.relative_to(output_dir)),
                "prompt": prompt,
                "num_frames": int(wan_clip.shape[0]),
                "conditioning_frame": 0,
                "selected_frame_indices": selected_frame_indices,
                "dataset_frame_delta": args.frame_delta,
                "source_frame_offsets": source_frame_offsets,
                "task_id": int(item["task_id"]),
            }
        )
        metadata_rows.append({"video": str(relative_video_path), "prompt": prompt})

    (output_dir / "manifest.jsonl").write_text("\n".join(json.dumps(row) for row in manifest) + "\n")
    with (output_dir / "metadata.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["video", "prompt"])
        writer.writeheader()
        writer.writerows(metadata_rows)
    print(json.dumps({"output_dir": str(output_dir), "num_samples": len(manifest)}, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))
