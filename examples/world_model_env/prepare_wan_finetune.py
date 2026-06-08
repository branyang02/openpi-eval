from __future__ import annotations

import csv
import dataclasses
import json
import shlex
from pathlib import Path
from typing import Literal

import imageio.v3 as iio
import tyro

TrainMode = Literal["lora", "full"]


@dataclasses.dataclass
class Args:
    dataset_dir: str
    output_path: str = "./models/train/Wan2.2-TI2V-5B_lora"
    mode: TrainMode = "lora"
    height: int = 480
    width: int = 832
    num_frames: int = 17
    dataset_repeat: int = 100
    learning_rate: float | None = None
    num_epochs: int | None = None
    lora_rank: int = 32
    lora_target_modules: str = "q,k,v,o,ffn.0,ffn.2"
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B"
    model_paths: tuple[str, ...] | None = None
    model_paths_json: str | None = None
    tokenizer_path: str | None = None
    diffsynth_train_script: str = "examples/wanvideo/model_training/train.py"
    metadata_filename: str = "metadata.csv"
    data_file_keys: str = "video"
    gradient_accumulation_steps: int = 1
    use_gradient_checkpointing: bool = True
    enable_model_cpu_offload: bool = False
    enable_optimizer_cpu_offload: bool = False
    accelerate_config_file: str | None = None
    accelerate_num_processes: int | None = None
    validate_videos: bool = True


def read_metadata(metadata_path: Path) -> list[dict[str, str]]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")
    with metadata_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"metadata.csv is empty: {metadata_path}")
    required = {"video", "prompt"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"metadata.csv missing required column(s): {sorted(missing)}")
    return rows


def video_num_frames(video_path: Path) -> int:
    return sum(1 for _ in iio.imiter(video_path))


def validate_dataset(dataset_dir: Path, metadata_filename: str, expected_num_frames: int, validate_videos: bool) -> dict:
    metadata_path = dataset_dir / metadata_filename
    rows = read_metadata(metadata_path)
    checked_videos = 0
    if validate_videos:
        for row in rows:
            video_path = dataset_dir / row["video"]
            if not video_path.exists():
                raise FileNotFoundError(f"Video from metadata.csv does not exist: {video_path}")
            frames = video_num_frames(video_path)
            if frames != expected_num_frames:
                raise ValueError(f"{video_path} has {frames} frames, expected {expected_num_frames}.")
            checked_videos += 1
    return {
        "dataset_dir": str(dataset_dir),
        "metadata_path": str(metadata_path),
        "num_rows": len(rows),
        "checked_videos": checked_videos,
        "expected_num_frames": expected_num_frames,
    }


def build_accelerate_prefix(args: Args) -> list[str]:
    command = ["accelerate", "launch"]
    if args.accelerate_config_file is not None:
        command.extend(["--config_file", args.accelerate_config_file])
    if args.accelerate_num_processes is not None:
        command.extend(["--num_processes", str(args.accelerate_num_processes)])
    return command


def build_model_loading_args(args: Args) -> list[str]:
    if args.model_paths is not None and args.model_paths_json is not None:
        raise ValueError("Only one of model_paths or model_paths_json may be provided.")
    if args.model_paths_json is not None:
        decoded = json.loads(args.model_paths_json)
        if not isinstance(decoded, list) or not decoded:
            raise ValueError("model_paths_json must decode to a non-empty JSON list.")
        return ["--model_paths", args.model_paths_json]
    if args.model_paths is not None:
        if not args.model_paths:
            raise ValueError("model_paths must contain at least one path when provided.")
        return ["--model_paths", json.dumps(list(args.model_paths))]
    return [
        "--model_id_with_origin_paths",
        (
            f"{args.model_id}:diffusion_pytorch_model*.safetensors,"
            f"{args.model_id}:models_t5_umt5-xxl-enc-bf16.pth,"
            f"{args.model_id}:Wan2.2_VAE.pth"
        ),
    ]


def build_command(args: Args, dataset_dir: Path) -> list[str]:
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    if args.mode == "lora":
        learning_rate = 1e-4 if learning_rate is None else learning_rate
        num_epochs = 5 if num_epochs is None else num_epochs
    elif args.mode == "full":
        learning_rate = 1e-5 if learning_rate is None else learning_rate
        num_epochs = 2 if num_epochs is None else num_epochs
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    command = [
        *build_accelerate_prefix(args),
        args.diffsynth_train_script,
        "--dataset_base_path",
        str(dataset_dir),
        "--dataset_metadata_path",
        str(dataset_dir / args.metadata_filename),
        "--data_file_keys",
        args.data_file_keys,
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--num_frames",
        str(args.num_frames),
        "--dataset_repeat",
        str(args.dataset_repeat),
        *build_model_loading_args(args),
        *([] if args.tokenizer_path is None else ["--tokenizer_path", args.tokenizer_path]),
        "--learning_rate",
        str(learning_rate),
        "--num_epochs",
        str(num_epochs),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--remove_prefix_in_ckpt",
        "pipe.dit.",
        "--output_path",
        args.output_path,
        "--extra_inputs",
        "input_image",
    ]
    if args.use_gradient_checkpointing:
        command.append("--use_gradient_checkpointing")
    if args.enable_model_cpu_offload:
        command.append("--enable_model_cpu_offload")
    if args.enable_optimizer_cpu_offload:
        command.append("--enable_optimizer_cpu_offload")
    if args.mode == "lora":
        command.extend(
            [
                "--lora_base_model",
                "dit",
                "--lora_target_modules",
                args.lora_target_modules,
                "--lora_rank",
                str(args.lora_rank),
            ]
        )
    else:
        command.extend(["--trainable_models", "dit"])
    return command


def main(args: Args) -> None:
    dataset_dir = Path(args.dataset_dir)
    summary = validate_dataset(dataset_dir, args.metadata_filename, args.num_frames, args.validate_videos)
    command = build_command(args, dataset_dir)
    output = {
        "mode": args.mode,
        "dataset": summary,
        "command": command,
        "command_text": " ".join(shlex.quote(part) for part in command),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
