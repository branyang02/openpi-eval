from __future__ import annotations

import csv
import json

import pytest
import torch

from prepare_wan_finetune import Args, build_command, validate_dataset
from world_model.media import save_video


def write_metadata(dataset_dir, video_name: str = "videos/sample.mp4") -> None:
    (dataset_dir / "videos").mkdir(parents=True, exist_ok=True)
    with (dataset_dir / "metadata.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["video", "prompt"])
        writer.writeheader()
        writer.writerow({"video": video_name, "prompt": "Robot manipulation in MetaWorld."})


def test_prepare_wan_finetune_validates_dataset_and_builds_lora_command(tmp_path) -> None:
    write_metadata(tmp_path)
    save_video(torch.rand(5, 3, 16, 16), tmp_path / "videos" / "sample.mp4", fps=12)

    summary = validate_dataset(tmp_path, "metadata.csv", expected_num_frames=5, validate_videos=True)
    command = build_command(Args(dataset_dir=str(tmp_path), num_frames=5), tmp_path)

    assert summary["num_rows"] == 1
    assert summary["checked_videos"] == 1
    assert "--extra_inputs" in command
    assert "input_image" in command
    assert "--lora_base_model" in command
    assert "--trainable_models" not in command


def test_prepare_wan_finetune_builds_full_training_command(tmp_path) -> None:
    command = build_command(Args(dataset_dir=str(tmp_path), mode="full"), tmp_path)

    assert "--trainable_models" in command
    assert "dit" in command
    assert "--lora_base_model" not in command


def test_prepare_wan_finetune_accepts_nested_model_paths_json(tmp_path) -> None:
    model_paths = ["/ckpt/text.pth", ["/ckpt/dit-1.safetensors", "/ckpt/dit-2.safetensors"], "/ckpt/vae.pth"]
    command = build_command(
        Args(dataset_dir=str(tmp_path), model_paths_json=json.dumps(model_paths), tokenizer_path="/ckpt/google/umt5-xxl"),
        tmp_path,
    )

    assert "--model_paths" in command
    assert json.loads(command[command.index("--model_paths") + 1]) == model_paths
    assert "--model_id_with_origin_paths" not in command
    assert "--tokenizer_path" in command
    assert command[command.index("--tokenizer_path") + 1] == "/ckpt/google/umt5-xxl"


def test_prepare_wan_finetune_rejects_wrong_frame_count(tmp_path) -> None:
    write_metadata(tmp_path)
    save_video(torch.rand(4, 3, 16, 16), tmp_path / "videos" / "sample.mp4", fps=12)

    with pytest.raises(ValueError, match="expected 5"):
        validate_dataset(tmp_path, "metadata.csv", expected_num_frames=5, validate_videos=True)
