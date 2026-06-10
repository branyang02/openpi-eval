from __future__ import annotations

import csv
import json

import pytest
import torch

from run_wan_finetune import Args, main
from world_model.media import save_video


def write_wan_dataset(dataset_dir, *, num_frames: int = 5) -> None:
    (dataset_dir / "videos").mkdir(parents=True)
    save_video(torch.rand(num_frames, 3, 16, 16), dataset_dir / "videos" / "sample.mp4", fps=12)
    with (dataset_dir / "metadata.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["video", "prompt"])
        writer.writeheader()
        writer.writerow({"video": "videos/sample.mp4", "prompt": "Robot manipulation in MetaWorld."})


def write_fake_diffsynth_repo(repo_dir) -> None:
    (repo_dir / "examples" / "wanvideo" / "model_training").mkdir(parents=True)
    (repo_dir / "examples" / "wanvideo" / "model_training" / "train.py").write_text("print('fake train')\n")
    (repo_dir / "diffsynth").mkdir()


def write_fake_wan_checkpoint(checkpoint_dir) -> None:
    for relative in [
        "Wan2.2_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "diffusion_pytorch_model.safetensors.index.json",
        "diffusion_pytorch_model-00001-of-00003.safetensors",
        "google/umt5-xxl/spiece.model",
        "google/umt5-xxl/tokenizer.json",
        "google/umt5-xxl/tokenizer_config.json",
    ]:
        path = checkpoint_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stub\n")


def test_run_wan_finetune_preflight_writes_official_lora_command(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    repo_dir = tmp_path / "DiffSynth-Studio"
    checkpoint_dir = tmp_path / "Wan2.2-TI2V-5B"
    preflight_dir = tmp_path / "preflight"
    write_wan_dataset(dataset_dir)
    write_fake_diffsynth_repo(repo_dir)
    write_fake_wan_checkpoint(checkpoint_dir)

    main(
        Args(
            dataset_dir=str(dataset_dir),
            diffsynth_repo_dir=str(repo_dir),
            checkpoint_dir=str(checkpoint_dir),
            preflight_output_dir=str(preflight_dir),
            output_path=str(tmp_path / "lora"),
            num_frames=5,
            height=16,
            width=16,
            dataset_repeat=2,
            num_epochs=1,
            validate_diffsynth_import=False,
            run=False,
        )
    )

    preflight = json.loads((preflight_dir / "wan_finetune_preflight.json").read_text())
    command = preflight["command"]

    assert preflight["dataset"]["num_rows"] == 1
    assert preflight["checkpoint"]["diffusion_shards"] == ["diffusion_pytorch_model-00001-of-00003.safetensors"]
    assert preflight["checkpoint"]["tokenizer_path"] == str(checkpoint_dir / "google" / "umt5-xxl")
    assert "--data_file_keys" in command
    assert command[command.index("--data_file_keys") + 1] == "video"
    assert "--model_paths" in command
    assert "--model_id_with_origin_paths" not in command
    model_paths = json.loads(command[command.index("--model_paths") + 1])
    assert model_paths == [
        str(checkpoint_dir / "models_t5_umt5-xxl-enc-bf16.pth"),
        [str(checkpoint_dir / "diffusion_pytorch_model-00001-of-00003.safetensors")],
        str(checkpoint_dir / "Wan2.2_VAE.pth"),
    ]
    assert "--tokenizer_path" in command
    assert command[command.index("--tokenizer_path") + 1] == str(checkpoint_dir / "google" / "umt5-xxl")
    assert "--extra_inputs" in command
    assert command[command.index("--extra_inputs") + 1] == "input_image"
    assert "--lora_base_model" in command
    assert "--use_gradient_checkpointing" in command
    assert preflight["run_requested"] is False


def test_run_wan_finetune_preflight_rejects_incomplete_checkpoint(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    repo_dir = tmp_path / "DiffSynth-Studio"
    checkpoint_dir = tmp_path / "Wan2.2-TI2V-5B"
    write_wan_dataset(dataset_dir)
    write_fake_diffsynth_repo(repo_dir)
    checkpoint_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="missing required"):
        main(
            Args(
                dataset_dir=str(dataset_dir),
                diffsynth_repo_dir=str(repo_dir),
                checkpoint_dir=str(checkpoint_dir),
                num_frames=5,
                validate_diffsynth_import=False,
            )
        )
