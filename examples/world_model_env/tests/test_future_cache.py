from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from cache_future_rollouts import Args as CacheArgs
from cache_future_rollouts import main as cache_main
from eval_idm import Args as EvalIdmArgs
from eval_idm import main as eval_idm_main
from eval_pipeline import Args as EvalArgs
from eval_pipeline import main as eval_main
from evaluate_future_cache import Args as EvaluateCacheArgs
from evaluate_future_cache import main as evaluate_cache_main
from inspect_dataset import Args as InspectArgs
from world_model.config import DatasetConfig, TrainConfig
from world_model.data import (
    CachedFutureDataset,
    MixedFutureDataset,
    SyntheticMetaWorldFramePairDataset,
    expected_wan_source_frame_offsets,
)
from world_model.train_lib import create_dataset_with_optional_cache, run_idm_training
from world_model.wan22 import Wan22Result


def mark_cache_as_wan_with_selected_indices(
    cache_dir: Path,
    selected_frame_indices: list[int],
    *,
    future_frame_strategy: str = "first",
) -> None:
    rows = [json.loads(line) for line in (cache_dir / "manifest.jsonl").read_text().splitlines() if line.strip()]
    config = json.loads((cache_dir / "config.json").read_text())
    source_frame_offsets = expected_wan_source_frame_offsets(
        config["dataset_config"]["frame_delta"],
        config["dataset_config"]["num_future_frames"],
    )
    for row in rows:
        row["source"] = "wan_lora"
        row["future_frame_strategy"] = future_frame_strategy
        row["selected_frame_indices"] = selected_frame_indices
        row["total_video_frames"] = max(selected_frame_indices) + 1
        row["dataset_frame_delta"] = config["dataset_config"]["frame_delta"]
        row["source_frame_offsets"] = source_frame_offsets
    (cache_dir / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    config["future_source"] = "wan_lora"
    config["future_frame_selection"] = {
        "future_frame_strategy": future_frame_strategy,
        "selected_frame_indices": selected_frame_indices,
        "selected_frame_indices_by_dataset_index": {
            str(int(row["dataset_index"])): selected_frame_indices for row in rows
        },
        "total_video_frames": max(selected_frame_indices) + 1,
        "dataset_frame_delta": config["dataset_config"]["frame_delta"],
        "frame_delta": config["dataset_config"]["frame_delta"],
        "source_frame_offsets": source_frame_offsets,
        "num_future_frames": config["dataset_config"]["num_future_frames"],
    }
    (cache_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def test_cache_future_rollouts_writes_dataset_future_cache(tmp_path) -> None:
    cache_main(
        CacheArgs(
            future_source="dataset_future",
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            max_samples=3,
            synthetic_samples=3,
            image_size=32,
            num_future_frames=4,
            action_horizon=8,
        )
    )

    rows = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]

    assert len(rows) == 3
    assert (tmp_path / rows[0]["current_image"]).exists()
    assert (tmp_path / rows[0]["future_tensor"]).exists()
    assert (tmp_path / rows[0]["video"]).exists()
    assert rows[0]["future_shape"] == [4, 1, 3, 32, 32]


def test_cache_future_rollouts_skips_completed_futures_on_second_run(tmp_path, monkeypatch) -> None:
    args = CacheArgs(
        future_source="dataset_future",
        dataset_source="synthetic",
        output_dir=str(tmp_path),
        max_samples=3,
        synthetic_samples=3,
        image_size=32,
        num_future_frames=4,
        action_horizon=8,
    )
    cache_main(args)
    manifest_text = (tmp_path / "manifest.jsonl").read_text()
    future_paths = sorted((tmp_path / "futures").glob("sample_*.pt"))
    future_mtimes = {path: path.stat().st_mtime_ns for path in future_paths}
    saved_paths: list[Path] = []
    original_save = torch.save

    def recording_save(obj, f, *args, **kwargs):
        saved_paths.append(Path(f).resolve())
        return original_save(obj, f, *args, **kwargs)

    monkeypatch.setattr(torch, "save", recording_save)

    cache_main(args)

    assert saved_paths == []
    assert (tmp_path / "manifest.jsonl").read_text() == manifest_text
    assert {path: path.stat().st_mtime_ns for path in future_paths} == future_mtimes


def test_cache_future_rollouts_resumes_partial_cache_manifest(tmp_path, monkeypatch) -> None:
    args = CacheArgs(
        future_source="dataset_future",
        dataset_source="synthetic",
        output_dir=str(tmp_path),
        max_samples=3,
        synthetic_samples=3,
        image_size=32,
        num_future_frames=4,
        action_horizon=8,
    )
    cache_main(args)
    rows = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    future_0 = tmp_path / rows[0]["future_tensor"]
    future_1 = tmp_path / rows[1]["future_tensor"]
    future_2 = tmp_path / rows[2]["future_tensor"]
    completed_mtimes = {future_0: future_0.stat().st_mtime_ns, future_1: future_1.stat().st_mtime_ns}
    (tmp_path / "manifest.jsonl").write_text(json.dumps(rows[0]) + "\n")
    future_2.unlink()
    saved_paths: list[Path] = []
    original_save = torch.save

    def recording_save(obj, f, *args, **kwargs):
        saved_paths.append(Path(f).resolve())
        return original_save(obj, f, *args, **kwargs)

    monkeypatch.setattr(torch, "save", recording_save)

    cache_main(args)

    resumed_rows = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    config = json.loads((tmp_path / "config.json").read_text())

    assert [row["dataset_index"] for row in resumed_rows] == [0, 1, 2]
    assert saved_paths == [future_2.resolve()]
    assert {path: path.stat().st_mtime_ns for path in completed_mtimes} == completed_mtimes
    assert config["num_samples"] == 3
    for row in resumed_rows:
        assert (tmp_path / row["current_image"]).exists()
        assert (tmp_path / row["future_tensor"]).exists()
        assert (tmp_path / row["video"]).exists()


def test_cache_future_rollouts_rejects_resume_with_different_dataset_config(
    tmp_path,
    monkeypatch,
) -> None:
    args = CacheArgs(
        future_source="dataset_future",
        dataset_source="synthetic",
        output_dir=str(tmp_path),
        max_samples=3,
        synthetic_samples=3,
        image_size=32,
        num_future_frames=4,
        action_horizon=8,
    )
    cache_main(args)
    manifest_text = (tmp_path / "manifest.jsonl").read_text()
    config_text = (tmp_path / "config.json").read_text()
    future_paths = sorted((tmp_path / "futures").glob("sample_*.pt"))
    future_mtimes = {path: path.stat().st_mtime_ns for path in future_paths}
    saved_paths: list[Path] = []
    original_save = torch.save

    def recording_save(obj, f, *save_args, **save_kwargs):
        saved_paths.append(Path(f).resolve())
        return original_save(obj, f, *save_args, **save_kwargs)

    monkeypatch.setattr(torch, "save", recording_save)

    with pytest.raises(ValueError, match="max_samples"):
        cache_main(dataclasses.replace(args, max_samples=2))

    assert saved_paths == []
    assert (tmp_path / "manifest.jsonl").read_text() == manifest_text
    assert (tmp_path / "config.json").read_text() == config_text
    assert {path: path.stat().st_mtime_ns for path in future_paths} == future_mtimes


def test_cache_future_rollouts_rejects_balanced_split_resume_before_dataset_load(tmp_path) -> None:
    old_config = DatasetConfig(
        source="lerobot",
        image_keys=("corner4.image",),
        image_size=32,
        frame_delta=1,
        num_future_frames=1,
        action_horizon=2,
        episodes=(10, 20),
        samples_per_episode=1,
        synthetic_samples=8,
    )
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps({"source": "dataset_future", "dataset_index": 0, "future_tensor": "futures/sample_000000.pt"}) + "\n"
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "future_source": "dataset_future",
                "dataset_config": dataclasses.asdict(old_config),
                "num_samples": 1,
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="samples_per_episode"):
        cache_main(
            CacheArgs(
                future_source="dataset_future",
                dataset_source="lerobot",
                output_dir=str(tmp_path),
                repo_id=old_config.repo_id,
                image_key="corner4.image",
                image_size=32,
                frame_delta=1,
                num_future_frames=1,
                action_horizon=2,
                episodes=(10, 20),
                samples_per_episode=2,
                synthetic_samples=8,
            )
        )


def test_samples_per_episode_cli_args_do_not_carry_max_samples_default() -> None:
    assert CacheArgs(dataset_source="lerobot", samples_per_episode=2).max_samples is None
    assert EvaluateCacheArgs(dataset_source="lerobot", samples_per_episode=2).max_samples is None
    assert InspectArgs(samples_per_episode=2).max_samples is None


def test_cache_future_rollouts_writes_wan_lora_cache(tmp_path, monkeypatch) -> None:
    class FakeWanLoraGenerator:
        def generate_future_stack(self, current_images, *, task_text, output_dir, image_size, num_future_frames, seed):
            output_dir.mkdir(parents=True, exist_ok=True)
            video_path = output_dir / "wan_lora_view0.mp4"
            video_path.write_bytes(b"fake mp4")
            return Wan22Result(
                prompt=task_text,
                seed=seed,
                input_image_path=output_dir / "wan_lora_view0_input.png",
                video_path=video_path,
                future_images=torch.zeros(num_future_frames, 1, 3, image_size, image_size),
                selected_frame_indices=tuple(range(1, num_future_frames + 1)),
                total_video_frames=num_future_frames + 1,
            )

    monkeypatch.setattr("cache_future_rollouts.build_wan_lora_generator", lambda args: FakeWanLoraGenerator())

    cache_main(
        CacheArgs(
            future_source="wan_lora",
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            max_samples=2,
            synthetic_samples=2,
            image_size=32,
            num_future_frames=4,
            action_horizon=8,
            diffsynth_repo_dir="/tmp/fake-diffsynth",
            wan_lora_checkpoint_dir="/tmp/fake-wan",
            wan_lora_path="/tmp/fake-lora.safetensors",
        )
    )

    rows = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    cached_future = torch.load(tmp_path / rows[0]["future_tensor"], map_location="cpu", weights_only=False)

    assert len(rows) == 2
    assert rows[0]["source"] == "wan_lora"
    assert rows[0]["video"] == "wan_lora_raw/sample_000000/wan_lora_view0.mp4"
    assert rows[0]["generation_seed"] == 7
    assert rows[0]["future_frame_strategy"] == "first"
    assert rows[0]["selected_frame_indices"] == [1, 2, 3, 4]
    assert rows[0]["total_video_frames"] == 5
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["future_frame_selection"]["future_frame_strategy"] == "first"
    assert config["future_frame_selection"]["selected_frame_indices"] == [1, 2, 3, 4]
    assert config["future_frame_selection"]["total_video_frames"] == 5
    assert config["future_frame_selection"]["dataset_frame_delta"] == 1
    assert config["future_frame_selection"]["frame_delta"] == 1
    assert config["future_frame_selection"]["source_frame_offsets"] == [1, 2, 3, 4]
    assert config["future_frame_selection"]["num_future_frames"] == 4
    assert rows[0]["dataset_frame_delta"] == 1
    assert rows[0]["source_frame_offsets"] == [1, 2, 3, 4]
    assert cached_future.shape == (4, 1, 3, 32, 32)


def test_cache_future_rollouts_can_vary_wan_generation_seed_without_dataset_seed(tmp_path, monkeypatch) -> None:
    seen_seeds: list[int] = []

    class FakeWanLoraGenerator:
        def generate_future_stack(self, current_images, *, task_text, output_dir, image_size, num_future_frames, seed):
            del current_images
            output_dir.mkdir(parents=True, exist_ok=True)
            video_path = output_dir / "wan_lora_view0.mp4"
            video_path.write_bytes(b"fake mp4")
            seen_seeds.append(seed)
            return Wan22Result(
                prompt=task_text,
                seed=seed,
                input_image_path=output_dir / "wan_lora_view0_input.png",
                video_path=video_path,
                future_images=torch.zeros(num_future_frames, 1, 3, image_size, image_size),
                selected_frame_indices=tuple(range(1, num_future_frames + 1)),
                total_video_frames=num_future_frames + 1,
            )

    monkeypatch.setattr("cache_future_rollouts.build_wan_lora_generator", lambda args: FakeWanLoraGenerator())

    cache_main(
        CacheArgs(
            future_source="wan_lora",
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            max_samples=2,
            synthetic_samples=2,
            image_size=32,
            num_future_frames=4,
            action_horizon=8,
            seed=7,
            generation_seed=123,
            diffsynth_repo_dir="/tmp/fake-diffsynth",
            wan_lora_checkpoint_dir="/tmp/fake-wan",
            wan_lora_path="/tmp/fake-lora.safetensors",
        )
    )

    rows = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    config = json.loads((tmp_path / "config.json").read_text())

    assert seen_seeds == [123, 124]
    assert [row["generation_seed"] for row in rows] == [123, 124]
    assert config["dataset_config"]["seed"] == 7
    assert config["generation_seed"] == 123


def test_cache_future_rollouts_rejects_wan_lora_resume_with_different_generation_seed(tmp_path, monkeypatch) -> None:
    class FakeWanLoraGenerator:
        def generate_future_stack(self, current_images, *, task_text, output_dir, image_size, num_future_frames, seed):
            del current_images
            output_dir.mkdir(parents=True, exist_ok=True)
            video_path = output_dir / "wan_lora_view0.mp4"
            video_path.write_bytes(b"fake mp4")
            return Wan22Result(
                prompt=task_text,
                seed=seed,
                input_image_path=output_dir / "wan_lora_view0_input.png",
                video_path=video_path,
                future_images=torch.zeros(num_future_frames, 1, 3, image_size, image_size),
                selected_frame_indices=tuple(range(1, num_future_frames + 1)),
                total_video_frames=num_future_frames + 1,
            )

    monkeypatch.setattr("cache_future_rollouts.build_wan_lora_generator", lambda args: FakeWanLoraGenerator())
    args = CacheArgs(
        future_source="wan_lora",
        dataset_source="synthetic",
        output_dir=str(tmp_path),
        max_samples=2,
        synthetic_samples=2,
        image_size=32,
        num_future_frames=4,
        action_horizon=8,
        seed=7,
        generation_seed=123,
        diffsynth_repo_dir="/tmp/fake-diffsynth",
        wan_lora_checkpoint_dir="/tmp/fake-wan",
        wan_lora_path="/tmp/fake-lora.safetensors",
    )
    cache_main(args)

    with pytest.raises(ValueError, match="different generation_seed"):
        cache_main(dataclasses.replace(args, generation_seed=456))


def test_cache_future_rollouts_rejects_wan_lora_resume_with_different_temporal_contract(tmp_path, monkeypatch) -> None:
    class FakeWanLoraGenerator:
        def generate_future_stack(self, current_images, *, task_text, output_dir, image_size, num_future_frames, seed):
            del current_images
            output_dir.mkdir(parents=True, exist_ok=True)
            video_path = output_dir / "wan_lora_view0.mp4"
            video_path.write_bytes(b"fake mp4")
            return Wan22Result(
                prompt=task_text,
                seed=seed,
                input_image_path=output_dir / "wan_lora_view0_input.png",
                video_path=video_path,
                future_images=torch.zeros(num_future_frames, 1, 3, image_size, image_size),
                selected_frame_indices=tuple(range(1, num_future_frames + 1)),
                total_video_frames=num_future_frames + 1,
            )

    monkeypatch.setattr("cache_future_rollouts.build_wan_lora_generator", lambda args: FakeWanLoraGenerator())
    args = CacheArgs(
        future_source="wan_lora",
        dataset_source="synthetic",
        output_dir=str(tmp_path),
        max_samples=1,
        synthetic_samples=1,
        image_size=32,
        frame_delta=1,
        num_future_frames=4,
        action_horizon=8,
        diffsynth_repo_dir="/tmp/fake-diffsynth",
        wan_lora_checkpoint_dir="/tmp/fake-wan",
        wan_lora_path="/tmp/fake-lora.safetensors",
    )
    cache_main(args)

    with pytest.raises(ValueError, match="different frame_delta"):
        cache_main(dataclasses.replace(args, frame_delta=2))
    with pytest.raises(ValueError, match="different num_future_frames"):
        cache_main(dataclasses.replace(args, num_future_frames=2))


def test_cache_future_rollouts_rejects_wan_lora_resume_with_different_future_frame_strategy(
    tmp_path, monkeypatch
) -> None:
    class FakeWanLoraGenerator:
        def generate_future_stack(self, current_images, *, task_text, output_dir, image_size, num_future_frames, seed):
            del current_images
            output_dir.mkdir(parents=True, exist_ok=True)
            video_path = output_dir / "wan_lora_view0.mp4"
            video_path.write_bytes(b"fake mp4")
            return Wan22Result(
                prompt=task_text,
                seed=seed,
                input_image_path=output_dir / "wan_lora_view0_input.png",
                video_path=video_path,
                future_images=torch.zeros(num_future_frames, 1, 3, image_size, image_size),
                selected_frame_indices=tuple(range(1, num_future_frames + 1)),
                total_video_frames=num_future_frames + 1,
            )

    monkeypatch.setattr("cache_future_rollouts.build_wan_lora_generator", lambda args: FakeWanLoraGenerator())
    args = CacheArgs(
        future_source="wan_lora",
        dataset_source="synthetic",
        output_dir=str(tmp_path),
        max_samples=1,
        synthetic_samples=1,
        image_size=32,
        frame_delta=4,
        num_future_frames=4,
        action_horizon=20,
        diffsynth_repo_dir="/tmp/fake-diffsynth",
        wan_lora_checkpoint_dir="/tmp/fake-wan",
        wan_lora_path="/tmp/fake-lora.safetensors",
    )
    cache_main(args)

    with pytest.raises(ValueError, match="different future_frame_strategy"):
        cache_main(dataclasses.replace(args, wan_lora_future_frame_strategy="source_offsets"))


def test_cache_future_rollouts_rejects_multiplied_wan_lora_selected_indices(tmp_path, monkeypatch) -> None:
    class FakeWanLoraGenerator:
        def generate_future_stack(self, current_images, *, task_text, output_dir, image_size, num_future_frames, seed):
            output_dir.mkdir(parents=True, exist_ok=True)
            video_path = output_dir / "wan_lora_view0.mp4"
            video_path.write_bytes(b"fake mp4")
            return Wan22Result(
                prompt=task_text,
                seed=seed,
                input_image_path=output_dir / "wan_lora_view0_input.png",
                video_path=video_path,
                future_images=torch.zeros(num_future_frames, 1, 3, image_size, image_size),
                selected_frame_indices=tuple(2 * offset for offset in range(1, num_future_frames + 1)),
                total_video_frames=(2 * num_future_frames) + 1,
            )

    monkeypatch.setattr("cache_future_rollouts.build_wan_lora_generator", lambda args: FakeWanLoraGenerator())

    with pytest.raises(ValueError, match=r"generated-video frame contract \[1, 2, 3, 4\]"):
        cache_main(
            CacheArgs(
                future_source="wan_lora",
                dataset_source="synthetic",
                output_dir=str(tmp_path),
                max_samples=1,
                synthetic_samples=1,
                image_size=32,
                frame_delta=2,
                num_future_frames=4,
                action_horizon=8,
                diffsynth_repo_dir="/tmp/fake-diffsynth",
                wan_lora_checkpoint_dir="/tmp/fake-wan",
                wan_lora_path="/tmp/fake-lora.safetensors",
            )
        )


def test_cache_future_rollouts_wan_lora_frame_delta_four_records_source_offsets(tmp_path, monkeypatch) -> None:
    class FakeWanLoraGenerator:
        def generate_future_stack(self, current_images, *, task_text, output_dir, image_size, num_future_frames, seed):
            del current_images
            output_dir.mkdir(parents=True, exist_ok=True)
            video_path = output_dir / "wan_lora_view0.mp4"
            video_path.write_bytes(b"fake mp4")
            return Wan22Result(
                prompt=task_text,
                seed=seed,
                input_image_path=output_dir / "wan_lora_view0_input.png",
                video_path=video_path,
                future_images=torch.zeros(num_future_frames, 1, 3, image_size, image_size),
                selected_frame_indices=tuple(range(1, num_future_frames + 1)),
                total_video_frames=num_future_frames + 1,
            )

    monkeypatch.setattr("cache_future_rollouts.build_wan_lora_generator", lambda args: FakeWanLoraGenerator())

    cache_main(
        CacheArgs(
            future_source="wan_lora",
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            max_samples=1,
            synthetic_samples=1,
            image_size=32,
            frame_delta=4,
            num_future_frames=2,
            action_horizon=8,
            diffsynth_repo_dir="/tmp/fake-diffsynth",
            wan_lora_checkpoint_dir="/tmp/fake-wan",
            wan_lora_path="/tmp/fake-lora.safetensors",
        )
    )

    rows = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    config = json.loads((tmp_path / "config.json").read_text())

    assert rows[0]["selected_frame_indices"] == [1, 2]
    assert rows[0]["dataset_frame_delta"] == 4
    assert rows[0]["source_frame_offsets"] == [4, 8]
    assert config["future_frame_selection"]["selected_frame_indices"] == [1, 2]
    assert config["future_frame_selection"]["dataset_frame_delta"] == 4
    assert config["future_frame_selection"]["frame_delta"] == 4
    assert config["future_frame_selection"]["source_frame_offsets"] == [4, 8]


def test_cache_future_rollouts_wan_lora_source_offsets_records_selected_indices(tmp_path, monkeypatch) -> None:
    class FakeWanLoraGenerator:
        def generate_future_stack(self, current_images, *, task_text, output_dir, image_size, num_future_frames, seed):
            del current_images
            output_dir.mkdir(parents=True, exist_ok=True)
            video_path = output_dir / "wan_lora_view0.mp4"
            video_path.write_bytes(b"fake mp4")
            selected_frame_indices = tuple(4 * offset for offset in range(1, num_future_frames + 1))
            return Wan22Result(
                prompt=task_text,
                seed=seed,
                input_image_path=output_dir / "wan_lora_view0_input.png",
                video_path=video_path,
                future_images=torch.zeros(num_future_frames, 1, 3, image_size, image_size),
                selected_frame_indices=selected_frame_indices,
                total_video_frames=max(selected_frame_indices) + 1,
            )

    monkeypatch.setattr("cache_future_rollouts.build_wan_lora_generator", lambda args: FakeWanLoraGenerator())

    cache_main(
        CacheArgs(
            future_source="wan_lora",
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            max_samples=1,
            synthetic_samples=1,
            image_size=32,
            frame_delta=4,
            num_future_frames=4,
            action_horizon=20,
            wan_lora_future_frame_strategy="source_offsets",
            diffsynth_repo_dir="/tmp/fake-diffsynth",
            wan_lora_checkpoint_dir="/tmp/fake-wan",
            wan_lora_path="/tmp/fake-lora.safetensors",
        )
    )

    rows = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    config = json.loads((tmp_path / "config.json").read_text())

    assert rows[0]["future_frame_strategy"] == "source_offsets"
    assert rows[0]["selected_frame_indices"] == [4, 8, 12, 16]
    assert rows[0]["total_video_frames"] == 17
    assert rows[0]["source_frame_offsets"] == [4, 8, 12, 16]
    assert config["future_frame_selection"]["future_frame_strategy"] == "source_offsets"
    assert config["future_frame_selection"]["selected_frame_indices"] == [4, 8, 12, 16]
    assert config["future_frame_selection"]["source_frame_offsets"] == [4, 8, 12, 16]


def test_cached_future_dataset_accepts_real_wan_cache_row_source_offsets(tmp_path, monkeypatch) -> None:
    class FakeWanLoraGenerator:
        def generate_future_stack(self, current_images, *, task_text, output_dir, image_size, num_future_frames, seed):
            del current_images
            output_dir.mkdir(parents=True, exist_ok=True)
            video_path = output_dir / "wan_lora_view0.mp4"
            video_path.write_bytes(b"fake mp4")
            return Wan22Result(
                prompt=task_text,
                seed=seed,
                input_image_path=output_dir / "wan_lora_view0_input.png",
                video_path=video_path,
                future_images=torch.zeros(num_future_frames, 1, 3, image_size, image_size),
                selected_frame_indices=tuple(range(1, num_future_frames + 1)),
                total_video_frames=num_future_frames + 1,
            )

    monkeypatch.setattr("cache_future_rollouts.build_wan_lora_generator", lambda args: FakeWanLoraGenerator())
    cache_main(
        CacheArgs(
            future_source="wan_lora",
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            max_samples=1,
            synthetic_samples=1,
            image_size=32,
            frame_delta=4,
            num_future_frames=2,
            action_horizon=8,
            diffsynth_repo_dir="/tmp/fake-diffsynth",
            wan_lora_checkpoint_dir="/tmp/fake-wan",
            wan_lora_path="/tmp/fake-lora.safetensors",
        )
    )
    config = DatasetConfig(
        source="synthetic",
        image_keys=("corner4.image",),
        image_size=32,
        frame_delta=4,
        synthetic_samples=1,
        num_future_frames=2,
        action_horizon=8,
    )

    cached_dataset = CachedFutureDataset(SyntheticMetaWorldFramePairDataset(config), tmp_path)

    assert len(cached_dataset) == 1
    assert cached_dataset[0]["future_images"].shape == (2, 1, 3, 32, 32)


def test_cached_future_dataset_rejects_wan_cache_row_source_offset_mismatch(tmp_path, monkeypatch) -> None:
    class FakeWanLoraGenerator:
        def generate_future_stack(self, current_images, *, task_text, output_dir, image_size, num_future_frames, seed):
            del current_images
            output_dir.mkdir(parents=True, exist_ok=True)
            video_path = output_dir / "wan_lora_view0.mp4"
            video_path.write_bytes(b"fake mp4")
            return Wan22Result(
                prompt=task_text,
                seed=seed,
                input_image_path=output_dir / "wan_lora_view0_input.png",
                video_path=video_path,
                future_images=torch.zeros(num_future_frames, 1, 3, image_size, image_size),
                selected_frame_indices=tuple(range(1, num_future_frames + 1)),
                total_video_frames=num_future_frames + 1,
            )

    monkeypatch.setattr("cache_future_rollouts.build_wan_lora_generator", lambda args: FakeWanLoraGenerator())
    cache_main(
        CacheArgs(
            future_source="wan_lora",
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            max_samples=1,
            synthetic_samples=1,
            image_size=32,
            frame_delta=4,
            num_future_frames=2,
            action_horizon=8,
            diffsynth_repo_dir="/tmp/fake-diffsynth",
            wan_lora_checkpoint_dir="/tmp/fake-wan",
            wan_lora_path="/tmp/fake-lora.safetensors",
        )
    )
    rows = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines() if line.strip()]
    rows[0]["source_frame_offsets"] = [1, 2]
    (tmp_path / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    config = DatasetConfig(
        source="synthetic",
        image_keys=("corner4.image",),
        image_size=32,
        frame_delta=4,
        synthetic_samples=1,
        num_future_frames=2,
        action_horizon=8,
    )

    with pytest.raises(ValueError, match="manifest source-frame temporal metadata"):
        CachedFutureDataset(SyntheticMetaWorldFramePairDataset(config), tmp_path)


def test_cache_future_rollouts_rejects_wan_lora_tensor_without_temporal_metadata(tmp_path, monkeypatch) -> None:
    class FakeWanLoraGenerator:
        def generate_future_stack(self, *args, **kwargs):
            raise AssertionError("existing tensor should not trigger generation")

    monkeypatch.setattr("cache_future_rollouts.build_wan_lora_generator", lambda args: FakeWanLoraGenerator())
    futures_dir = tmp_path / "futures"
    futures_dir.mkdir()
    torch.save(torch.zeros(4, 1, 3, 32, 32), futures_dir / "sample_000000.pt")

    with pytest.raises(ValueError, match="without selected generated-video frame indices"):
        cache_main(
            CacheArgs(
                future_source="wan_lora",
                dataset_source="synthetic",
                output_dir=str(tmp_path),
                max_samples=1,
                synthetic_samples=1,
                image_size=32,
                num_future_frames=4,
                action_horizon=8,
                diffsynth_repo_dir="/tmp/fake-diffsynth",
                wan_lora_checkpoint_dir="/tmp/fake-wan",
                wan_lora_path="/tmp/fake-lora.safetensors",
            )
        )


def test_evaluate_future_cache_scores_dataset_future_cache(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "quality"
    cache_main(
        CacheArgs(
            future_source="dataset_future",
            dataset_source="synthetic",
            output_dir=str(cache_dir),
            max_samples=3,
            synthetic_samples=3,
            image_size=32,
            num_future_frames=4,
            action_horizon=8,
        )
    )
    evaluate_cache_main(
        EvaluateCacheArgs(
            cache_dir=str(cache_dir),
            dataset_source="synthetic",
            output_dir=str(output_dir),
            max_samples=3,
            synthetic_samples=3,
            image_size=32,
            num_future_frames=4,
            action_horizon=8,
            visual_samples=2,
            visual_tile_size=40,
        )
    )

    metrics = json.loads((output_dir / "future_cache_metrics.json").read_text())

    assert metrics["num_samples"] == 3
    assert metrics["future_mse"] == 0.0
    assert metrics["future_mae"] == 0.0
    assert metrics["future_psnr"] == 99.0
    assert (output_dir / "per_sample_metrics.jsonl").exists()
    contact_sheet = Image.open(output_dir / "future_cache_contact_sheet.png")
    assert contact_sheet.width > 0
    assert contact_sheet.height > 0


def test_evaluate_future_cache_records_wan_decoded_video_proof(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "quality"
    cache_main(
        CacheArgs(
            future_source="dataset_future",
            dataset_source="synthetic",
            output_dir=str(cache_dir),
            max_samples=2,
            synthetic_samples=2,
            image_size=32,
            num_future_frames=4,
            action_horizon=8,
        )
    )
    mark_cache_as_wan_with_selected_indices(cache_dir, [1, 2, 3, 4])

    evaluate_cache_main(
        EvaluateCacheArgs(
            cache_dir=str(cache_dir),
            dataset_source="synthetic",
            output_dir=str(output_dir),
            max_samples=2,
            synthetic_samples=2,
            image_size=32,
            num_future_frames=4,
            action_horizon=8,
            visual_samples=0,
        )
    )

    metrics = json.loads((output_dir / "future_cache_metrics.json").read_text())
    per_sample_rows = [
        json.loads(line) for line in (output_dir / "per_sample_metrics.jsonl").read_text().splitlines() if line.strip()
    ]
    summary = metrics["wan_decode_validation"]
    proof = per_sample_rows[0]["wan_decode_proof"]

    assert summary["num_samples"] == 2
    assert summary["all_samples_passed"] is True
    assert summary["decoded_video_frame_counts"] == [5]
    assert summary["selected_frame_indices"] == [[1, 2, 3, 4]]
    assert 0.0 <= summary["max_conditioning_frame_mae"] < 0.25
    assert proof["future_tensor_shape"] == [4, 1, 3, 32, 32]
    assert proof["future_shape_matches_manifest"] is True
    assert proof["selected_frame_indices"] == [1, 2, 3, 4]
    assert proof["selected_frame_count_matches_future"] is True
    assert proof["decoded_video_frame_count"] == 5
    assert proof["video_frame_count_matches_manifest"] is True
    assert proof["selected_indices_within_video"] is True
    assert 0.0 <= proof["conditioning_frame_mae"] < 0.25
    assert proof["issues"] == []
    assert proof["passed"] is True


def test_evaluate_future_cache_skips_samples_without_valid_future_frames(tmp_path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "quality"
    cache_main(
        CacheArgs(
            future_source="dataset_future",
            dataset_source="synthetic",
            output_dir=str(cache_dir),
            max_samples=3,
            synthetic_samples=3,
            image_size=32,
            num_future_frames=4,
            action_horizon=8,
        )
    )

    class BoundaryMaskedDataset(SyntheticMetaWorldFramePairDataset):
        def __getitem__(self, index):
            item = super().__getitem__(index)
            mask = torch.zeros(self.config.num_future_frames, dtype=torch.float32)
            if int(index) == 1:
                mask[0] = 1.0
            item["future_image_mask"] = mask
            return item

    monkeypatch.setattr("evaluate_future_cache.create_dataset", lambda config: BoundaryMaskedDataset(config))

    evaluate_cache_main(
        EvaluateCacheArgs(
            cache_dir=str(cache_dir),
            dataset_source="synthetic",
            output_dir=str(output_dir),
            max_samples=3,
            synthetic_samples=3,
            image_size=32,
            num_future_frames=4,
            action_horizon=8,
            visual_samples=0,
        )
    )

    metrics = json.loads((output_dir / "future_cache_metrics.json").read_text())
    per_sample_rows = [
        json.loads(line) for line in (output_dir / "per_sample_metrics.jsonl").read_text().splitlines() if line.strip()
    ]

    assert metrics["num_samples"] == 1
    assert metrics["num_total_samples"] == 3
    assert metrics["num_skipped_samples"] == 2
    assert metrics["skipped_samples"] == [
        {
            "cache_index": 0,
            "dataset_index": 0,
            "source": "dataset_future",
            "reason": "no_valid_future_frames",
        },
        {
            "cache_index": 2,
            "dataset_index": 2,
            "source": "dataset_future",
            "reason": "no_valid_future_frames",
        },
    ]
    assert len(per_sample_rows) == 1
    assert per_sample_rows[0]["cache_index"] == 1
    assert per_sample_rows[0]["dataset_index"] == 1
    assert per_sample_rows[0]["valid_future_frames"] == 1


def test_evaluate_future_cache_fails_when_all_samples_without_valid_future_frames(tmp_path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "quality"
    cache_main(
        CacheArgs(
            future_source="dataset_future",
            dataset_source="synthetic",
            output_dir=str(cache_dir),
            max_samples=2,
            synthetic_samples=2,
            image_size=32,
            num_future_frames=4,
            action_horizon=8,
        )
    )

    class AllBoundaryMaskedDataset(SyntheticMetaWorldFramePairDataset):
        def __getitem__(self, index):
            item = super().__getitem__(index)
            item["future_image_mask"] = torch.zeros(self.config.num_future_frames, dtype=torch.float32)
            return item

    monkeypatch.setattr("evaluate_future_cache.create_dataset", lambda config: AllBoundaryMaskedDataset(config))

    with pytest.raises(ValueError, match="All 2 cached future samples were skipped"):
        evaluate_cache_main(
            EvaluateCacheArgs(
                cache_dir=str(cache_dir),
                dataset_source="synthetic",
                output_dir=str(output_dir),
                max_samples=2,
                synthetic_samples=2,
                image_size=32,
                num_future_frames=4,
                action_horizon=8,
                visual_samples=0,
            )
        )


def test_evaluate_future_cache_accepts_explicit_visual_indices(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "quality"
    cache_main(
        CacheArgs(
            future_source="dataset_future",
            dataset_source="synthetic",
            output_dir=str(cache_dir),
            max_samples=3,
            synthetic_samples=3,
            image_size=32,
            num_future_frames=4,
            action_horizon=8,
        )
    )

    evaluate_cache_main(
        EvaluateCacheArgs(
            cache_dir=str(cache_dir),
            dataset_source="synthetic",
            output_dir=str(output_dir),
            max_samples=3,
            synthetic_samples=3,
            image_size=32,
            num_future_frames=4,
            action_horizon=8,
            visual_indices=(2, 0),
            visual_tile_size=40,
        )
    )

    metrics = json.loads((output_dir / "future_cache_metrics.json").read_text())

    assert metrics["visual_indices"] == [2, 0]
    assert (output_dir / "future_cache_contact_sheet.png").exists()


def test_evaluate_future_cache_detects_bad_cached_future(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "quality"
    cache_main(
        CacheArgs(
            future_source="dataset_future",
            dataset_source="synthetic",
            output_dir=str(cache_dir),
            max_samples=3,
            synthetic_samples=3,
            image_size=32,
            num_future_frames=4,
            action_horizon=8,
        )
    )
    first_row = json.loads((cache_dir / "manifest.jsonl").read_text().splitlines()[0])
    future_path = cache_dir / first_row["future_tensor"]
    future = torch.load(future_path, map_location="cpu", weights_only=False)
    torch.save(torch.zeros_like(future), future_path)

    evaluate_cache_main(
        EvaluateCacheArgs(
            cache_dir=str(cache_dir),
            dataset_source="synthetic",
            output_dir=str(output_dir),
            max_samples=3,
            synthetic_samples=3,
            image_size=32,
            num_future_frames=4,
            action_horizon=8,
            visual_samples=0,
        )
    )

    metrics = json.loads((output_dir / "future_cache_metrics.json").read_text())

    assert metrics["future_mse"] > 0.0
    assert metrics["future_mae"] > 0.0


def test_cached_future_dataset_replaces_future_images(tmp_path) -> None:
    config = DatasetConfig(
        source="synthetic",
        image_keys=("corner4.image",),
        image_size=32,
        frame_delta=1,
        synthetic_samples=3,
        num_future_frames=4,
        action_horizon=8,
    )
    cache_main(
        CacheArgs(
            future_source="dataset_future",
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            max_samples=3,
            synthetic_samples=3,
            image_size=32,
            num_future_frames=4,
            action_horizon=8,
        )
    )
    base_dataset = SyntheticMetaWorldFramePairDataset(config)
    cached_dataset = CachedFutureDataset(base_dataset, tmp_path)

    item = cached_dataset[0]

    assert item["future_images"].shape == (4, 1, 3, 32, 32)
    assert torch.allclose(item["future_image_mask"], torch.ones(4))


def test_cached_future_dataset_rejects_samples_per_episode_config_mismatch(tmp_path) -> None:
    base_config = DatasetConfig(
        source="lerobot",
        image_keys=("corner4.image",),
        image_size=32,
        frame_delta=1,
        num_future_frames=4,
        action_horizon=8,
        episodes=(0, 1),
        samples_per_episode=2,
    )
    cache_config = dataclasses.asdict(dataclasses.replace(base_config, samples_per_episode=1))
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps({"source": "dataset_future", "dataset_index": 0, "future_tensor": "futures/sample_000000.pt"}) + "\n"
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"future_source": "dataset_future", "dataset_config": cache_config}) + "\n"
    )

    class FakeBaseDataset:
        config = base_config

    with pytest.raises(ValueError, match="samples_per_episode"):
        CachedFutureDataset(FakeBaseDataset(), tmp_path)


def test_mixed_future_dataset_includes_gt_and_cached_futures(tmp_path) -> None:
    config = DatasetConfig(
        source="synthetic",
        image_keys=("corner4.image",),
        image_size=32,
        frame_delta=1,
        synthetic_samples=3,
        num_future_frames=4,
        action_horizon=8,
    )
    cache_main(
        CacheArgs(
            future_source="dataset_future",
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            max_samples=3,
            synthetic_samples=3,
            image_size=32,
            num_future_frames=4,
            action_horizon=8,
        )
    )
    rows = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    first_future_path = tmp_path / rows[0]["future_tensor"]
    cached_marker = torch.zeros((4, 1, 3, 32, 32))
    torch.save(cached_marker, first_future_path)

    base_dataset = SyntheticMetaWorldFramePairDataset(config)
    mixed_dataset = MixedFutureDataset(base_dataset, tmp_path)

    assert len(mixed_dataset) == 2 * len(base_dataset)
    assert not torch.allclose(mixed_dataset[0]["future_images"], cached_marker)
    assert torch.allclose(mixed_dataset[len(base_dataset)]["future_images"], cached_marker)


def test_default_optional_cache_still_replaces_futures(tmp_path) -> None:
    config = DatasetConfig(
        source="synthetic",
        image_keys=("corner4.image",),
        image_size=32,
        frame_delta=1,
        synthetic_samples=3,
        num_future_frames=4,
        action_horizon=8,
    )
    cache_main(
        CacheArgs(
            future_source="dataset_future",
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            max_samples=3,
            synthetic_samples=3,
            image_size=32,
            num_future_frames=4,
            action_horizon=8,
        )
    )
    rows = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    cached_marker = torch.zeros((4, 1, 3, 32, 32))
    torch.save(cached_marker, tmp_path / rows[0]["future_tensor"])

    dataset = create_dataset_with_optional_cache(config, tmp_path)

    assert len(dataset) == 3
    assert torch.allclose(dataset[0]["future_images"], cached_marker)


def test_cached_future_dataset_rejects_misaligned_wan_selected_frame_indices(tmp_path) -> None:
    config = DatasetConfig(
        source="synthetic",
        image_keys=("corner4.image",),
        image_size=32,
        frame_delta=1,
        synthetic_samples=3,
        num_future_frames=4,
        action_horizon=8,
    )
    cache_main(
        CacheArgs(
            future_source="dataset_future",
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            max_samples=3,
            synthetic_samples=3,
            image_size=32,
            frame_delta=1,
            num_future_frames=4,
            action_horizon=8,
        )
    )
    mark_cache_as_wan_with_selected_indices(tmp_path, [1, 6, 11, 16], future_frame_strategy="first")
    base_dataset = SyntheticMetaWorldFramePairDataset(config)

    with pytest.raises(ValueError, match=r"generated-video frame contract \[1, 2, 3, 4\]"):
        CachedFutureDataset(base_dataset, tmp_path)
    with pytest.raises(ValueError, match=r"generated-video frame contract \[1, 2, 3, 4\]"):
        create_dataset_with_optional_cache(config, tmp_path)


def test_cached_future_dataset_rejects_wan_manifest_missing_selected_frame_indices(tmp_path) -> None:
    config = DatasetConfig(
        source="synthetic",
        image_keys=("corner4.image",),
        image_size=32,
        frame_delta=1,
        synthetic_samples=3,
        num_future_frames=4,
        action_horizon=8,
    )
    cache_main(
        CacheArgs(
            future_source="dataset_future",
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            max_samples=3,
            synthetic_samples=3,
            image_size=32,
            frame_delta=1,
            num_future_frames=4,
            action_horizon=8,
        )
    )
    mark_cache_as_wan_with_selected_indices(tmp_path, [1, 2, 3, 4], future_frame_strategy="first")
    rows = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines() if line.strip()]
    rows[0]["selected_frame_indices"] = None
    (tmp_path / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(ValueError, match="manifest is missing selected generated-video frame indices"):
        CachedFutureDataset(SyntheticMetaWorldFramePairDataset(config), tmp_path)


def test_cached_future_dataset_rejects_wan_cache_without_temporal_metadata(tmp_path) -> None:
    config = DatasetConfig(
        source="synthetic",
        image_keys=("corner4.image",),
        image_size=32,
        frame_delta=1,
        synthetic_samples=3,
        num_future_frames=4,
        action_horizon=8,
    )
    cache_main(
        CacheArgs(
            future_source="dataset_future",
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            max_samples=3,
            synthetic_samples=3,
            image_size=32,
            frame_delta=1,
            num_future_frames=4,
            action_horizon=8,
        )
    )
    config_path = tmp_path / "config.json"
    cache_config = json.loads(config_path.read_text())
    cache_config["future_source"] = "wan_lora"
    cache_config.pop("future_frame_selection", None)
    config_path.write_text(json.dumps(cache_config, indent=2) + "\n")
    rows = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines() if line.strip()]
    for row in rows:
        row["source"] = "wan_lora"
        row.pop("selected_frame_indices", None)
        row.pop("source_frame_offsets", None)
        row.pop("dataset_frame_delta", None)
    (tmp_path / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(ValueError, match="neither config.json nor manifest.jsonl records"):
        CachedFutureDataset(SyntheticMetaWorldFramePairDataset(config), tmp_path)


def test_cached_future_dataset_without_base_config_rejects_unverifiable_wan_temporal_metadata(tmp_path) -> None:
    cache_main(
        CacheArgs(
            future_source="dataset_future",
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            max_samples=1,
            synthetic_samples=1,
            image_size=32,
            frame_delta=1,
            num_future_frames=4,
            action_horizon=8,
        )
    )
    mark_cache_as_wan_with_selected_indices(tmp_path, [1, 2, 3, 4], future_frame_strategy="first")
    config_path = tmp_path / "config.json"
    cache_config = json.loads(config_path.read_text())
    cache_config["dataset_config"].pop("frame_delta")
    config_path.write_text(json.dumps(cache_config, indent=2) + "\n")

    class NoConfigDataset:
        pass

    with pytest.raises(ValueError, match="missing dataset_config.frame_delta"):
        CachedFutureDataset(NoConfigDataset(), tmp_path)


def test_cached_future_dataset_accepts_aligned_wan_first4_selected_frame_indices(tmp_path) -> None:
    config = DatasetConfig(
        source="synthetic",
        image_keys=("corner4.image",),
        image_size=32,
        frame_delta=1,
        synthetic_samples=3,
        num_future_frames=4,
        action_horizon=8,
    )
    cache_main(
        CacheArgs(
            future_source="dataset_future",
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            max_samples=3,
            synthetic_samples=3,
            image_size=32,
            frame_delta=1,
            num_future_frames=4,
            action_horizon=8,
        )
    )
    mark_cache_as_wan_with_selected_indices(tmp_path, [1, 2, 3, 4], future_frame_strategy="first")

    cached_dataset = CachedFutureDataset(SyntheticMetaWorldFramePairDataset(config), tmp_path)
    optional_cached_dataset = create_dataset_with_optional_cache(config, tmp_path)

    assert len(cached_dataset) == 3
    assert len(optional_cached_dataset) == 3
    assert cached_dataset[0]["future_images"].shape == (4, 1, 3, 32, 32)
    assert optional_cached_dataset[0]["future_images"].shape == (4, 1, 3, 32, 32)


def test_cached_future_dataset_accepts_wan_source_offsets_selected_frame_indices(tmp_path) -> None:
    config = DatasetConfig(
        source="synthetic",
        image_keys=("corner4.image",),
        image_size=32,
        frame_delta=4,
        synthetic_samples=3,
        num_future_frames=4,
        action_horizon=20,
    )
    cache_main(
        CacheArgs(
            future_source="dataset_future",
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            max_samples=3,
            synthetic_samples=3,
            image_size=32,
            frame_delta=4,
            num_future_frames=4,
            action_horizon=20,
        )
    )
    mark_cache_as_wan_with_selected_indices(tmp_path, [4, 8, 12, 16], future_frame_strategy="source_offsets")

    cached_dataset = CachedFutureDataset(SyntheticMetaWorldFramePairDataset(config), tmp_path)

    assert len(cached_dataset) == 3
    assert cached_dataset[0]["future_images"].shape == (4, 1, 3, 32, 32)


def test_cached_future_dataset_rejects_source_offsets_indices_when_strategy_is_first(tmp_path) -> None:
    config = DatasetConfig(
        source="synthetic",
        image_keys=("corner4.image",),
        image_size=32,
        frame_delta=4,
        synthetic_samples=3,
        num_future_frames=4,
        action_horizon=20,
    )
    cache_main(
        CacheArgs(
            future_source="dataset_future",
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            max_samples=3,
            synthetic_samples=3,
            image_size=32,
            frame_delta=4,
            num_future_frames=4,
            action_horizon=20,
        )
    )
    mark_cache_as_wan_with_selected_indices(tmp_path, [4, 8, 12, 16], future_frame_strategy="first")

    with pytest.raises(ValueError, match=r"generated-video frame contract \[1, 2, 3, 4\]"):
        CachedFutureDataset(SyntheticMetaWorldFramePairDataset(config), tmp_path)


def test_cached_future_dataset_preserves_base_future_mask(tmp_path) -> None:
    config = DatasetConfig(
        source="synthetic",
        image_keys=("corner4.image",),
        image_size=32,
        frame_delta=1,
        synthetic_samples=3,
        num_future_frames=4,
        action_horizon=8,
    )
    cache_main(
        CacheArgs(
            future_source="dataset_future",
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            max_samples=3,
            synthetic_samples=3,
            image_size=32,
            num_future_frames=4,
            action_horizon=8,
        )
    )
    base_dataset = SyntheticMetaWorldFramePairDataset(config)
    original_getitem = base_dataset.__class__.__getitem__

    class MaskedDataset(SyntheticMetaWorldFramePairDataset):
        def __getitem__(self, index):
            item = original_getitem(self, index)
            item["future_image_mask"] = torch.tensor([1.0, 1.0, 0.0, 0.0])
            return item

    cached_dataset = CachedFutureDataset(MaskedDataset(config), tmp_path)
    item = cached_dataset[0]

    assert torch.allclose(item["future_image_mask"], torch.tensor([1.0, 1.0, 0.0, 0.0]))


def test_cached_future_dataset_rejects_config_mismatch(tmp_path) -> None:
    config = DatasetConfig(
        source="synthetic",
        image_keys=("corner4.image",),
        image_size=32,
        synthetic_samples=3,
        num_future_frames=4,
        action_horizon=8,
        seed=999,
    )
    cache_main(
        CacheArgs(
            future_source="dataset_future",
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            max_samples=3,
            synthetic_samples=3,
            image_size=32,
            num_future_frames=4,
            action_horizon=8,
            seed=7,
        )
    )

    with pytest.raises(ValueError, match="dataset_config"):
        CachedFutureDataset(SyntheticMetaWorldFramePairDataset(config), tmp_path)


def test_cached_future_dataset_rejects_missing_config(tmp_path) -> None:
    config = DatasetConfig(
        source="synthetic",
        image_keys=("corner4.image",),
        image_size=32,
        synthetic_samples=3,
        num_future_frames=4,
        action_horizon=8,
    )
    cache_main(
        CacheArgs(
            future_source="dataset_future",
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            max_samples=3,
            synthetic_samples=3,
            image_size=32,
            num_future_frames=4,
            action_horizon=8,
        )
    )
    (tmp_path / "config.json").unlink()

    with pytest.raises(FileNotFoundError, match="Cached future config not found"):
        CachedFutureDataset(SyntheticMetaWorldFramePairDataset(config), tmp_path)


def test_cached_future_dataset_rejects_path_escape(tmp_path) -> None:
    cache_main(
        CacheArgs(
            future_source="dataset_future",
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            max_samples=3,
            synthetic_samples=3,
            image_size=32,
            num_future_frames=4,
            action_horizon=8,
        )
    )
    rows = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    rows[0]["future_tensor"] = "../escape.pt"
    (tmp_path / "manifest.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    base_dataset = SyntheticMetaWorldFramePairDataset(
        DatasetConfig(
            source="synthetic",
            image_keys=("corner4.image",),
            image_size=32,
            frame_delta=1,
            synthetic_samples=3,
            num_future_frames=4,
            action_horizon=8,
        )
    )
    cached_dataset = CachedFutureDataset(base_dataset, tmp_path)

    with pytest.raises(ValueError, match="escapes"):
        cached_dataset[0]


def test_eval_pipeline_can_use_cached_future_dir(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "eval"
    cache_main(
        CacheArgs(
            future_source="dataset_future",
            dataset_source="synthetic",
            output_dir=str(cache_dir),
            max_samples=8,
            synthetic_samples=8,
            image_size=32,
            frame_delta=1,
            num_future_frames=4,
            action_horizon=8,
        )
    )
    eval_main(
        EvalArgs(
            future_source="cached",
            cached_future_dir=str(cache_dir),
            dataset_source="synthetic",
            output_dir=str(output_dir),
            max_samples=8,
            synthetic_samples=8,
            image_size=32,
            num_future_frames=4,
            action_horizon=8,
            batch_size=4,
            device="cpu",
        )
    )

    output = json.loads((output_dir / "pipeline_eval.json").read_text())

    assert output["future_source"] == "cached"
    assert output["metrics"]["idm_mse"] >= 0.0


def test_eval_idm_can_use_cached_future_dir(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    train_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval"
    cache_main(
        CacheArgs(
            future_source="dataset_future",
            dataset_source="synthetic",
            output_dir=str(cache_dir),
            max_samples=16,
            synthetic_samples=16,
            image_size=32,
            num_future_frames=4,
            action_horizon=8,
        )
    )
    run_idm_training(
        TrainConfig(
            dataset=DatasetConfig(
                source="synthetic",
                image_keys=("corner4.image",),
                image_size=32,
                frame_delta=1,
                max_samples=16,
                synthetic_samples=16,
                num_future_frames=4,
                action_horizon=8,
            ),
            output_dir=str(train_dir),
            epochs=1,
            batch_size=4,
            device="cpu",
            seed=11,
        )
    )
    eval_idm_main(
        EvalIdmArgs(
            checkpoint=str(train_dir / "idm_checkpoint.pt"),
            cached_future_dir=str(cache_dir),
            dataset_source="synthetic",
            output_dir=str(eval_dir),
            image_keys=("corner4.image",),
            max_samples=16,
            synthetic_samples=16,
            image_size=32,
            frame_delta=1,
            num_future_frames=4,
            action_horizon=8,
            batch_size=4,
            device="cpu",
        )
    )

    output = json.loads((eval_dir / "eval_metrics.json").read_text())

    assert output["cached_future_dir"] == str(cache_dir)
    assert output["idm_mse"] >= 0.0


def test_build_wan_generators_propagate_conditioning_frame_contract() -> None:
    from cache_future_rollouts import build_wan_generator, build_wan_lora_generator

    args = CacheArgs(
        future_source="wan2_2",
        wan_repo_dir="/tmp/fake-wan-repo",
        wan_checkpoint_dir="/tmp/fake-wan-checkpoint",
        diffsynth_repo_dir="/tmp/fake-diffsynth",
        wan_lora_checkpoint_dir="/tmp/fake-wan",
        wan_lora_path="/tmp/fake-lora.safetensors",
        wan_verify_conditioning_frame=False,
        wan_conditioning_frame_max_mae=0.05,
        generation_seed=123,
    )

    wan_generator = build_wan_generator(args)
    lora_generator = build_wan_lora_generator(args)

    assert wan_generator.verify_conditioning_frame is False
    assert wan_generator.conditioning_frame_max_mae == 0.05
    assert wan_generator.config.base_seed == 123
    assert lora_generator.config.verify_conditioning_frame is False
    assert lora_generator.config.conditioning_frame_max_mae == 0.05
    assert lora_generator.config.base_seed == 123


def test_cache_future_rollouts_rejects_raw_wan_frame_delta_above_one(tmp_path) -> None:
    with pytest.raises(ValueError, match="Raw Wan2.2.*frame_delta=1"):
        cache_main(
            CacheArgs(
                future_source="wan2_2",
                dataset_source="synthetic",
                output_dir=str(tmp_path),
                max_samples=1,
                synthetic_samples=1,
                frame_delta=4,
                action_horizon=16,
                wan_repo_dir="/tmp/fake-wan-repo",
                wan_checkpoint_dir="/tmp/fake-wan-checkpoint",
            )
        )


def test_cache_future_rollouts_requires_wan_paths_for_wan_source(tmp_path) -> None:
    with pytest.raises(ValueError, match="wan-repo-dir"):
        cache_main(CacheArgs(future_source="wan2_2", output_dir=str(tmp_path)))


def test_cache_future_rollouts_requires_wan_lora_paths(tmp_path) -> None:
    with pytest.raises(ValueError, match="diffsynth-repo-dir"):
        cache_main(CacheArgs(future_source="wan_lora", output_dir=str(tmp_path)))
