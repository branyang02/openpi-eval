from __future__ import annotations

import json

import pytest
import torch

from interpolate_wan_prefix_caches import Args, _validate_alpha, interpolate_prefix_caches
from world_model.pi05_wan_action_expert import CachedWanPrefixActionDataset


def _write_json(path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_prefix_cache(
    cache_dir,
    *,
    prefix_offset: float = 0.0,
    task: str = "pick",
    cache_kind: str = "pi05_wan_current_prefix",
    contains_future_ground_truth_latents: bool = False,
    contains_future_latents: bool = False,
    include_strong_alignment_keys: bool = True,
) -> None:
    cache_dir.mkdir()
    _write_json(
        cache_dir / "config.json",
        {
            "cache_kind": cache_kind,
            "num_samples": 2,
            "contains_future_ground_truth_latents": contains_future_ground_truth_latents,
            "contains_future_latents": contains_future_latents,
            "dataset_config": {"source": "synthetic", "samples_per_episode": 2},
        },
    )
    manifest_rows = []
    for index in range(2):
        row_file = f"sample_{index:06d}.pt"
        prefix = torch.full((3, 4), float(index), dtype=torch.float32) + prefix_offset
        metadata = {
            "cache_kind": cache_kind,
            "episode_index": 2,
            "frame_index": index * 10,
            "task_index": 0,
            "task": task,
            "wan_action_mode": "partial_wan_prefix_action_expert",
            "contains_future_ground_truth_latents": contains_future_ground_truth_latents,
            "contains_future_latents": contains_future_latents,
        }
        manifest_row = {
            "episode_index": 2,
            "frame_index": index * 10,
            "task_index": 0,
            "row_file": row_file,
            "prefix_shape": [3, 4],
            "cache_kind": cache_kind,
            "wan_action_mode": "partial_wan_prefix_action_expert",
            "contains_future_ground_truth_latents": contains_future_ground_truth_latents,
            "contains_future_latents": contains_future_latents,
        }
        if include_strong_alignment_keys:
            metadata["dataset_index"] = index
            manifest_row["dataset_index"] = index
            manifest_row["source_dataset_index"] = 100 + index
        sample = {
            "prefix_tokens": prefix,
            "state": torch.full((4,), float(index), dtype=torch.float32),
            "actions": torch.full((2, 4), float(index + 1), dtype=torch.float32),
            "action_mask": torch.ones(2, dtype=torch.float32),
            "task": task,
            "metadata": metadata,
        }
        torch.save(sample, cache_dir / row_file)
        manifest_rows.append(manifest_row)
    _write_jsonl(cache_dir / "manifest.jsonl", manifest_rows)


def test_validate_alpha_accepts_unit_interval_and_rejects_invalid() -> None:
    assert _validate_alpha(0) == 0.0
    assert _validate_alpha(0.5) == 0.5
    assert _validate_alpha(1) == 1.0

    for bad in (-0.1, 1.1, True, "0.5"):
        with pytest.raises(ValueError, match="alpha must"):
            _validate_alpha(bad)


def test_interpolate_prefix_caches_writes_blended_cache(tmp_path) -> None:
    base_cache = tmp_path / "base"
    target_cache = tmp_path / "target"
    output_dir = tmp_path / "mixed"
    _write_prefix_cache(base_cache, prefix_offset=0.0)
    _write_prefix_cache(target_cache, prefix_offset=10.0)

    result = interpolate_prefix_caches(
        Args(
            base_cache_path=str(base_cache),
            target_cache_path=str(target_cache),
            output_dir=str(output_dir),
            alpha=0.25,
        )
    )

    assert result["num_samples"] == 2
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert config["cache_kind"] == "pi05_wan_prefix_interpolation"
    assert config["prefix_interpolation"]["alpha"] == 0.25
    manifest_rows = [
        json.loads(line) for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert manifest_rows[0]["source"] == "interpolated_wan_prefix_tokens"
    assert manifest_rows[0]["prefix_interpolation"]["alpha"] == 0.25

    dataset = CachedWanPrefixActionDataset(output_dir)
    row = dataset[1]
    assert torch.allclose(row["prefix_tokens"], torch.full((3, 4), 3.5))
    assert torch.allclose(row["state"], torch.full((4,), 1.0))
    assert torch.allclose(row["actions"], torch.full((2, 4), 2.0))
    assert row["task"] == "pick"
    assert row["metadata"]["prefix_interpolation"]["alpha"] == 0.25


def test_interpolate_prefix_caches_requires_overwrite_for_existing_output(tmp_path) -> None:
    base_cache = tmp_path / "base"
    target_cache = tmp_path / "target"
    output_dir = tmp_path / "mixed"
    output_dir.mkdir()
    _write_prefix_cache(base_cache)
    _write_prefix_cache(target_cache, prefix_offset=1.0)

    with pytest.raises(FileExistsError, match="Output directory already exists"):
        interpolate_prefix_caches(
            Args(
                base_cache_path=str(base_cache),
                target_cache_path=str(target_cache),
                output_dir=str(output_dir),
                alpha=0.5,
            )
        )


def test_interpolate_prefix_caches_rejects_unaligned_rows(tmp_path) -> None:
    base_cache = tmp_path / "base"
    target_cache = tmp_path / "target"
    output_dir = tmp_path / "mixed"
    _write_prefix_cache(base_cache)
    _write_prefix_cache(target_cache, prefix_offset=1.0)
    manifest_rows = [
        json.loads(line) for line in (target_cache / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    manifest_rows[1]["episode_index"] = 99
    _write_jsonl(target_cache / "manifest.jsonl", manifest_rows)

    with pytest.raises(ValueError, match="not aligned on episode_index"):
        interpolate_prefix_caches(
            Args(
                base_cache_path=str(base_cache),
                target_cache_path=str(target_cache),
                output_dir=str(output_dir),
                alpha=0.5,
            )
        )
    assert not output_dir.exists()


def test_interpolate_half_alpha_preserves_ground_truth_from_contributing_base(tmp_path) -> None:
    base_cache = tmp_path / "ground_truth"
    target_cache = tmp_path / "generated"
    output_dir = tmp_path / "mixed"
    _write_prefix_cache(
        base_cache,
        contains_future_ground_truth_latents=True,
        contains_future_latents=True,
    )
    _write_prefix_cache(
        target_cache,
        prefix_offset=10.0,
        contains_future_ground_truth_latents=False,
        contains_future_latents=True,
    )

    interpolate_prefix_caches(
        Args(
            base_cache_path=str(base_cache),
            target_cache_path=str(target_cache),
            output_dir=str(output_dir),
            alpha=0.5,
        )
    )

    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert config["contains_future_ground_truth_latents"] is True
    assert config["contains_future_latents"] is True

    manifest_rows = [
        json.loads(line) for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert manifest_rows
    assert all(row["contains_future_ground_truth_latents"] is True for row in manifest_rows)
    assert all(row["contains_future_latents"] is True for row in manifest_rows)

    sample = torch.load(output_dir / manifest_rows[0]["row_file"], weights_only=False)
    assert sample["metadata"]["contains_future_ground_truth_latents"] is True
    assert sample["metadata"]["contains_future_latents"] is True


def test_interpolate_generated_endpoint_drops_non_contributing_ground_truth(tmp_path) -> None:
    base_cache = tmp_path / "ground_truth"
    target_cache = tmp_path / "generated"
    output_dir = tmp_path / "mixed"
    _write_prefix_cache(
        base_cache,
        contains_future_ground_truth_latents=True,
        contains_future_latents=True,
    )
    _write_prefix_cache(
        target_cache,
        prefix_offset=10.0,
        contains_future_ground_truth_latents=False,
        contains_future_latents=False,
    )

    interpolate_prefix_caches(
        Args(
            base_cache_path=str(base_cache),
            target_cache_path=str(target_cache),
            output_dir=str(output_dir),
            alpha=1.0,
        )
    )

    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert config["contains_future_ground_truth_latents"] is False
    assert config["contains_future_latents"] is False

    manifest_rows = [
        json.loads(line) for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert manifest_rows
    assert all(row["contains_future_ground_truth_latents"] is False for row in manifest_rows)
    assert all(row["contains_future_latents"] is False for row in manifest_rows)

    sample = torch.load(output_dir / manifest_rows[0]["row_file"], weights_only=False)
    assert sample["metadata"]["contains_future_ground_truth_latents"] is False
    assert sample["metadata"]["contains_future_latents"] is False


def test_interpolate_sets_consistent_cache_kind_across_config_manifest_and_samples(tmp_path) -> None:
    base_cache = tmp_path / "base"
    target_cache = tmp_path / "target"
    output_dir = tmp_path / "mixed"
    _write_prefix_cache(base_cache)
    _write_prefix_cache(target_cache, prefix_offset=10.0)

    interpolate_prefix_caches(
        Args(
            base_cache_path=str(base_cache),
            target_cache_path=str(target_cache),
            output_dir=str(output_dir),
            alpha=0.5,
        )
    )

    expected_kind = "pi05_wan_prefix_interpolation"
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert config["cache_kind"] == expected_kind

    manifest_rows = [
        json.loads(line) for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert manifest_rows
    assert all(row["cache_kind"] == expected_kind for row in manifest_rows)

    for row in manifest_rows:
        sample = torch.load(output_dir / row["row_file"], weights_only=False)
        assert sample["metadata"]["cache_kind"] == expected_kind


def test_interpolate_rejects_missing_strong_alignment_metadata(tmp_path) -> None:
    base_cache = tmp_path / "base"
    target_cache = tmp_path / "target"
    output_dir = tmp_path / "mixed"
    _write_prefix_cache(base_cache, include_strong_alignment_keys=False)
    _write_prefix_cache(target_cache, prefix_offset=10.0, include_strong_alignment_keys=False)

    with pytest.raises(ValueError, match="strong alignment"):
        interpolate_prefix_caches(
            Args(
                base_cache_path=str(base_cache),
                target_cache_path=str(target_cache),
                output_dir=str(output_dir),
                alpha=0.5,
            )
        )
    assert not output_dir.exists()
