from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch

from diagnose_future_motion import Args, diagnose_dataset, main, motion_metrics_for_item
from world_model.config import DatasetConfig
from world_model.data import expected_wan_source_frame_offsets


def _image(value: float) -> torch.Tensor:
    return torch.full((1, 3, 1, 1), value, dtype=torch.float32)


def _future(values: list[float]) -> torch.Tensor:
    return torch.stack([_image(value) for value in values], dim=0)


def _sample(current: float, futures: list[float], mask: list[float] | None = None) -> dict[str, torch.Tensor]:
    return {
        "current_images": _image(current),
        "future_images": _future(futures),
        "future_image_mask": torch.tensor(mask if mask is not None else [1.0] * len(futures)),
        "dataset_index": torch.tensor(0),
    }


class _MotionDataset(torch.utils.data.Dataset):
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.items = [
            _sample(0.0, [1.0, 3.0]),
            {**_sample(10.0, [10.0, 14.0]), "dataset_index": torch.tensor(1)},
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.items[index]


def _write_wan_cache(cache_dir: Path, config: DatasetConfig) -> None:
    futures_dir = cache_dir / "futures"
    futures_dir.mkdir(parents=True)
    cached_futures = [
        _future([0.0, 0.0]),
        _future([10.0, 10.0]),
    ]
    rows = []
    selected_frame_indices = [1, 2]
    source_frame_offsets = expected_wan_source_frame_offsets(config.frame_delta, config.num_future_frames)
    for index, future in enumerate(cached_futures):
        path = futures_dir / f"sample_{index:06d}.pt"
        torch.save(future, path)
        rows.append(
            {
                "source": "wan_lora",
                "dataset_index": index,
                "future_tensor": f"futures/{path.name}",
                "future_shape": list(future.shape),
                "selected_frame_indices": selected_frame_indices,
                "dataset_frame_delta": config.frame_delta,
                "source_frame_offsets": source_frame_offsets,
            }
        )
    (cache_dir / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (cache_dir / "config.json").write_text(
        json.dumps(
            {
                "future_source": "wan_lora",
                "dataset_config": dataclasses.asdict(config),
                "num_samples": len(rows),
                "future_frame_selection": {
                    "future_frame_strategy": "first",
                    "selected_frame_indices": selected_frame_indices,
                    "selected_frame_indices_by_dataset_index": {
                        str(row["dataset_index"]): selected_frame_indices for row in rows
                    },
                    "dataset_frame_delta": config.frame_delta,
                    "frame_delta": config.frame_delta,
                    "source_frame_offsets": source_frame_offsets,
                    "num_future_frames": config.num_future_frames,
                },
            },
            indent=2,
        )
        + "\n"
    )


def test_motion_metrics_for_item_respects_future_mask() -> None:
    row = motion_metrics_for_item(
        _sample(0.0, [1.0, 3.0, 7.0], mask=[1.0, 1.0, 0.0]),
        source="gt",
        sample_index=0,
    )

    assert row["current_to_first_future_mae"] == pytest.approx(1.0)
    assert row["current_to_all_futures_mae"] == pytest.approx(2.0)
    assert row["adjacent_future_delta_mae"] == pytest.approx(2.0)
    assert row["valid_future_indices"] == [0, 1]
    assert row["adjacent_future_delta_num_elements"] == 3


def test_diagnose_dataset_aggregates_motion_metrics() -> None:
    config = DatasetConfig(
        source="synthetic",
        image_keys=("corner4.image",),
        image_size=1,
        frame_delta=2,
        num_future_frames=2,
        action_horizon=2,
        synthetic_samples=2,
        max_samples=2,
    )
    report = diagnose_dataset(_MotionDataset(config), source="gt")
    metrics = report["aggregates"]["metrics"]

    assert report["aggregates"]["num_samples"] == 2
    assert metrics["current_to_first_future"]["mae"] == pytest.approx(0.5)
    assert metrics["current_to_all_futures"]["mae"] == pytest.approx(2.0)
    assert metrics["adjacent_future_delta"]["mae"] == pytest.approx(3.0)


def test_main_writes_gt_and_cached_wan_motion_diagnostics(tmp_path, monkeypatch) -> None:
    config = DatasetConfig(
        source="synthetic",
        image_keys=("corner4.image",),
        image_size=1,
        frame_delta=2,
        num_future_frames=2,
        action_horizon=2,
        synthetic_samples=2,
        max_samples=2,
    )
    dataset = _MotionDataset(config)
    cache_dir = tmp_path / "cache"
    _write_wan_cache(cache_dir, config)
    monkeypatch.setattr("diagnose_future_motion.create_dataset", lambda _config: dataset)

    main(
        Args(
            dataset_source="synthetic",
            image_keys=("corner4.image",),
            output_dir=str(tmp_path / "out"),
            cached_future_dir=str(cache_dir),
            image_size=1,
            frame_delta=2,
            num_future_frames=2,
            action_horizon=2,
            synthetic_samples=2,
            max_samples=2,
            write_markdown=True,
        )
    )

    output = json.loads((tmp_path / "out" / "future_motion_metrics.json").read_text())
    markdown = (tmp_path / "out" / "future_motion_summary.md").read_text()

    assert output["gt"]["aggregates"]["metrics"]["current_to_first_future"]["mae"] == pytest.approx(0.5)
    assert output["cached"]["aggregates"]["metrics"]["current_to_all_futures"]["mae"] == pytest.approx(0.0)
    assert output["cached"]["aggregates"]["metrics"]["adjacent_future_delta"]["mae"] == pytest.approx(0.0)
    assert output["cached"]["per_sample"][0]["cache_row_source"] == "wan_lora"
    assert "| cached | 2 | 0.000000 | 0.000000 | 0.000000 |" in markdown
