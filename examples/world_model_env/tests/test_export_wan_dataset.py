from __future__ import annotations

import csv
import json

import imageio.v3 as iio
import pytest

from export_wan_dataset import Args, main


def test_export_wan_dataset_writes_diffsynth_metadata_and_videos(tmp_path) -> None:
    main(
        Args(
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            max_samples=3,
            image_size=32,
            num_future_frames=4,
            synthetic_samples=3,
        )
    )

    rows = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    metadata_rows = list(csv.DictReader((tmp_path / "metadata.csv").open()))

    assert len(rows) == 3
    assert len(metadata_rows) == 3
    assert set(metadata_rows[0]) == {"video", "prompt"}
    for row in rows:
        assert (tmp_path / row["image"]).exists()
        assert (tmp_path / row["video"]).exists()
        assert (tmp_path / row["caption"]).exists()
        assert row["num_frames"] == 5
        assert row["conditioning_frame"] == 0
        assert row["selected_frame_indices"] == [1, 2, 3, 4]
        assert row["dataset_frame_delta"] == 1
        assert row["source_frame_offsets"] == [1, 2, 3, 4]
        assert "Robot manipulation in MetaWorld" in row["prompt"]

        frames = list(iio.imiter(tmp_path / row["video"]))
        assert len(frames) == 5


def test_export_wan_dataset_frame_delta_four_records_source_offsets(tmp_path) -> None:
    main(
        Args(
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            max_samples=1,
            image_size=32,
            frame_delta=4,
            num_future_frames=4,
            synthetic_samples=1,
            action_horizon=16,
        )
    )

    row = json.loads((tmp_path / "manifest.jsonl").read_text().splitlines()[0])

    assert row["selected_frame_indices"] == [1, 2, 3, 4]
    assert row["dataset_frame_delta"] == 4
    assert row["source_frame_offsets"] == [4, 8, 12, 16]


def test_export_wan_dataset_requires_wan_frame_count(tmp_path) -> None:
    with pytest.raises(ValueError, match="4n\\+1"):
        main(
            Args(
                dataset_source="synthetic",
                output_dir=str(tmp_path),
                max_samples=1,
                num_future_frames=2,
                synthetic_samples=1,
            )
        )
