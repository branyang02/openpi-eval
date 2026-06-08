from __future__ import annotations

import json

import imageio.v3 as iio

from dry_run_pipeline import Args, main


def test_dry_run_pipeline_writes_visuals_and_action_chunk(tmp_path) -> None:
    main(
        Args(
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            image_size=32,
            num_future_frames=4,
            action_horizon=8,
            synthetic_samples=8,
            device="cpu",
        )
    )

    output = json.loads((tmp_path / "dry_run.json").read_text())

    assert output["future_source"] == "dataset_future"
    assert output["current_images_shape"] == [1, 3, 32, 32]
    assert output["future_images_shape"] == [4, 1, 3, 32, 32]
    assert output["action_chunk_shape"] == [8, 4]
    assert (tmp_path / "current_frame.png").exists()
    assert (tmp_path / "wan_like_future.mp4").exists()
    assert len(list(iio.imiter(tmp_path / "wan_like_future.mp4"))) == 5
