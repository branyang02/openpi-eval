from __future__ import annotations

import subprocess

import numpy as np
import pytest
import torch

from world_model.config import Wan22Config
from world_model.wan22 import Wan22FutureGenerator, select_future_frame_indices, verify_conditioning_frame


def _solid_frame(value: int, size: int = 16) -> np.ndarray:
    return np.full((size, size, 3), fill_value=value, dtype=np.uint8)


def _image_from_uint8(frame: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(frame).permute(2, 0, 1).to(torch.float32) / 255.0


def test_select_future_frame_indices_skips_conditioning_frame() -> None:
    assert select_future_frame_indices(total_frames=17, num_future_frames=4) == [1, 2, 3, 4]
    with pytest.raises(ValueError, match="skip the conditioning frame"):
        select_future_frame_indices(total_frames=2, num_future_frames=4)
    with pytest.raises(ValueError, match="future_frame_strategy must be one of"):
        select_future_frame_indices(total_frames=17, num_future_frames=4, strategy="linspace")


def test_select_future_frame_indices_source_offsets_uses_dataset_offsets() -> None:
    assert select_future_frame_indices(
        total_frames=17,
        num_future_frames=4,
        frame_delta=4,
        strategy="source_offsets",
    ) == [4, 8, 12, 16]
    with pytest.raises(ValueError, match="Need at least 17 frames"):
        select_future_frame_indices(
            total_frames=16,
            num_future_frames=4,
            frame_delta=4,
            strategy="source_offsets",
        )


def test_wan22_generator_builds_official_ti2v_command(tmp_path) -> None:
    repo_dir = tmp_path / "Wan2.2"
    checkpoint_dir = tmp_path / "Wan2.2-TI2V-5B"
    repo_dir.mkdir()
    checkpoint_dir.mkdir()
    (repo_dir / "generate.py").write_text("print('fake wan')\n")
    commands = []

    def fake_runner(command, **kwargs):
        commands.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    conditioning_frame = _solid_frame(96)

    def fake_reader(_video_path):
        # Wan emits the conditioning image as frame 0; later frames are the future.
        return [conditioning_frame] + [_solid_frame((index + 1) * 30) for index in range(4)]

    generator = Wan22FutureGenerator(
        Wan22Config(
            repo_dir=str(repo_dir),
            checkpoint_dir=str(checkpoint_dir),
            offload_model=True,
            convert_model_dtype=True,
            t5_cpu=True,
            base_seed=13,
            python_executable="python3",
        ),
        runner=fake_runner,
        video_reader=fake_reader,
    )

    result = generator.generate_future_stack(
        _image_from_uint8(conditioning_frame).unsqueeze(0),
        task_text="put the banana in the bowl",
        output_dir=tmp_path / "out",
        image_size=16,
        num_future_frames=2,
    )

    command, kwargs = commands[0]
    assert kwargs["cwd"] == str(repo_dir.resolve())
    assert command[:2] == ["python3", str(repo_dir.resolve() / "generate.py")]
    assert "--task" in command
    assert "ti2v-5B" in command
    assert "--image" in command
    assert "--save_file" in command
    assert "--offload_model" in command
    assert "--convert_model_dtype" in command
    assert "--t5_cpu" in command
    assert "put the banana in the bowl" in result.prompt
    assert result.future_images.shape == (2, 1, 3, 16, 16)
    assert result.seed == 13
    assert result.selected_frame_indices == (1, 2)
    assert result.total_video_frames == 5


def test_wan22_generator_wraps_subprocess_failure(tmp_path) -> None:
    repo_dir = tmp_path / "Wan2.2"
    checkpoint_dir = tmp_path / "Wan2.2-TI2V-5B"
    repo_dir.mkdir()
    checkpoint_dir.mkdir()
    (repo_dir / "generate.py").write_text("print('fake wan')\n")

    def failing_runner(command, **_kwargs):
        raise subprocess.CalledProcessError(
            returncode=2,
            cmd=command,
            output="partial stdout",
            stderr="wan stderr",
        )

    generator = Wan22FutureGenerator(
        Wan22Config(repo_dir=str(repo_dir), checkpoint_dir=str(checkpoint_dir)),
        runner=failing_runner,
        video_reader=lambda _video_path: [],
    )

    with pytest.raises(RuntimeError, match="Wan2.2 generation failed") as exc_info:
        generator.generate_view(
            torch.rand(3, 16, 16),
            task_text="open the drawer",
            output_dir=tmp_path / "out",
            image_size=16,
            num_future_frames=2,
        )

    message = str(exc_info.value)
    assert str(repo_dir.resolve() / "generate.py") in message
    assert "partial stdout" in message
    assert "wan stderr" in message


def test_wan22_generator_rejects_zero_frame_video(tmp_path) -> None:
    repo_dir = tmp_path / "Wan2.2"
    checkpoint_dir = tmp_path / "Wan2.2-TI2V-5B"
    repo_dir.mkdir()
    checkpoint_dir.mkdir()
    (repo_dir / "generate.py").write_text("print('fake wan')\n")

    generator = Wan22FutureGenerator(
        Wan22Config(repo_dir=str(repo_dir), checkpoint_dir=str(checkpoint_dir)),
        runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="", stderr=""),
        video_reader=lambda _video_path: [],
    )

    with pytest.raises(ValueError, match="did not produce any video frames"):
        generator.generate_future_stack(
            torch.rand(1, 3, 16, 16),
            task_text="close the drawer",
            output_dir=tmp_path / "out",
            image_size=16,
            num_future_frames=2,
        )


def _make_generator(tmp_path, *, video_reader, verify_conditioning_frame=True, conditioning_frame_max_mae=0.2):
    repo_dir = tmp_path / "Wan2.2"
    checkpoint_dir = tmp_path / "Wan2.2-TI2V-5B"
    repo_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    (repo_dir / "generate.py").write_text("print('fake wan')\n")
    return Wan22FutureGenerator(
        Wan22Config(repo_dir=str(repo_dir), checkpoint_dir=str(checkpoint_dir)),
        runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="", stderr=""),
        video_reader=video_reader,
        verify_conditioning_frame=verify_conditioning_frame,
        conditioning_frame_max_mae=conditioning_frame_max_mae,
    )


def test_verify_conditioning_frame_accepts_matching_frame() -> None:
    conditioning_frame = _solid_frame(100, size=8)
    current_image = _image_from_uint8(conditioning_frame)

    mae = verify_conditioning_frame(conditioning_frame, current_image, image_size=8, max_mean_abs_error=0.2)

    assert mae == pytest.approx(0.0, abs=1e-6)


def test_verify_conditioning_frame_allows_small_reconstruction_noise() -> None:
    current_image = torch.full((3, 8, 8), 0.5)
    # Frame 0 from a real VAE round-trip is close to, but not identical to, the conditioning image.
    noisy_frame = ((current_image + 0.03).clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)

    mae = verify_conditioning_frame(noisy_frame, current_image, image_size=8, max_mean_abs_error=0.2)

    assert 0.0 < mae < 0.2


def test_verify_conditioning_frame_rejects_mismatched_frame() -> None:
    current_image = torch.zeros((3, 8, 8))
    shifted_frame = _solid_frame(255, size=8)

    with pytest.raises(ValueError, match="conditioning image"):
        verify_conditioning_frame(shifted_frame, current_image, image_size=8, max_mean_abs_error=0.2)


def test_wan22_generator_rejects_shifted_conditioning_frame(tmp_path) -> None:
    conditioning_frame = _solid_frame(96)

    def shifted_reader(_video_path):
        # The conditioning frame is gray-96; here frame 0 is a later, very different frame.
        return [_solid_frame(220)] + [_solid_frame((index + 1) * 30) for index in range(4)]

    generator = _make_generator(tmp_path, video_reader=shifted_reader)

    with pytest.raises(ValueError, match="conditioning image"):
        generator.generate_future_stack(
            _image_from_uint8(conditioning_frame).unsqueeze(0),
            task_text="put the banana in the bowl",
            output_dir=tmp_path / "out",
            image_size=16,
            num_future_frames=2,
        )


def test_wan22_generator_rejects_missing_conditioning_frame(tmp_path) -> None:
    conditioning_frame = _solid_frame(230)

    def missing_reader(_video_path):
        # The decoder dropped the conditioning frame, so frame 0 is a blank/black frame.
        return [_solid_frame(0)] + [_solid_frame((index + 1) * 30) for index in range(4)]

    generator = _make_generator(tmp_path, video_reader=missing_reader)

    with pytest.raises(ValueError, match="conditioning image"):
        generator.generate_future_stack(
            _image_from_uint8(conditioning_frame).unsqueeze(0),
            task_text="put the banana in the bowl",
            output_dir=tmp_path / "out",
            image_size=16,
            num_future_frames=2,
        )


def test_wan22_generator_can_disable_conditioning_check(tmp_path) -> None:
    conditioning_frame = _solid_frame(96)

    def shifted_reader(_video_path):
        return [_solid_frame(220)] + [_solid_frame((index + 1) * 30) for index in range(4)]

    generator = _make_generator(tmp_path, video_reader=shifted_reader, verify_conditioning_frame=False)

    result = generator.generate_future_stack(
        _image_from_uint8(conditioning_frame).unsqueeze(0),
        task_text="put the banana in the bowl",
        output_dir=tmp_path / "out",
        image_size=16,
        num_future_frames=2,
    )

    assert result.future_images.shape == (2, 1, 3, 16, 16)
    assert result.selected_frame_indices == (1, 2)
