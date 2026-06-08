from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image


def chw_float_to_uint8(image: torch.Tensor) -> np.ndarray:
    return (image.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def save_png(image: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(chw_float_to_uint8(image)).save(path)


def save_video(frames: torch.Tensor, path: str | Path, fps: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, [chw_float_to_uint8(frame) for frame in frames], fps=fps)
