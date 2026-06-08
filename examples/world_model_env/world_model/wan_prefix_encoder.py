from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import torch
import torch.nn.functional as F

from world_model.wan_vae_encoder import add_repo_to_path, resolve_torch_dtype

DEFAULT_WAN_REPO_DIR = "/tmp/DiffSynth-Studio"
DEFAULT_WAN_CHECKPOINT_DIR = "/tmp/wan2.2-ti2v-5b"
DIFFSYNTH_WAN22_TI2V_CURRENT_SOURCE = "diffsynth_wan2.2_ti2v_5b_current_vae_text"
TEXT_COMPRESSION_DESCRIPTION = "masked_mean_pool_then_adaptive_avg_pool1d_or_pad_to_prefix_dim"


class WanCurrentPrefixEncoder(Protocol):
    prefix_dim: int

    def encode_prefix(self, current_images: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
        """Encode current RGB images and prompt text to ``(B, N, prefix_dim)`` tokens."""


def _validate_current_inputs(current_images: torch.Tensor, prompts: Sequence[str]) -> None:
    if current_images.ndim != 4:
        raise ValueError(f"current_images must have shape (B, 3, H, W), got {tuple(current_images.shape)}.")
    if current_images.shape[1] != 3:
        raise ValueError(f"current_images must have 3 RGB channels, got {current_images.shape[1]}.")
    if len(prompts) != current_images.shape[0]:
        raise ValueError(f"prompts length {len(prompts)} must match batch size {current_images.shape[0]}.")


def _fit_last_dim(values: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}.")
    if values.shape[-1] == width:
        return values
    if values.shape[-1] > width:
        original_shape = values.shape[:-1]
        pooled = F.adaptive_avg_pool1d(values.reshape(-1, 1, values.shape[-1]), width)
        return pooled.reshape(*original_shape, width)
    return F.pad(values, (0, width - values.shape[-1]))


def _prompt_hash_token(prompt: str, prefix_dim: int, *, device: torch.device) -> torch.Tensor:
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    repeats = (prefix_dim + len(digest) - 1) // len(digest)
    values = list((digest * repeats)[:prefix_dim])
    token = torch.tensor(values, dtype=torch.float32, device=device)
    return token.div(127.5).sub(1.0)


class FakeWanCurrentPrefixEncoder:
    """Deterministic current-only prefix encoder for unit tests and smoke runs."""

    def __init__(self, *, prefix_dim: int = 48, spatial_stride: int = 16) -> None:
        if prefix_dim <= 0:
            raise ValueError(f"prefix_dim must be positive, got {prefix_dim}.")
        if spatial_stride <= 0:
            raise ValueError(f"spatial_stride must be positive, got {spatial_stride}.")
        self.prefix_dim = prefix_dim
        self.spatial_stride = spatial_stride

    @torch.no_grad()
    def encode_prefix(self, current_images: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
        _validate_current_inputs(current_images, prompts)
        images = current_images.to(dtype=torch.float32).clamp(0.0, 1.0).mul(2.0).sub(1.0)
        batch_size, _, height, width = images.shape
        latent_height = max(1, height // self.spatial_stride)
        latent_width = max(1, width // self.spatial_stride)
        pooled = F.adaptive_avg_pool2d(images, (latent_height, latent_width))
        pooled = pooled.permute(0, 2, 3, 1).reshape(batch_size, latent_height * latent_width, 3)

        channel_indices = torch.arange(self.prefix_dim, device=images.device) % 3
        channel_scale = torch.linspace(0.75, 1.25, self.prefix_dim, device=images.device).view(1, 1, -1)
        image_tokens = pooled[:, :, channel_indices] * channel_scale

        text_tokens = torch.stack(
            [_prompt_hash_token(str(prompt), self.prefix_dim, device=images.device) for prompt in prompts],
            dim=0,
        ).unsqueeze(1)
        return torch.cat([text_tokens, image_tokens], dim=1).to(dtype=torch.float32)


def _resolve_existing_file(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _resolve_existing_dir(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


class FrozenDiffSynthWanCurrentPrefixEncoder:
    """Frozen Wan2.2 TI2V VAE plus T5 current-prefix encoder.

    This class intentionally never loads the Wan DiT denoiser. It produces a
    compact pi0.5-style prefix from the current image latent tokens and one
    pooled text token.
    """

    def __init__(
        self,
        *,
        repo_dir: str | Path = DEFAULT_WAN_REPO_DIR,
        checkpoint_dir: str | Path = DEFAULT_WAN_CHECKPOINT_DIR,
        vae_checkpoint_path: str | Path | None = None,
        text_encoder_checkpoint_path: str | Path | None = None,
        tokenizer_dir: str | Path | None = None,
        prefix_dim: int = 48,
        dtype: str = "bfloat16",
        tiled: bool = False,
    ) -> None:
        if prefix_dim <= 0:
            raise ValueError(f"prefix_dim must be positive, got {prefix_dim}.")
        checkpoint_root = Path(checkpoint_dir).expanduser()
        self.repo_dir = _resolve_existing_dir(repo_dir, label="DiffSynth-Studio repo")
        if not (self.repo_dir / "diffsynth").exists():
            raise FileNotFoundError(f"DiffSynth-Studio repo is missing diffsynth package: {self.repo_dir}")
        self.vae_checkpoint_path = _resolve_existing_file(
            vae_checkpoint_path or checkpoint_root / "Wan2.2_VAE.pth",
            label="Wan2.2 VAE checkpoint",
        )
        self.text_encoder_checkpoint_path = _resolve_existing_file(
            text_encoder_checkpoint_path or checkpoint_root / "models_t5_umt5-xxl-enc-bf16.pth",
            label="Wan T5 text encoder checkpoint",
        )
        self.tokenizer_dir = _resolve_existing_dir(
            tokenizer_dir or checkpoint_root / "google" / "umt5-xxl",
            label="Wan T5 tokenizer directory",
        )
        self.prefix_dim = prefix_dim
        self.dtype = dtype
        self.tiled = tiled
        self._pipe_by_device: dict[str, object] = {}

    def _load_pipeline(self, device: torch.device) -> object:
        device_key = str(device)
        if device_key in self._pipe_by_device:
            return self._pipe_by_device[device_key]

        add_repo_to_path(self.repo_dir)
        torch_dtype = resolve_torch_dtype(self.dtype)
        from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device_key,
            model_configs=[
                ModelConfig(path=str(self.vae_checkpoint_path)),
                ModelConfig(path=str(self.text_encoder_checkpoint_path)),
            ],
            tokenizer_config=ModelConfig(path=str(self.tokenizer_dir)),
            redirect_common_files=False,
        )
        if pipe.vae is None:
            raise RuntimeError(f"DiffSynth did not load a Wan VAE from {self.vae_checkpoint_path}.")
        if pipe.text_encoder is None:
            raise RuntimeError(
                f"DiffSynth did not load a Wan T5 text encoder from {self.text_encoder_checkpoint_path}."
            )
        if pipe.tokenizer is None:
            raise RuntimeError(f"DiffSynth did not load a Wan tokenizer from {self.tokenizer_dir}.")
        if pipe.dit is not None or pipe.dit2 is not None:
            raise RuntimeError("Wan current prefix encoder unexpectedly loaded a DiT denoiser; refusing to continue.")
        pipe.vae.eval().requires_grad_(False)
        pipe.text_encoder.eval().requires_grad_(False)
        self._pipe_by_device[device_key] = pipe
        return pipe

    def _encode_text_token(self, pipe: object, prompts: Sequence[str]) -> torch.Tensor:
        ids, mask = pipe.tokenizer(list(prompts), return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        context = pipe.text_encoder(ids, mask).to(dtype=torch.float32)
        valid = mask.gt(0)
        context = context.masked_fill(~valid.unsqueeze(-1), 0.0)
        pooled = context.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=context.dtype)
        return _fit_last_dim(pooled, self.prefix_dim).unsqueeze(1)

    def _encode_image_tokens(self, pipe: object, current_images: torch.Tensor) -> torch.Tensor:
        torch_dtype = resolve_torch_dtype(self.dtype)
        videos = current_images.clamp(0.0, 1.0).mul(2.0).sub(1.0).unsqueeze(2).contiguous()
        latents = pipe.vae.encode(
            [video.to(dtype=torch_dtype) for video in videos],
            device=str(current_images.device),
            tiled=self.tiled,
        ).to(device=current_images.device, dtype=torch.float32)
        if latents.ndim != 5:
            raise RuntimeError(f"Wan VAE returned latents with shape {tuple(latents.shape)}, expected (B, C, T, H, W).")
        tokens = latents.permute(0, 2, 3, 4, 1).reshape(latents.shape[0], -1, latents.shape[1])
        return _fit_last_dim(tokens, self.prefix_dim)

    @torch.no_grad()
    def encode_prefix(self, current_images: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
        _validate_current_inputs(current_images, prompts)
        pipe = self._load_pipeline(current_images.device)
        text_token = self._encode_text_token(pipe, prompts)
        image_tokens = self._encode_image_tokens(pipe, current_images)
        return torch.cat([text_token, image_tokens], dim=1).to(dtype=torch.float32)


__all__ = [
    "DEFAULT_WAN_CHECKPOINT_DIR",
    "DEFAULT_WAN_REPO_DIR",
    "DIFFSYNTH_WAN22_TI2V_CURRENT_SOURCE",
    "TEXT_COMPRESSION_DESCRIPTION",
    "FakeWanCurrentPrefixEncoder",
    "FrozenDiffSynthWanCurrentPrefixEncoder",
    "WanCurrentPrefixEncoder",
]
