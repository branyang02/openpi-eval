"""Self-contained world-model policy server for closed-loop MetaWorld smoke tests.

This server bridges a trained inverse-dynamics model (IDM) to the websocket
request/response protocol spoken by the repository's ``openpi_client`` clients
(see ``examples/metaworld/main.py``). It is intentionally free of MetaWorld /
MuJoCo dependencies: the environment driver stays in the root ``examples/metaworld``
example and connects to this server as a client.

Inference path (mirrors ``infer_wan_idm.py`` minus Wan):

    observation images + state
        -> current_images  (B, num_views, 3, H, W)
        -> future_images   (B, num_future_frames, num_views, 3, H, W)   [FutureProvider]
        -> IDM(current, future, state)
        -> action chunk    (B, action_horizon, action_dim)

The "future" frames are produced by a pluggable ``FutureProvider``. The default
:class:`RepeatCurrentFutureProvider` simply repeats the current frame, which keeps
plumbing/smoke tests fast and dependency-free. Real video world models (Wan2.2,
etc.) are slotted in at the same seam -- see :func:`build_future_provider` and the
``WorldModelPolicy(future_provider=...)`` argument.

Run the server (after ``uv sync`` so ``openpi-client`` + ``websockets`` are present):

    uv run serve_world_model.py --idm-checkpoint /path/to/idm.pt

Then point the MetaWorld driver at it:

    MUJOCO_GL=egl uv run examples/metaworld/main.py --env_name reach-v3
"""

from __future__ import annotations

import dataclasses
import logging
import time
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, Sequence, runtime_checkable

import numpy as np
import torch

from cache_pi05_wan_prefix_tokens import PrefixBackend, WanPrefixEncoderConfig, build_wan_prefix_encoder
from world_model.action_modes import WanActionMode, get_action_mode_spec
from world_model.config import FutureFrameStrategy, ModelConfig, Wan22Config
from world_model.data import (
    expected_wan_selected_frame_indices,
    image_to_chw_float,
    validate_wan_selected_frame_indices,
)
from world_model.diffsynth_wan import DiffSynthWanLoraConfig, DiffSynthWanLoraFutureGenerator
from world_model.models import InverseDynamicsModel
from world_model.pi05_wan_action_expert import (
    LoadedWanPi05ActionExpert,
    load_wan_pi05_action_expert_checkpoint,
    predict_denormalized_action_chunk,
)
from world_model.train_lib import (
    ActionNormalizer,
    StateNormalizer,
    create_flow_sample_noise,
    get_action_normalizer,
    get_state_normalizer,
    idm_history_kwargs,
    idm_uses_flow_matching,
    load_idm_checkpoint,
    load_idm_training_frame_delta,
    normalize_state_for_idm,
    resolve_device,
)
from world_model.wan_dit_prefix_encoder import DEFAULT_WAN_DIT_LAYERS
from world_model.wan_prefix_encoder import DEFAULT_WAN_CHECKPOINT_DIR, DEFAULT_WAN_REPO_DIR
from world_model.wan_vae_encoder import build_frozen_wan_vae_encoder

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_KEYS: tuple[str, ...] = ("observation/image",)
DEFAULT_STATE_KEY = "observation/state"
DEFAULT_PROMPT_KEY = "prompt"
IDM_HISTORY_KEYS: tuple[str, str, str] = ("prev_state_history", "prev_action_history", "history_mask")
PolicyKind = Literal["decoded_video_idm", "current_wan_prefix_action_expert"]


def _wan_action_mode_contract_metadata(mode: WanActionMode | str) -> dict[str, Any]:
    spec = get_action_mode_spec(mode)
    return {
        "mode": spec.mode.value,
        "runs_wan_generation": spec.runs_wan_generation,
        "generates_video": spec.generates_video,
        "produces_future_latents": spec.produces_future_latents,
        "consumes_future_pixels": spec.consumes_future_pixels,
        "pi05_style_current_prefix_reuse": spec.pi05_style_current_prefix_reuse,
        "exposes_reusable_action_memory": spec.exposes_reusable_action_memory,
        "native_wan_attention_kv_cache": spec.native_wan_attention_kv_cache,
        "memory_contract": spec.memory_contract,
    }


# --------------------------------------------------------------------------------------
# Future providers (extension point for Wan / video world models).
# --------------------------------------------------------------------------------------
@runtime_checkable
class FutureProvider(Protocol):
    """Produces future frames consumed by the IDM.

    A provider maps the current multi-view frames to a stack of predicted future
    frames. To plug in a video world model (e.g. Wan2.2), implement this protocol
    by wrapping ``world_model.wan22.Wan22FutureGenerator`` and pass the instance via
    ``WorldModelPolicy(future_provider=...)``.

    Args:
        current_images: ``(B, num_views, 3, H, W)`` float tensor in ``[0, 1]``.
        num_future_frames: number of future frames the IDM expects.
        prompts: optional per-item task strings (used by language-conditioned world
            models; ignored by :class:`RepeatCurrentFutureProvider`).

    Returns:
        ``(B, num_future_frames, num_views, 3, H, W)`` float tensor.
    """

    def __call__(
        self,
        current_images: torch.Tensor,
        *,
        num_future_frames: int,
        prompts: Sequence[str] | None = None,
    ) -> torch.Tensor: ...


class RepeatCurrentFutureProvider:
    """No-Wan future provider that repeats the current frame as every future frame.

    This is the fast, dependency-free default used for closed-loop plumbing tests.
    The IDM then sees a zero-delta transition; outputs are not physically meaningful
    but exercise the full request/response path end to end.
    """

    def __call__(
        self,
        current_images: torch.Tensor,
        *,
        num_future_frames: int,
        prompts: Sequence[str] | None = None,
    ) -> torch.Tensor:
        del prompts
        if current_images.ndim != 5:
            raise ValueError(
                f"current_images must be (B, num_views, 3, H, W), got shape {tuple(current_images.shape)}."
            )
        if num_future_frames < 1:
            raise ValueError(f"num_future_frames must be >= 1, got {num_future_frames}.")
        batch, views, channels, height, width = current_images.shape
        return current_images.unsqueeze(1).expand(batch, num_future_frames, views, channels, height, width).contiguous()


@runtime_checkable
class WanPrefixEncoderLike(Protocol):
    """Current-only Wan prefix encoder used by the pi0.5 action expert policy."""

    def encode_prefix(self, current_images: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
        """Encode current RGB images and prompt text to ``(B, N, D)`` prefix tokens."""


class WanLoraFutureProvider:
    """FutureProvider wrapper around the cached DiffSynth Wan LoRA generator."""

    name = "wan_lora"

    def __init__(
        self,
        generator: DiffSynthWanLoraFutureGenerator,
        *,
        image_size: int,
        frame_delta: int,
        output_dir: str | Path = "output/serve_world_model/wan_lora",
        seed: int | None = None,
    ) -> None:
        self.generator = generator
        self.image_size = image_size
        self.frame_delta = int(frame_delta)
        expected_wan_selected_frame_indices(self.frame_delta, 1)
        self.output_dir = Path(output_dir)
        self.seed = seed
        self._request_index = 0

    def _validate_selected_frame_indices(self, result: Any, *, num_future_frames: int, view_index: int) -> None:
        config = getattr(self.generator, "config", None)
        strategy: FutureFrameStrategy = getattr(config, "future_frame_strategy", "first")
        expected = expected_wan_selected_frame_indices(
            self.frame_delta,
            num_future_frames,
            strategy=strategy,
        )
        selected_frame_indices = getattr(result, "selected_frame_indices", None)
        if selected_frame_indices is None:
            raise ValueError(
                "DiffSynthWanLoraFutureGenerator must return result.selected_frame_indices; "
                f"expected generated-video indices {expected} for future_frame_strategy={strategy!r}, "
                f"frame_delta={self.frame_delta}, num_future_frames={num_future_frames}."
            )
        try:
            actual = validate_wan_selected_frame_indices(
                selected_frame_indices,
                frame_delta=self.frame_delta,
                num_future_frames=num_future_frames,
                strategy=strategy,
                context=f"DiffSynthWanLoraFutureGenerator result.selected_frame_indices for view_index={view_index}",
            )
        except ValueError as exc:
            raise ValueError(
                "future_provider='wan_lora' requires DiffSynth result.selected_frame_indices to use "
                f"post-conditioning generated-video indices; {exc}"
            ) from exc
        if actual != expected:
            raise AssertionError("validate_wan_selected_frame_indices returned unexpected indices.")

    def metadata_config(self) -> dict[str, Any]:
        config = getattr(self.generator, "config", None)
        metadata: dict[str, Any] = {
            "frame_delta": self.frame_delta,
            "output_dir": str(self.output_dir),
        }
        if config is None:
            return metadata
        metadata.update(
            {
                "lora_path": config.lora_path,
                "checkpoint_dir": config.checkpoint_dir,
                "height": config.height,
                "width": config.width,
                "num_frames": config.num_frames,
                "num_inference_steps": config.num_inference_steps,
                "prompt_template": config.prompt_template,
                "base_seed": config.base_seed,
                "seed": self.seed,
                "future_frame_strategy": config.future_frame_strategy,
                "device": config.device,
            }
        )
        return metadata

    def __call__(
        self,
        current_images: torch.Tensor,
        *,
        num_future_frames: int,
        prompts: Sequence[str] | None = None,
    ) -> torch.Tensor:
        if current_images.ndim != 5:
            raise ValueError(
                f"current_images must be (B, num_views, 3, H, W), got shape {tuple(current_images.shape)}."
            )
        if num_future_frames < 1:
            raise ValueError(f"num_future_frames must be >= 1, got {num_future_frames}.")
        batch_size, num_views, channels, height, width = current_images.shape
        if channels != 3:
            raise ValueError(f"current_images must have 3 color channels, got {channels}.")
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"current_images spatial size ({height}, {width}) must match image_size={self.image_size}."
            )
        if prompts is None:
            raise ValueError("future_provider='wan_lora' requires observation prompt text.")
        if len(prompts) != batch_size:
            raise ValueError(
                f"future_provider='wan_lora' requires {batch_size} prompt(s), got {len(prompts)}."
            )
        if any(not prompt.strip() for prompt in prompts):
            raise ValueError("future_provider='wan_lora' requires non-empty prompt text.")

        request_index = self._request_index
        self._request_index += 1
        batch_futures = []
        current_cpu = current_images.detach().cpu()
        for batch_index, prompt in enumerate(prompts):
            view_futures = []
            sample_output_dir = self.output_dir / f"request_{request_index:06d}" / f"sample_{batch_index:03d}"
            for view_index in range(num_views):
                result = self.generator.generate_future_stack(
                    current_cpu[batch_index],
                    task_text=prompt,
                    output_dir=sample_output_dir,
                    image_size=self.image_size,
                    num_future_frames=num_future_frames,
                    view_index=view_index,
                    seed=self.seed,
                )
                if result.future_images.ndim != 5 or result.future_images.shape[1] != 1:
                    raise ValueError(
                        "DiffSynthWanLoraFutureGenerator must return future_images with shape "
                        f"(F, 1, 3, H, W), got {tuple(result.future_images.shape)}."
                    )
                self._validate_selected_frame_indices(
                    result,
                    num_future_frames=num_future_frames,
                    view_index=view_index,
                )
                view_futures.append(result.future_images[:, 0])
            batch_futures.append(torch.stack(view_futures, dim=1))
        return torch.stack(batch_futures, dim=0)


def _validate_wan_lora_prompt_template(prompt_template: str) -> str:
    if not prompt_template.strip():
        raise ValueError("--wan-lora-prompt-template must not be blank.")
    if "{task}" not in prompt_template:
        raise ValueError("--wan-lora-prompt-template must contain '{task}'.")
    return prompt_template


def _validate_wan_lora_seed(seed: int | None) -> int | None:
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"--wan-lora-seed must be a nonnegative integer when set, got {seed!r}.")
    return seed


def build_wan_lora_future_provider(
    *,
    image_size: int,
    frame_delta: int | None = None,
    diffsynth_repo_dir: str | None = None,
    wan_lora_checkpoint_dir: str | None = None,
    wan_lora_path: str | None = None,
    wan_lora_height: int | None = None,
    wan_lora_width: int | None = None,
    wan_lora_num_frames: int = 17,
    wan_lora_num_inference_steps: int = 2,
    wan_lora_alpha: float = 1.0,
    wan_lora_tiled: bool = True,
    wan_lora_device: str | None = None,
    wan_lora_future_frame_strategy: FutureFrameStrategy = "first",
    wan_lora_output_dir: str | Path = "output/serve_world_model/wan_lora",
    wan_lora_prompt_template: str = Wan22Config.prompt_template,
    wan_lora_seed: int | None = None,
) -> WanLoraFutureProvider:
    if diffsynth_repo_dir is None:
        raise ValueError("--diffsynth-repo-dir is required when --future-provider wan_lora.")
    if wan_lora_checkpoint_dir is None:
        raise ValueError("--wan-lora-checkpoint-dir is required when --future-provider wan_lora.")
    if wan_lora_path is None:
        raise ValueError("--wan-lora-path is required when --future-provider wan_lora.")
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}.")
    height = image_size if wan_lora_height is None else wan_lora_height
    width = image_size if wan_lora_width is None else wan_lora_width
    if height <= 0 or width <= 0:
        raise ValueError(f"wan_lora height and width must be positive, got height={height}, width={width}.")
    if wan_lora_device is None:
        raise ValueError("wan_lora_device is required when building future_provider='wan_lora'.")
    if frame_delta is None:
        raise ValueError("frame_delta is required when building future_provider='wan_lora'.")
    prompt_template = _validate_wan_lora_prompt_template(wan_lora_prompt_template)
    seed = _validate_wan_lora_seed(wan_lora_seed)
    config_kwargs = {
        "diffsynth_repo_dir": diffsynth_repo_dir,
        "checkpoint_dir": wan_lora_checkpoint_dir,
        "lora_path": wan_lora_path,
        "height": height,
        "width": width,
        "num_frames": wan_lora_num_frames,
        "num_inference_steps": wan_lora_num_inference_steps,
        "lora_alpha": wan_lora_alpha,
        "device": wan_lora_device,
        "tiled": wan_lora_tiled,
        "frame_delta": frame_delta,
        "future_frame_strategy": wan_lora_future_frame_strategy,
        "prompt_template": prompt_template,
    }
    if seed is not None:
        config_kwargs["base_seed"] = seed
    return WanLoraFutureProvider(
        DiffSynthWanLoraFutureGenerator(DiffSynthWanLoraConfig(**config_kwargs)),
        image_size=image_size,
        frame_delta=frame_delta,
        output_dir=wan_lora_output_dir,
        seed=seed,
    )


def build_future_provider(name: str, **kwargs: Any) -> FutureProvider:
    """Factory for named future providers (used by the CLI).

    ``"repeat_current"`` is the smoke-test provider. ``"wan_lora"`` constructs a
    real DiffSynth Wan LoRA provider and reuses the loaded pipeline across calls.
    """
    if name == "repeat_current":
        if kwargs:
            raise ValueError(f"future_provider='repeat_current' does not accept config args: {sorted(kwargs)}.")
        return RepeatCurrentFutureProvider()
    if name == "wan_lora":
        return build_wan_lora_future_provider(**kwargs)
    raise ValueError(f"Unknown future provider: {name!r}. Available: 'repeat_current', 'wan_lora'.")


# --------------------------------------------------------------------------------------
# Observation parsing helpers.
# --------------------------------------------------------------------------------------
def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _view_to_tensor(value: Any, image_size: int, *, key: str) -> tuple[torch.Tensor, bool]:
    """Return ``(tensor (N, 3, H, W), batched)`` for one camera view.

    A 3-D array is a single ``(H, W, 3)`` / ``(3, H, W)`` image (N = 1). A 4-D array
    is a batch of ``B`` such images.
    """
    array = _to_numpy(value)
    if array.ndim == 3:
        return image_to_chw_float(array, image_size).unsqueeze(0), False
    if array.ndim == 4:
        frames = [image_to_chw_float(array[i], image_size) for i in range(array.shape[0])]
        return torch.stack(frames, dim=0), True
    raise ValueError(f'Observation image "{key}" must be (H, W, 3) or (B, H, W, 3), got shape {tuple(array.shape)}.')


def _state_to_tensor(value: Any, state_dim: int, *, key: str) -> tuple[torch.Tensor, bool]:
    array = _to_numpy(value).astype(np.float32)
    if array.ndim == 1:
        tensor, batched = torch.from_numpy(array).unsqueeze(0), False
    elif array.ndim == 2:
        tensor, batched = torch.from_numpy(array), True
    else:
        raise ValueError(
            f'Observation state "{key}" must be (state_dim,) or (B, state_dim), got shape {tuple(array.shape)}.'
        )
    if tensor.shape[-1] != state_dim:
        raise ValueError(
            f'Observation state "{key}" has dimension {tensor.shape[-1]}, but the IDM expects state_dim={state_dim}.'
        )
    return tensor, batched


def _history_array_to_tensor(value: Any, *, key: str) -> torch.Tensor:
    array = _to_numpy(value).astype(np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f'IDM history "{key}" must contain only finite values.')
    return torch.from_numpy(array)


def _parse_supplied_idm_history(
    obs: dict,
    *,
    history_length: int,
    state_dim: int,
    action_dim: int,
    batch_size: int,
    batched: bool,
) -> dict[str, torch.Tensor] | None:
    present = [key in obs for key in IDM_HISTORY_KEYS]
    if not any(present):
        return None
    if history_length == 0:
        raise ValueError(
            "Observation supplied IDM history tensors, but this model has idm_history_length=0."
        )
    if not all(present):
        missing = [key for key, is_present in zip(IDM_HISTORY_KEYS, present, strict=True) if not is_present]
        raise ValueError(f"Observation IDM history is missing required key(s): {missing}.")

    prev_state_history = _history_array_to_tensor(obs["prev_state_history"], key="prev_state_history")
    prev_action_history = _history_array_to_tensor(obs["prev_action_history"], key="prev_action_history")
    history_mask = _history_array_to_tensor(obs["history_mask"], key="history_mask")

    if batched:
        expected_state = (batch_size, history_length, state_dim)
        expected_action = (batch_size, history_length, action_dim)
        expected_mask = (batch_size, history_length)
        if tuple(prev_state_history.shape) != expected_state:
            raise ValueError(
                f"prev_state_history must have shape {expected_state} for batched observations, "
                f"got {tuple(prev_state_history.shape)}."
            )
        if tuple(prev_action_history.shape) != expected_action:
            raise ValueError(
                f"prev_action_history must have shape {expected_action} for batched observations, "
                f"got {tuple(prev_action_history.shape)}."
            )
        if tuple(history_mask.shape) != expected_mask:
            raise ValueError(
                f"history_mask must have shape {expected_mask} for batched observations, "
                f"got {tuple(history_mask.shape)}."
            )
    else:
        expected_state = (history_length, state_dim)
        expected_action = (history_length, action_dim)
        expected_mask = (history_length,)
        if tuple(prev_state_history.shape) != expected_state:
            raise ValueError(
                f"prev_state_history must have shape {expected_state} for unbatched observations, "
                f"got {tuple(prev_state_history.shape)}."
            )
        if tuple(prev_action_history.shape) != expected_action:
            raise ValueError(
                f"prev_action_history must have shape {expected_action} for unbatched observations, "
                f"got {tuple(prev_action_history.shape)}."
            )
        if tuple(history_mask.shape) != expected_mask:
            raise ValueError(
                f"history_mask must have shape {expected_mask} for unbatched observations, "
                f"got {tuple(history_mask.shape)}."
            )
        prev_state_history = prev_state_history.unsqueeze(0)
        prev_action_history = prev_action_history.unsqueeze(0)
        history_mask = history_mask.unsqueeze(0)

    return {
        "prev_state_history": prev_state_history,
        "prev_action_history": prev_action_history,
        "history_mask": history_mask,
    }


def _normalize_prompts(value: Any, batch_size: int) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value] * batch_size
    try:
        prompts = [str(item) for item in value]
    except TypeError as exc:
        raise ValueError(
            f'Observation prompt must be a string or a sequence with length {batch_size}, got {type(value).__name__}.'
        ) from exc
    if len(prompts) != batch_size:
        raise ValueError(
            f"Observation prompt sequence length ({len(prompts)}) must match batch size ({batch_size})."
        )
    return prompts


def _future_provider_name(provider: FutureProvider) -> str:
    if isinstance(provider, RepeatCurrentFutureProvider):
        return "repeat_current"
    explicit_name = getattr(provider, "name", None)
    if isinstance(explicit_name, str) and explicit_name:
        return explicit_name
    callable_name = getattr(provider, "__name__", None)
    if isinstance(callable_name, str) and callable_name:
        return callable_name
    return provider.__class__.__name__


def _validate_future_images(
    future_images: Any,
    *,
    expected_shape: tuple[int, int, int, int, int, int],
    provider_name: str,
) -> torch.Tensor:
    if not isinstance(future_images, torch.Tensor):
        raise TypeError(
            f"FutureProvider {provider_name!r} must return a torch.Tensor, got {type(future_images).__name__}."
        )
    if tuple(future_images.shape) != expected_shape:
        raise ValueError(
            f"FutureProvider {provider_name!r} returned shape {tuple(future_images.shape)}, "
            f"expected exact shape {expected_shape}."
        )
    if not torch.is_floating_point(future_images):
        raise ValueError(
            f"FutureProvider {provider_name!r} must return a floating dtype tensor, got {future_images.dtype}."
        )
    if not torch.isfinite(future_images).all().item():
        raise ValueError(f"FutureProvider {provider_name!r} returned non-finite values.")
    min_value = future_images.min().item()
    max_value = future_images.max().item()
    if min_value < 0.0 or max_value > 1.0:
        raise ValueError(
            f"FutureProvider {provider_name!r} must return values in range [0, 1], "
            f"got min={min_value:.6g}, max={max_value:.6g}."
        )
    return future_images


def _needs_live_wan_vae_latents(model_config: ModelConfig) -> bool:
    return (
        model_config.idm_visual_encoder == "wan_vae"
        and model_config.wan_vae_use_cached_latents
        and model_config.idm_future_conditioning != "current_only"
    )


def _wan_vae_video_from_images(current_images: torch.Tensor, future_images: torch.Tensor) -> torch.Tensor:
    """Mirror WanVaeTransitionEncoder._video_from_images for live serving."""
    current = current_images[:, 0].unsqueeze(1)
    future = future_images[:, :, 0]
    video = torch.cat([current, future], dim=1)
    return video.permute(0, 2, 1, 3, 4).mul(2.0).sub(1.0).contiguous()


def _require_explicit_repeat_current(future_provider: str, *, allow_repeat_current: bool) -> None:
    if future_provider == "repeat_current" and not allow_repeat_current:
        raise ValueError(
            "future_provider='repeat_current' is smoke-test behavior, not a real world-model eval. "
            "Pass --allow-repeat-current to run the repeat_current smoke server explicitly."
        )


def _synchronize_device_for_timing(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


# --------------------------------------------------------------------------------------
# Policy.
# --------------------------------------------------------------------------------------
class _SessionHistory:
    """Rolling previous-(state, action) buffer for one closed-loop serving session.

    Maintains the IDM history contract: ``prev_state_history`` ``(B, H, state_dim)``,
    ``prev_action_history`` ``(B, H, action_dim)`` and ``history_mask`` ``(B, H)``, with
    the most recent step in the LAST index (oldest first), matching the dataset window
    built in ``world_model.data`` and consumed by ``world_model.models``. RAW observation
    states and RAW (denormalized) actions are stored here; normalization is applied at
    inference time via :func:`world_model.train_lib.idm_history_kwargs`, so the buffer
    stays consistent with training. ``history_mask`` is 0 for slots not yet filled.

    Each websocket connection (and the policy's own direct-``infer`` path) owns a separate
    instance, so sessions never share history.
    """

    def __init__(self, *, history_length: int, state_dim: int, action_dim: int, device: torch.device) -> None:
        self.history_length = int(history_length)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.device = device
        self._batch_size: int | None = None
        self.prev_state_history: torch.Tensor | None = None
        self.prev_action_history: torch.Tensor | None = None
        self.history_mask: torch.Tensor | None = None

    def reset(self) -> None:
        """Forget all buffered history so the next request starts fresh (zero mask)."""
        self._batch_size = None
        self.prev_state_history = None
        self.prev_action_history = None
        self.history_mask = None

    def _allocate(self, batch_size: int) -> None:
        self._batch_size = batch_size
        self.prev_state_history = torch.zeros(
            batch_size, self.history_length, self.state_dim, device=self.device, dtype=torch.float32
        )
        self.prev_action_history = torch.zeros(
            batch_size, self.history_length, self.action_dim, device=self.device, dtype=torch.float32
        )
        self.history_mask = torch.zeros(batch_size, self.history_length, device=self.device, dtype=torch.float32)

    def current(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(prev_state_history, prev_action_history, history_mask)`` for this batch.

        The buffer is (re)allocated to zeros whenever the batch size changes, so a session
        that switches batch size resets its history rather than mixing rows.
        """
        if self._batch_size != batch_size:
            self._allocate(batch_size)
        return self.prev_state_history, self.prev_action_history, self.history_mask

    def update(self, state: torch.Tensor, action: torch.Tensor) -> None:
        """Roll one ``(state, action)`` step in as the most recent (last) history entry.

        ``state`` is the raw observation state ``(B, state_dim)`` and ``action`` is the raw
        (denormalized) first action of the returned chunk ``(B, action_dim)``.
        """
        batch_size = int(state.shape[0])
        if self._batch_size != batch_size:
            self._allocate(batch_size)
        if self.history_length == 0:
            return
        new_state = state.detach().to(self.device, dtype=torch.float32).unsqueeze(1)
        new_action = action.detach().to(self.device, dtype=torch.float32).unsqueeze(1)
        new_mask = torch.ones(batch_size, 1, device=self.device, dtype=torch.float32)
        self.prev_state_history = torch.cat([self.prev_state_history[:, 1:], new_state], dim=1)
        self.prev_action_history = torch.cat([self.prev_action_history[:, 1:], new_action], dim=1)
        self.history_mask = torch.cat([self.history_mask[:, 1:], new_mask], dim=1)


def _serving_idm_history_kwargs(
    batch: dict[str, torch.Tensor],
    *,
    action_normalizer: ActionNormalizer | None,
    state_normalizer: StateNormalizer | None,
) -> dict[str, torch.Tensor]:
    history_kwargs = idm_history_kwargs(
        batch,
        action_normalizer=action_normalizer,
        state_normalizer=None,
    )
    if state_normalizer is not None and "prev_state_history" in history_kwargs:
        # Checkpoint-loaded IDMs may normalize the current state in a forward pre-hook,
        # but that hook does not touch history kwargs. Normalize history explicitly here.
        history_kwargs["prev_state_history"] = state_normalizer.normalize(history_kwargs["prev_state_history"])
    return history_kwargs


class WorldModelPolicy:
    """Inverse-dynamics policy that returns an action chunk per observation.

    Compatible with ``openpi_client.base_policy.BasePolicy`` by duck typing
    (``infer`` / ``infer_many`` / ``warmup_many`` / ``reset``), so it can be served
    by this module's websocket server or by ``openpi.serving.WebsocketPolicyServer``.

    Observations follow the ``examples/metaworld/main.py`` contract: one image array
    per entry in ``image_keys`` (single ``(H, W, 3)`` or batched ``(B, H, W, 3)``),
    a state array under ``state_key``, and an optional ``prompt``. The response is
    ``{"actions": np.ndarray}`` shaped ``(B, action_horizon, action_dim)`` for batched
    input or ``(action_horizon, action_dim)`` for a single observation.
    """

    def __init__(
        self,
        idm: InverseDynamicsModel,
        model_config: ModelConfig,
        *,
        image_keys: Sequence[str] = DEFAULT_IMAGE_KEYS,
        state_key: str = DEFAULT_STATE_KEY,
        prompt_key: str = DEFAULT_PROMPT_KEY,
        future_provider: FutureProvider | None = None,
        action_normalizer: ActionNormalizer | None = None,
        state_normalizer: StateNormalizer | None = None,
        device: torch.device | str | None = None,
        flow_seed: int | None = 0,
    ) -> None:
        self.image_keys = tuple(image_keys)
        if len(self.image_keys) != model_config.num_views:
            raise ValueError(
                f"Number of image_keys ({len(self.image_keys)}) must equal the IDM num_views "
                f"({model_config.num_views}). Got image_keys={self.image_keys}."
            )
        if device is None:
            self.device = next(idm.parameters()).device
        else:
            self.device = torch.device(device) if isinstance(device, str) else device
        self.idm = idm.to(self.device).eval()
        self.model_config = model_config
        self.state_key = state_key
        self.prompt_key = prompt_key
        self.future_provider = future_provider or RepeatCurrentFutureProvider()
        self.future_provider_name = _future_provider_name(self.future_provider)
        self._needs_live_wan_vae_latents = _needs_live_wan_vae_latents(model_config)
        self._wan_vae_encoder = (
            build_frozen_wan_vae_encoder(model_config) if self._needs_live_wan_vae_latents else None
        )
        self.action_normalizer = action_normalizer.to(self.device) if action_normalizer is not None else None
        self.state_normalizer = state_normalizer.to(self.device) if state_normalizer is not None else None
        self.flow_seed = flow_seed
        self._uses_flow = idm_uses_flow_matching(idm)
        # Policy-owned default history buffer for direct ``infer`` calls. The websocket
        # handler allocates an independent one per connection via ``new_history_state``.
        self._history_state = self.new_history_state()

    def new_history_state(self) -> _SessionHistory:
        """Create an independent rolling history buffer for one serving session.

        The websocket handler calls this once per connection so concurrent clients never
        share previous state/action history. ``infer`` falls back to the policy's own
        default buffer when no ``history_state`` is passed.
        """
        return _SessionHistory(
            history_length=self.model_config.idm_history_length,
            state_dim=self.model_config.state_dim,
            action_dim=self.model_config.action_dim,
            device=self.device,
        )

    def _live_wan_vae_latents(self, current_images: torch.Tensor, future_images: torch.Tensor) -> torch.Tensor:
        if self._wan_vae_encoder is None:
            raise RuntimeError("Live Wan VAE latents were requested but the serving encoder is not initialized.")
        video = _wan_vae_video_from_images(current_images, future_images)
        latents = self._wan_vae_encoder.encode_videos(video)
        if not isinstance(latents, torch.Tensor):
            raise TypeError(
                "Wan VAE encoder encode_videos must return a torch.Tensor, "
                f"got {type(latents).__name__}."
            )
        return latents.to(device=self.device, dtype=torch.float32)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        image_keys: Sequence[str] = DEFAULT_IMAGE_KEYS,
        state_key: str = DEFAULT_STATE_KEY,
        prompt_key: str = DEFAULT_PROMPT_KEY,
        future_provider: FutureProvider | str = "repeat_current",
        future_provider_kwargs: dict[str, Any] | None = None,
        device: torch.device | str = "auto",
        flow_seed: int | None = 0,
    ) -> WorldModelPolicy:
        resolved_device = resolve_device(device) if isinstance(device, str) else device
        idm, model_config = load_idm_checkpoint(checkpoint_path, resolved_device)
        idm.eval()
        normalizer = get_action_normalizer(idm, resolved_device)
        state_normalizer = get_state_normalizer(idm, resolved_device)
        if callable(future_provider):
            if future_provider_kwargs:
                raise ValueError("future_provider_kwargs can only be used with a named future provider.")
            provider = future_provider
        else:
            provider_kwargs = dict(future_provider_kwargs or {})
            if future_provider == "wan_lora":
                provider_kwargs.setdefault("image_size", model_config.image_size)
                if provider_kwargs.get("wan_lora_device") is None:
                    provider_kwargs["wan_lora_device"] = str(resolved_device)
                training_frame_delta = load_idm_training_frame_delta(checkpoint_path)
                if training_frame_delta is None:
                    raise ValueError(
                        f"IDM checkpoint {str(checkpoint_path)!r} does not record its training frame_delta; "
                        "cannot serve future_provider='wan_lora' because selected Wan frames must match the "
                        "temporal gap the IDM was trained on."
                    )
                provider_kwargs["frame_delta"] = training_frame_delta
            provider = build_future_provider(future_provider, **provider_kwargs)
        return cls(
            idm,
            model_config,
            image_keys=image_keys,
            state_key=state_key,
            prompt_key=prompt_key,
            future_provider=provider,
            action_normalizer=normalizer,
            state_normalizer=state_normalizer,
            device=resolved_device,
            flow_seed=flow_seed,
        )

    def _parse_observation(
        self, obs: dict
    ) -> tuple[torch.Tensor, torch.Tensor, bool, list[str] | None, dict[str, torch.Tensor] | None]:
        missing = [key for key in (*self.image_keys, self.state_key) if key not in obs]
        if missing:
            raise KeyError(f"Observation is missing required key(s): {missing}.")

        view_tensors = []
        view_batched = []
        for key in self.image_keys:
            tensor, batched = _view_to_tensor(obs[key], self.model_config.image_size, key=key)
            view_tensors.append(tensor)
            view_batched.append(batched)
        state, state_batched = _state_to_tensor(obs[self.state_key], self.model_config.state_dim, key=self.state_key)

        flags = [*view_batched, state_batched]
        if any(flags) and not all(flags):
            raise ValueError(
                "Observation mixes single and batched entries; provide all images and state either "
                "unbatched (image (H, W, 3), state (state_dim,)) or batched (image (B, H, W, 3), state (B, state_dim))."
            )
        batched = all(flags)

        sizes = {tensor.shape[0] for tensor in view_tensors} | {state.shape[0]}
        if len(sizes) != 1:
            raise ValueError(
                f"Inconsistent batch sizes across observation entries: views={[t.shape[0] for t in view_tensors]}, "
                f"state={state.shape[0]}."
            )
        batch_size = sizes.pop()

        current_images = torch.stack(view_tensors, dim=1)  # (B, num_views, 3, H, W)
        prompts = _normalize_prompts(obs.get(self.prompt_key), batch_size)
        supplied_history = _parse_supplied_idm_history(
            obs,
            history_length=self.model_config.idm_history_length,
            state_dim=self.model_config.state_dim,
            action_dim=self.model_config.action_dim,
            batch_size=batch_size,
            batched=batched,
        )
        return current_images, state, batched, prompts, supplied_history

    def infer(self, obs: dict, *, history_state: _SessionHistory | None = None) -> dict:
        infer_start = time.perf_counter()
        current_images, state, batched, prompts, supplied_history = self._parse_observation(obs)
        current_images = current_images.to(self.device)
        state = state.to(self.device)
        # ``history_state`` is an internal seam (not part of the websocket/client API): the
        # server passes one per connection, direct callers fall back to the policy default.
        history_state = self._history_state if history_state is None else history_state

        _synchronize_device_for_timing(self.device)
        future_provider_start = time.perf_counter()
        future_images = self.future_provider(
            current_images,
            num_future_frames=self.model_config.num_future_frames,
            prompts=prompts,
        )
        _synchronize_device_for_timing(self.device)
        future_provider_ms = (time.perf_counter() - future_provider_start) * 1000.0
        expected_future_shape = (
            current_images.shape[0],
            self.model_config.num_future_frames,
            self.model_config.num_views,
            3,
            self.model_config.image_size,
            self.model_config.image_size,
        )
        future_images = _validate_future_images(
            future_images,
            expected_shape=expected_future_shape,
            provider_name=self.future_provider_name,
        ).to(self.device)

        # Build normalized history kwargs from the session's raw buffer, mirroring the
        # training/offline contract (``world_model.train_lib.idm_history_kwargs``).
        history_kwargs: dict[str, torch.Tensor] = {}
        if self.model_config.idm_history_length > 0:
            if supplied_history is None:
                prev_state_history, prev_action_history, history_mask = history_state.current(current_images.shape[0])
                history_batch = {
                    "prev_state_history": prev_state_history,
                    "prev_action_history": prev_action_history,
                    "history_mask": history_mask,
                }
            else:
                history_batch = {key: value.to(self.device) for key, value in supplied_history.items()}
            history_kwargs = _serving_idm_history_kwargs(
                history_batch,
                action_normalizer=self.action_normalizer,
                state_normalizer=self.state_normalizer,
            )

        _synchronize_device_for_timing(self.device)
        idm_start = time.perf_counter()
        with torch.no_grad():
            idm_kwargs = dict(history_kwargs)
            if self._needs_live_wan_vae_latents:
                idm_kwargs["wan_vae_latents"] = self._live_wan_vae_latents(current_images, future_images)
            sample_noise = None
            if self._uses_flow:
                generator = None
                if self.flow_seed is not None:
                    generator = torch.Generator(device=self.device).manual_seed(self.flow_seed)
                sample_noise = create_flow_sample_noise(
                    self.idm,
                    batch_size=current_images.shape[0],
                    device=self.device,
                    dtype=current_images.dtype,
                    generator=generator,
                )
            actions = self.idm(
                current_images,
                future_images,
                normalize_state_for_idm(self.idm, state, self.state_normalizer),
                None,
                sample_noise=sample_noise,
                **idm_kwargs,
            )
            if self.action_normalizer is not None:
                actions = self.action_normalizer.denormalize(actions)
        _synchronize_device_for_timing(self.device)
        idm_ms = (time.perf_counter() - idm_start) * 1000.0

        # Only maintain server-side fallback history for requests that did not supply
        # client history. Supplied history is authoritative, and not rolling it into this
        # buffer avoids mixing client-managed and server-managed episode state.
        if self.model_config.idm_history_length > 0 and supplied_history is None:
            history_state.update(state, actions[:, 0])

        actions_np = actions.detach().cpu().numpy().astype(np.float32)
        if not batched:
            actions_np = actions_np[0]
        return {
            "actions": actions_np,
            "server_timing": {
                "infer_ms": (time.perf_counter() - infer_start) * 1000.0,
                "future_provider_ms": future_provider_ms,
                "idm_ms": idm_ms,
            },
        }

    def infer_many(self, obs: Sequence[dict]) -> list[dict]:
        return [self.infer(item) for item in obs]

    def warmup_many(self, obs: dict, batch_sizes: Sequence[int]) -> None:
        del obs, batch_sizes

    def reset(self) -> None:
        self._history_state.reset()

    @property
    def metadata(self) -> dict:
        metadata = {
            "policy": "world_model_idm",
            "idm_arch": self.model_config.idm_arch,
            "idm_history_length": self.model_config.idm_history_length,
            "idm_history_keys": list(IDM_HISTORY_KEYS),
            "idm_history_client_supplied_preferred": self.model_config.idm_history_length > 0,
            "idm_history_note": (
                "For idm_history_length > 0, client-supplied chronological previous (state, action) "
                "history with history_mask is preferred/required for faithful MetaWorld serving when "
                "replan_steps > 1. Server-side fallback history is non-authoritative because it cannot "
                "know clipped executed actions or client episode resets."
            ),
            "num_views": self.model_config.num_views,
            "image_keys": list(self.image_keys),
            "state_key": self.state_key,
            "image_size": self.model_config.image_size,
            "state_dim": self.model_config.state_dim,
            "action_horizon": self.model_config.action_horizon,
            "action_dim": self.model_config.action_dim,
            "num_future_frames": self.model_config.num_future_frames,
            "wan_action_mode": WanActionMode.DECODED_VIDEO_IDM.value,
            "wan_action_mode_contract": _wan_action_mode_contract_metadata(WanActionMode.DECODED_VIDEO_IDM),
            "future_provider": self.future_provider_name,
            "future_provider_smoke": self.future_provider_name == "repeat_current",
            "uses_flow_matching": self._uses_flow,
            "live_wan_vae_latents": self._needs_live_wan_vae_latents,
            "state_normalized": self.state_normalizer is not None,
        }
        if isinstance(self.future_provider, WanLoraFutureProvider):
            metadata["future_provider_config"] = self.future_provider.metadata_config()
        return metadata


class WanPrefixActionExpertPolicy:
    """pi0.5-style action expert served from current-image Wan prefix tokens.

    This policy is a sibling to :class:`WorldModelPolicy`: it follows the same
    websocket duck-typed policy surface, but skips future generation entirely. The
    inference path is strictly ``current image + prompt -> prefix tokens -> action
    expert`` and never constructs or forwards future image tensors.
    """

    def __init__(
        self,
        loaded_action_expert: LoadedWanPi05ActionExpert,
        prefix_encoder: WanPrefixEncoderLike,
        *,
        image_keys: Sequence[str] = DEFAULT_IMAGE_KEYS,
        state_key: str = DEFAULT_STATE_KEY,
        prompt_key: str = DEFAULT_PROMPT_KEY,
        image_size: int = 224,
        device: torch.device | str | None = None,
        num_steps: int = 16,
        action_seed: int | None = 0,
    ) -> None:
        self.image_keys = tuple(image_keys)
        if len(self.image_keys) != 1:
            raise ValueError(
                "WanPrefixActionExpertPolicy requires exactly one image key because Wan current-prefix "
                f"encoding is single-view; got image_keys={self.image_keys}."
            )
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}.")
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}.")

        declared_mode = loaded_action_expert.wan_action_mode
        expected_mode = WanActionMode.CURRENT_WAN_PREFIX_ACTION_EXPERT.value
        if declared_mode is not None and declared_mode != expected_mode:
            raise ValueError(
                "WanPrefixActionExpertPolicy can only serve checkpoints declared as "
                f"{expected_mode!r}; got wan_action_mode={declared_mode!r}."
            )

        self.loaded_action_expert = loaded_action_expert
        self.action_expert = loaded_action_expert.model
        if device is None:
            self.device = next(self.action_expert.parameters()).device
        else:
            self.device = torch.device(device) if isinstance(device, str) else device
            self.action_expert.to(self.device)
        self.action_expert.eval()
        self.prefix_encoder = prefix_encoder
        self.state_key = state_key
        self.prompt_key = prompt_key
        self.image_size = int(image_size)
        self.num_steps = int(num_steps)
        self.action_seed = action_seed

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        prefix_encoder: WanPrefixEncoderLike,
        image_keys: Sequence[str] = DEFAULT_IMAGE_KEYS,
        state_key: str = DEFAULT_STATE_KEY,
        prompt_key: str = DEFAULT_PROMPT_KEY,
        image_size: int = 224,
        device: torch.device | str = "auto",
        num_steps: int = 16,
        action_seed: int | None = 0,
    ) -> WanPrefixActionExpertPolicy:
        resolved_device = resolve_device(device) if isinstance(device, str) else device
        loaded = load_wan_pi05_action_expert_checkpoint(checkpoint_path, resolved_device)
        return cls(
            loaded,
            prefix_encoder,
            image_keys=image_keys,
            state_key=state_key,
            prompt_key=prompt_key,
            image_size=image_size,
            device=resolved_device,
            num_steps=num_steps,
            action_seed=action_seed,
        )

    def _parse_observation(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor, bool, list[str]]:
        missing = [key for key in (*self.image_keys, self.state_key, self.prompt_key) if key not in obs]
        if missing:
            raise KeyError(f"Observation is missing required key(s): {missing}.")

        image_key = self.image_keys[0]
        current_images, image_batched = _view_to_tensor(obs[image_key], self.image_size, key=image_key)
        state, state_batched = _state_to_tensor(
            obs[self.state_key],
            self.action_expert.state_dim,
            key=self.state_key,
        )

        flags = [image_batched, state_batched]
        if any(flags) and not all(flags):
            raise ValueError(
                "Observation mixes single and batched entries; provide image and state either "
                "unbatched (image (H, W, 3), state (state_dim,)) or batched (image (B, H, W, 3), state (B, state_dim))."
            )
        batched = all(flags)
        if current_images.shape[0] != state.shape[0]:
            raise ValueError(
                f"Inconsistent batch sizes across observation entries: image={current_images.shape[0]}, "
                f"state={state.shape[0]}."
            )
        prompts = _normalize_prompts(obs.get(self.prompt_key), current_images.shape[0])
        if prompts is None:
            raise KeyError(f"Observation is missing required prompt key: {self.prompt_key!r}.")
        if any(not prompt.strip() for prompt in prompts):
            raise ValueError("WanPrefixActionExpertPolicy requires non-empty prompt text for every observation.")
        return current_images, state, batched, prompts

    def _validate_prefix_tokens(self, prefix_tokens: Any, *, batch_size: int) -> torch.Tensor:
        if not isinstance(prefix_tokens, torch.Tensor):
            raise TypeError(
                "Wan prefix encoder encode_prefix must return a torch.Tensor, "
                f"got {type(prefix_tokens).__name__}."
            )
        expected_prefix_dim = self.action_expert.prefix_dim
        if prefix_tokens.ndim != 3:
            raise ValueError(
                f"Wan prefix encoder must return shape (B, N, {expected_prefix_dim}), "
                f"got {tuple(prefix_tokens.shape)}."
            )
        if prefix_tokens.shape[0] != batch_size:
            raise ValueError(
                f"Wan prefix encoder returned batch size {prefix_tokens.shape[0]}, expected {batch_size}."
            )
        if prefix_tokens.shape[1] <= 0:
            raise ValueError("Wan prefix encoder must return at least one prefix token.")
        if prefix_tokens.shape[2] != expected_prefix_dim:
            raise ValueError(
                f"Wan prefix encoder returned prefix_dim={prefix_tokens.shape[2]}, "
                f"but the action expert expects prefix_dim={expected_prefix_dim}."
            )
        if not torch.is_floating_point(prefix_tokens):
            raise ValueError(f"Wan prefix encoder must return floating prefix tokens, got {prefix_tokens.dtype}.")
        if not torch.isfinite(prefix_tokens).all().item():
            raise ValueError("Wan prefix encoder returned non-finite prefix tokens.")
        return prefix_tokens.to(self.device)

    def new_history_state(self) -> None:
        """Compatibility no-op for the websocket server's per-connection hook."""
        return None

    def infer(self, obs: dict, *, history_state: Any | None = None) -> dict:
        del history_state
        infer_start = time.perf_counter()
        current_images, state, batched, prompts = self._parse_observation(obs)
        current_images = current_images.to(self.device)
        state = state.to(self.device)

        _synchronize_device_for_timing(self.device)
        prefix_encoder_start = time.perf_counter()
        with torch.no_grad():
            prefix_tokens = self.prefix_encoder.encode_prefix(current_images, prompts)
        _synchronize_device_for_timing(self.device)
        prefix_encoder_ms = (time.perf_counter() - prefix_encoder_start) * 1000.0
        prefix_tokens = self._validate_prefix_tokens(prefix_tokens, batch_size=current_images.shape[0])

        _synchronize_device_for_timing(self.device)
        action_expert_start = time.perf_counter()
        generator = None
        if self.action_seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(self.action_seed)
        actions = predict_denormalized_action_chunk(
            self.loaded_action_expert,
            prefix_tokens,
            state,
            num_steps=self.num_steps,
            generator=generator,
        )
        _synchronize_device_for_timing(self.device)
        action_expert_ms = (time.perf_counter() - action_expert_start) * 1000.0

        actions_np = actions.detach().cpu().numpy().astype(np.float32)
        if not batched:
            actions_np = actions_np[0]
        return {
            "actions": actions_np,
            "server_timing": {
                "infer_ms": (time.perf_counter() - infer_start) * 1000.0,
                "prefix_encoder_ms": prefix_encoder_ms,
                "action_expert_ms": action_expert_ms,
            },
        }

    def infer_many(self, obs: Sequence[dict]) -> list[dict]:
        return [self.infer(item) for item in obs]

    def warmup_many(self, obs: dict, batch_sizes: Sequence[int]) -> None:
        del obs, batch_sizes

    def reset(self) -> None:
        pass

    @property
    def metadata(self) -> dict:
        mode = self.loaded_action_expert.wan_action_mode or WanActionMode.CURRENT_WAN_PREFIX_ACTION_EXPERT.value
        metadata = {
            "policy": "pi05_wan_prefix_action_expert",
            "image_keys": list(self.image_keys),
            "state_key": self.state_key,
            "prompt_key": self.prompt_key,
            "image_size": self.image_size,
            "state_dim": self.action_expert.state_dim,
            "action_horizon": self.action_expert.action_horizon,
            "action_dim": self.action_expert.action_dim,
            "prefix_dim": self.action_expert.prefix_dim,
            "num_steps": self.num_steps,
            "prefix_encoder": self.prefix_encoder.__class__.__name__,
            "checkpoint_path": str(self.loaded_action_expert.checkpoint_path),
            "action_normalized": self.loaded_action_expert.action_norm_mean is not None
            and self.loaded_action_expert.action_norm_std is not None,
            "wan_action_mode": mode,
            "wan_action_mode_contract": _wan_action_mode_contract_metadata(mode),
        }
        return metadata


# --------------------------------------------------------------------------------------
# Websocket serving (lazy deps: openpi-client + websockets, installed via `uv sync`).
# --------------------------------------------------------------------------------------
def run_websocket_server(
    policy: Any,
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    metadata: dict | None = None,
    on_ready: Callable[[int], None] | None = None,
) -> None:
    """Serve ``policy`` over the openpi websocket protocol.

    Mirrors ``openpi.serving.websocket_policy_server.WebsocketPolicyServer`` so the
    existing ``openpi_client`` clients connect unchanged, while keeping this example
    self-contained (only ``openpi-client`` + ``websockets`` are required, not the
    full ``openpi`` package).
    """
    import asyncio

    asyncio.run(_serve_async(policy, host=host, port=port, metadata=metadata, on_ready=on_ready))


async def _serve_async(
    policy: Any,
    *,
    host: str,
    port: int,
    metadata: dict | None,
    on_ready: Callable[[int], None] | None,
) -> None:
    import http
    import time
    import traceback

    import websockets.asyncio.server as ws_server
    import websockets.frames
    from openpi_client import msgpack_numpy

    packer = msgpack_numpy.Packer()
    server_metadata = dict(metadata or {})

    def health_check(connection, request):
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None

    async def handler(websocket) -> None:
        logger.info("Connection from %s opened", websocket.remote_address)
        await websocket.send(packer.pack(server_metadata))
        # One history buffer per connection so concurrent clients never share state.
        history_state = policy.new_history_state()
        while True:
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())
                start_time = time.monotonic()
                action = dict(policy.infer(obs, history_state=history_state))
                server_timing = dict(action.get("server_timing", {}))
                server_timing["infer_ms"] = (time.monotonic() - start_time) * 1000
                action["server_timing"] = server_timing
                await websocket.send(packer.pack(action))
            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    logger.info("Serving WorldModelPolicy on ws://%s:%d", host, port)
    async with ws_server.serve(
        handler,
        host,
        port,
        compression=None,
        max_size=None,
        process_request=health_check,
        ping_timeout=600,
    ) as server:
        if on_ready is not None:
            socket = server.sockets[0]
            on_ready(int(socket.getsockname()[1]))
        await server.serve_forever()


@dataclasses.dataclass
class Args:
    """CLI arguments for the world-model policy server."""

    policy_kind: PolicyKind = "decoded_video_idm"
    idm_checkpoint: str | None = None
    pi05_checkpoint: str | None = None
    host: str = "0.0.0.0"
    port: int = 8000
    image_keys: tuple[str, ...] = DEFAULT_IMAGE_KEYS
    state_key: str = DEFAULT_STATE_KEY
    prompt_key: str = DEFAULT_PROMPT_KEY
    future_provider: str = "repeat_current"
    allow_repeat_current: bool = False
    device: str = "auto"
    flow_seed: int | None = 0
    diffsynth_repo_dir: str | None = None
    wan_lora_checkpoint_dir: str | None = None
    wan_lora_path: str | None = None
    wan_lora_height: int | None = None
    wan_lora_width: int | None = None
    wan_lora_num_frames: int = 17
    wan_lora_num_inference_steps: int = 2
    wan_lora_alpha: float = 1.0
    wan_lora_tiled: bool = True
    wan_lora_device: str | None = None
    wan_lora_future_frame_strategy: FutureFrameStrategy = "first"
    wan_lora_output_dir: str = "output/serve_world_model/wan_lora"
    wan_lora_prompt_template: str = Wan22Config.prompt_template
    wan_lora_seed: int | None = None
    pi05_image_size: int = 224
    pi05_num_steps: int = 16
    pi05_action_seed: int | None = 0
    prefix_backend: PrefixBackend = "vae_text"
    wan_repo_dir: str = DEFAULT_WAN_REPO_DIR
    wan_checkpoint_dir: str = DEFAULT_WAN_CHECKPOINT_DIR
    wan_vae_checkpoint_path: str | None = None
    wan_text_encoder_checkpoint_path: str | None = None
    wan_tokenizer_dir: str | None = None
    wan_dtype: str = "bfloat16"
    wan_tiled: bool = False
    dit_selected_layers: tuple[int, ...] = DEFAULT_WAN_DIT_LAYERS
    dit_hidden_pool: Literal["mean", "token_pool"] = "mean"
    dit_tokens_per_layer: int = 1
    dit_num_latent_frames: int = 1
    dit_timestep: float = 500.0
    dit_future_latent_fill: Literal["zeros", "noise"] = "zeros"
    dit_future_latent_seed: int = 0


def _future_provider_kwargs_from_args(args: Args) -> dict[str, Any]:
    if args.future_provider != "wan_lora":
        return {}
    if args.diffsynth_repo_dir is None:
        raise ValueError("--diffsynth-repo-dir is required when --future-provider wan_lora.")
    if args.wan_lora_checkpoint_dir is None:
        raise ValueError("--wan-lora-checkpoint-dir is required when --future-provider wan_lora.")
    if args.wan_lora_path is None:
        raise ValueError("--wan-lora-path is required when --future-provider wan_lora.")
    prompt_template = _validate_wan_lora_prompt_template(args.wan_lora_prompt_template)
    seed = _validate_wan_lora_seed(args.wan_lora_seed)
    return {
        "diffsynth_repo_dir": args.diffsynth_repo_dir,
        "wan_lora_checkpoint_dir": args.wan_lora_checkpoint_dir,
        "wan_lora_path": args.wan_lora_path,
        "wan_lora_height": args.wan_lora_height,
        "wan_lora_width": args.wan_lora_width,
        "wan_lora_num_frames": args.wan_lora_num_frames,
        "wan_lora_num_inference_steps": args.wan_lora_num_inference_steps,
        "wan_lora_alpha": args.wan_lora_alpha,
        "wan_lora_tiled": args.wan_lora_tiled,
        "wan_lora_device": args.wan_lora_device,
        "wan_lora_future_frame_strategy": args.wan_lora_future_frame_strategy,
        "wan_lora_output_dir": args.wan_lora_output_dir,
        "wan_lora_prompt_template": prompt_template,
        "wan_lora_seed": seed,
    }


def _pi05_checkpoint_prefix_dim(checkpoint_path: str | Path) -> int:
    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint {path} must be a mapping, got {type(checkpoint).__name__}.")
    model_kwargs = checkpoint.get("model_kwargs")
    if not isinstance(model_kwargs, dict):
        raise ValueError(f"Checkpoint {path} is missing required mapping key 'model_kwargs'.")
    prefix_dim = model_kwargs.get("prefix_dim")
    if isinstance(prefix_dim, bool) or not isinstance(prefix_dim, int) or prefix_dim <= 0:
        raise ValueError(
            f"Checkpoint {path} model_kwargs['prefix_dim'] must be a positive integer, got {prefix_dim!r}."
        )
    return prefix_dim


def _wan_prefix_encoder_config_from_args(args: Args, *, prefix_dim: int) -> WanPrefixEncoderConfig:
    if args.prefix_backend == "dit_hidden" and args.dit_num_latent_frames != 1:
        raise ValueError(
            "policy_kind='current_wan_prefix_action_expert' requires dit_num_latent_frames=1; "
            "use a current-only DiT prefix for this server policy."
        )
    return WanPrefixEncoderConfig(
        prefix_dim=prefix_dim,
        prefix_backend=args.prefix_backend,
        wan_repo_dir=args.wan_repo_dir,
        wan_checkpoint_dir=args.wan_checkpoint_dir,
        wan_vae_checkpoint_path=args.wan_vae_checkpoint_path,
        wan_text_encoder_checkpoint_path=args.wan_text_encoder_checkpoint_path,
        wan_tokenizer_dir=args.wan_tokenizer_dir,
        wan_dtype=args.wan_dtype,
        wan_tiled=args.wan_tiled,
        dit_selected_layers=args.dit_selected_layers,
        dit_hidden_pool=args.dit_hidden_pool,
        dit_tokens_per_layer=args.dit_tokens_per_layer,
        dit_num_latent_frames=args.dit_num_latent_frames,
        dit_timestep=args.dit_timestep,
        dit_future_latent_fill=args.dit_future_latent_fill,
        dit_future_latent_seed=args.dit_future_latent_seed,
    )


def _validate_decoded_video_idm_args(args: Args) -> None:
    if args.idm_checkpoint is None:
        raise ValueError("--idm-checkpoint is required when --policy-kind decoded_video_idm.")
    defaults = Args(idm_checkpoint=args.idm_checkpoint)
    prefix_only_overrides = []
    for field in (
        "pi05_checkpoint",
        "pi05_image_size",
        "pi05_num_steps",
        "pi05_action_seed",
        "prefix_backend",
        "wan_repo_dir",
        "wan_checkpoint_dir",
        "wan_vae_checkpoint_path",
        "wan_text_encoder_checkpoint_path",
        "wan_tokenizer_dir",
        "wan_dtype",
        "wan_tiled",
        "dit_selected_layers",
        "dit_hidden_pool",
        "dit_tokens_per_layer",
        "dit_num_latent_frames",
        "dit_timestep",
        "dit_future_latent_fill",
        "dit_future_latent_seed",
    ):
        if getattr(args, field) != getattr(defaults, field):
            prefix_only_overrides.append(field)
    if prefix_only_overrides:
        joined = ", ".join(f"--{field.replace('_', '-')}" for field in prefix_only_overrides)
        raise ValueError(
            "policy_kind='decoded_video_idm' does not use pi0.5/Wan prefix args; "
            f"remove {joined} or set --policy-kind current_wan_prefix_action_expert."
        )


def _validate_current_wan_prefix_args(args: Args) -> None:
    if args.pi05_checkpoint is None:
        raise ValueError("--pi05-checkpoint is required when --policy-kind current_wan_prefix_action_expert.")
    if args.idm_checkpoint is not None:
        raise ValueError(
            "policy_kind='current_wan_prefix_action_expert' does not load --idm-checkpoint; "
            "use --pi05-checkpoint for the action expert checkpoint."
        )
    if args.future_provider != "repeat_current":
        raise ValueError(
            "policy_kind='current_wan_prefix_action_expert' does not use --future-provider; "
            "prefix tokens are encoded from the current image only."
        )
    wan_lora_overrides = []
    defaults = Args(policy_kind=args.policy_kind, pi05_checkpoint=args.pi05_checkpoint)
    for field in (
        "diffsynth_repo_dir",
        "wan_lora_checkpoint_dir",
        "wan_lora_path",
        "wan_lora_height",
        "wan_lora_width",
        "wan_lora_num_frames",
        "wan_lora_num_inference_steps",
        "wan_lora_alpha",
        "wan_lora_tiled",
        "wan_lora_device",
        "wan_lora_future_frame_strategy",
        "wan_lora_output_dir",
        "wan_lora_prompt_template",
        "wan_lora_seed",
    ):
        if getattr(args, field) != getattr(defaults, field):
            wan_lora_overrides.append(field)
    if wan_lora_overrides:
        joined = ", ".join(f"--{field.replace('_', '-')}" for field in wan_lora_overrides)
        raise ValueError(
            "policy_kind='current_wan_prefix_action_expert' does not use Wan LoRA future-provider args; "
            f"remove {joined}."
        )
    if len(args.image_keys) != 1:
        raise ValueError(
            "policy_kind='current_wan_prefix_action_expert' requires exactly one --image-keys entry because "
            f"Wan current-prefix encoding is single-view; got {args.image_keys}."
        )
    if args.pi05_image_size <= 0:
        raise ValueError(f"--pi05-image-size must be positive, got {args.pi05_image_size}.")
    if args.pi05_num_steps <= 0:
        raise ValueError(f"--pi05-num-steps must be positive, got {args.pi05_num_steps}.")


def _build_current_wan_prefix_action_expert_policy(args: Args) -> WanPrefixActionExpertPolicy:
    _validate_current_wan_prefix_args(args)
    assert args.pi05_checkpoint is not None
    prefix_dim = _pi05_checkpoint_prefix_dim(args.pi05_checkpoint)
    prefix_encoder = build_wan_prefix_encoder(_wan_prefix_encoder_config_from_args(args, prefix_dim=prefix_dim))
    return WanPrefixActionExpertPolicy.from_checkpoint(
        args.pi05_checkpoint,
        prefix_encoder=prefix_encoder,
        image_keys=args.image_keys,
        state_key=args.state_key,
        prompt_key=args.prompt_key,
        image_size=args.pi05_image_size,
        device=args.device,
        num_steps=args.pi05_num_steps,
        action_seed=args.pi05_action_seed,
    )


def main(args: Args) -> None:
    if args.policy_kind == "decoded_video_idm":
        _validate_decoded_video_idm_args(args)
        assert args.idm_checkpoint is not None
        _require_explicit_repeat_current(args.future_provider, allow_repeat_current=args.allow_repeat_current)
        policy = WorldModelPolicy.from_checkpoint(
            args.idm_checkpoint,
            image_keys=args.image_keys,
            state_key=args.state_key,
            prompt_key=args.prompt_key,
            future_provider=args.future_provider,
            future_provider_kwargs=_future_provider_kwargs_from_args(args),
            device=args.device,
            flow_seed=args.flow_seed,
        )
    elif args.policy_kind == "current_wan_prefix_action_expert":
        policy = _build_current_wan_prefix_action_expert_policy(args)
    else:
        raise ValueError(f"Unknown policy_kind: {args.policy_kind!r}.")
    logger.info("Loaded policy: %s", policy.metadata)
    run_websocket_server(policy, host=args.host, port=args.port, metadata=policy.metadata)


if __name__ == "__main__":
    import tyro

    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
