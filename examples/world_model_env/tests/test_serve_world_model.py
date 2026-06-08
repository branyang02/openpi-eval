from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import serve_world_model as swm
from serve_world_model import (
    Args,
    RepeatCurrentFutureProvider,
    WanPrefixActionExpertPolicy,
    WorldModelPolicy,
    build_future_provider,
    main,
    run_websocket_server,
)
from world_model.config import ModelConfig, TrainConfig
from world_model.models import InverseDynamicsModel
from world_model.pi05_wan_action_expert import LoadedWanPi05ActionExpert, WanPi05ActionExpert
from world_model.train_lib import (
    ActionNormalizer,
    StateNormalizer,
    attach_state_normalizer,
    module_state_dict_for_checkpoint,
    save_idm_state_checkpoint,
)

DEFAULT_IMAGE_KEYS = ("observation/image", "observation/wrist_image")


def _tiny_model_config(**overrides) -> ModelConfig:
    base = dict(
        num_views=2,
        image_size=16,
        state_dim=4,
        action_dim=4,
        action_horizon=4,
        num_future_frames=1,
        idm_arch="stacked",
        latent_dim=32,
        task_embed_dim=8,
    )
    base.update(overrides)
    return ModelConfig(**base)


def _build_policy(config: ModelConfig | None = None, **policy_kwargs) -> WorldModelPolicy:
    config = config or _tiny_model_config()
    idm = InverseDynamicsModel(config).eval()
    policy_kwargs.setdefault("image_keys", DEFAULT_IMAGE_KEYS)
    return WorldModelPolicy(idm, config, **policy_kwargs)


def _make_obs(batch: int = 2, height: int = 16, width: int = 16, state_dim: int = 4) -> dict:
    rng = np.random.default_rng(0)
    return {
        "observation/image": rng.integers(0, 255, size=(batch, height, width, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 255, size=(batch, height, width, 3), dtype=np.uint8),
        "observation/state": rng.standard_normal((batch, state_dim)).astype(np.float32),
        "prompt": ["reach the goal"] * batch,
    }


def _make_single_view_obs(batch: int = 2, height: int = 16, width: int = 16, state_dim: int = 4) -> dict:
    """Observation matching the default one-view IDM (corner4 -> observation/image)."""
    rng = np.random.default_rng(0)
    return {
        "observation/image": rng.integers(0, 255, size=(batch, height, width, 3), dtype=np.uint8),
        "observation/state": rng.standard_normal((batch, state_dim)).astype(np.float32),
        "prompt": ["reach the goal"] * batch,
    }


def _single_view_policy(**policy_kwargs) -> WorldModelPolicy:
    config = _tiny_model_config(num_views=1)
    idm = InverseDynamicsModel(config).eval()
    return WorldModelPolicy(idm, config, **policy_kwargs)


def _history_model_config(history_length: int = 2, **overrides) -> ModelConfig:
    """A tiny single-view flow_transformer config with IDM history conditioning."""
    base = dict(
        num_views=1,
        idm_arch="flow_transformer",
        idm_history_length=history_length,
        idm_transformer_patch_size=16,
        idm_transformer_heads=8,
        idm_flow_sampling_steps=2,
    )
    base.update(overrides)
    return _tiny_model_config(**base)


def _detach_clone(value: torch.Tensor | None) -> torch.Tensor | None:
    return None if value is None else value.detach().cpu().clone()


class _CapturingHistoryIdm(torch.nn.Module):
    """Fake flow IDM that records the history kwargs it receives.

    It mimics a hist>0 flow_transformer: ``uses_flow_matching`` is True and the
    forward signature accepts ``prev_state_history`` / ``prev_action_history`` /
    ``history_mask``. Each call returns a deterministic action chunk whose horizon
    step ``j`` equals ``10 * call_index + j`` so tests can assert that exactly the
    FIRST action of the chunk is buffered into history.
    """

    uses_flow_matching = True

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.dummy = torch.nn.Parameter(torch.zeros(()))
        self.calls: list[dict] = []
        self._call_index = 0

    def forward(
        self,
        current_images,
        future_images,
        state,
        task_id=None,
        *,
        sample_noise=None,
        prev_state_history=None,
        prev_action_history=None,
        history_mask=None,
    ):
        del future_images, task_id, sample_noise
        self.calls.append(
            {
                "state": _detach_clone(state),
                "prev_state_history": _detach_clone(prev_state_history),
                "prev_action_history": _detach_clone(prev_action_history),
                "history_mask": _detach_clone(history_mask),
            }
        )
        batch = current_images.shape[0]
        action = torch.zeros(batch, self.config.action_horizon, self.config.action_dim, device=current_images.device)
        for horizon_index in range(self.config.action_horizon):
            action[:, horizon_index, :] = 10.0 * self._call_index + horizon_index
        self._call_index += 1
        return action


class _RecordingWanLatentIdm(torch.nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.dummy = torch.nn.Parameter(torch.zeros(()))
        self.calls: list[dict] = []

    def forward(
        self,
        current_images,
        future_images,
        state,
        task_id=None,
        *,
        sample_noise=None,
        wan_vae_latents=None,
    ):
        self.calls.append(
            {
                "current_images": _detach_clone(current_images),
                "future_images": _detach_clone(future_images),
                "state": _detach_clone(state),
                "task_id": task_id,
                "sample_noise": _detach_clone(sample_noise),
                "wan_vae_latents": _detach_clone(wan_vae_latents),
            }
        )
        return torch.zeros(
            current_images.shape[0],
            self.config.action_horizon,
            self.config.action_dim,
            device=current_images.device,
            dtype=current_images.dtype,
        )


class _RecordingWanVaeEncoder:
    def __init__(self) -> None:
        self.videos: list[torch.Tensor] = []

    def encode_videos(self, videos: torch.Tensor) -> torch.Tensor:
        self.videos.append(videos.detach().cpu().clone())
        batch_size = videos.shape[0]
        return torch.full(
            (batch_size, 48, 1, 1, 1),
            7.0,
            device=videos.device,
            dtype=torch.float16,
        )


class _FakeWanPrefixEncoder:
    def __init__(self, *, prefix_dim: int = 8, num_tokens: int = 3) -> None:
        self.prefix_dim = prefix_dim
        self.num_tokens = num_tokens
        self.calls: list[dict] = []

    def encode_prefix(self, current_images: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        self.calls.append(
            {
                "current_images": current_images.detach().cpu().clone(),
                "prompts": list(prompts),
            }
        )
        batch_size = current_images.shape[0]
        prompt_signal = torch.tensor([len(prompt) for prompt in prompts], device=current_images.device).view(
            batch_size, 1, 1
        )
        return torch.ones(
            batch_size,
            self.num_tokens,
            self.prefix_dim,
            device=current_images.device,
            dtype=torch.float32,
        ) * prompt_signal


def _tiny_wan_action_expert_kwargs(**overrides) -> dict[str, object]:
    kwargs = {
        "prefix_dim": 8,
        "state_dim": 4,
        "action_dim": 3,
        "action_horizon": 2,
        "hidden_dim": 16,
        "num_layers": 1,
        "num_heads": 4,
        "dropout": 0.0,
        "conditioning_mode": "wan_prefix_state",
        "timestep_conditioning": "additive",
        "decoder_arch": "encoder",
    }
    kwargs.update(overrides)
    return kwargs


def _tiny_loaded_wan_action_expert(
    *,
    checkpoint_path: str | Path = "/tmp/fake_wan_pi05_action_expert.pt",
    wan_action_mode: str | None = "current_wan_prefix_action_expert",
) -> LoadedWanPi05ActionExpert:
    model = WanPi05ActionExpert(**_tiny_wan_action_expert_kwargs()).eval()
    for parameter in model.parameters():
        parameter.data.zero_()
    return LoadedWanPi05ActionExpert(
        model=model,
        checkpoint_path=Path(checkpoint_path),
        action_normalization={"enabled": False},
        args={},
        metrics={},
        wan_action_mode=wan_action_mode,
    )


def _wan_prefix_policy(**kwargs) -> tuple[WanPrefixActionExpertPolicy, _FakeWanPrefixEncoder]:
    encoder = _FakeWanPrefixEncoder()
    kwargs.setdefault("image_size", 16)
    kwargs.setdefault("device", "cpu")
    policy = WanPrefixActionExpertPolicy(_tiny_loaded_wan_action_expert(), encoder, **kwargs)
    return policy, encoder


def _history_policy_with_capturing_idm(
    config: ModelConfig | None = None, **policy_kwargs
) -> tuple[WorldModelPolicy, _CapturingHistoryIdm]:
    config = config or _history_model_config()
    idm = _CapturingHistoryIdm(config)
    policy_kwargs.setdefault("image_keys", ("observation/image",))
    policy_kwargs.setdefault("device", "cpu")
    return WorldModelPolicy(idm, config, **policy_kwargs), idm


def _wan_lora_checkpoint_kwargs(**overrides):
    kwargs = {
        "diffsynth_repo_dir": "/tmp/DiffSynth-Studio",
        "wan_lora_checkpoint_dir": "/tmp/Wan2.2-TI2V-5B",
        "wan_lora_path": "/tmp/epoch-0.safetensors",
    }
    kwargs.update(overrides)
    return kwargs


def _assert_timing_value(value: object) -> None:
    assert isinstance(value, float)
    assert np.isfinite(value)
    assert value >= 0.0


def _assert_server_timing_shape(timing: dict) -> None:
    assert set(timing) == {"infer_ms", "future_provider_ms", "idm_ms"}
    for value in timing.values():
        _assert_timing_value(value)


def _assert_wan_prefix_timing_shape(timing: dict) -> None:
    assert set(timing) == {"infer_ms", "prefix_encoder_ms", "action_expert_ms"}
    for value in timing.values():
        _assert_timing_value(value)


def test_repeat_current_future_provider_repeats_current_frame() -> None:
    provider = RepeatCurrentFutureProvider()
    current = torch.rand(2, 3, 3, 8, 8)  # (batch, views, channels, h, w)

    future = provider(current, num_future_frames=2)

    assert future.shape == (2, 2, 3, 3, 8, 8)
    # Every future frame should equal the current observation frame.
    assert torch.equal(future[:, 0], current)
    assert torch.equal(future[:, 1], current)


def test_build_future_provider_constructs_wan_lora_generator(monkeypatch, tmp_path) -> None:
    constructed: dict = {}

    class FakeDiffSynthWanLoraFutureGenerator:
        def __init__(self, config):
            self.config = config
            constructed["config"] = config

    monkeypatch.setattr(swm, "DiffSynthWanLoraFutureGenerator", FakeDiffSynthWanLoraFutureGenerator)

    provider = build_future_provider(
        "wan_lora",
        image_size=16,
        frame_delta=4,
        diffsynth_repo_dir="/tmp/DiffSynth-Studio",
        wan_lora_checkpoint_dir="/tmp/Wan2.2-TI2V-5B",
        wan_lora_path="/tmp/epoch-0.safetensors",
        wan_lora_height=32,
        wan_lora_width=48,
        wan_lora_num_frames=9,
        wan_lora_num_inference_steps=5,
        wan_lora_alpha=0.25,
        wan_lora_tiled=False,
        wan_lora_device="cpu",
        wan_lora_future_frame_strategy="first",
        wan_lora_output_dir=str(tmp_path / "wan_live"),
        wan_lora_prompt_template="Robot task: {task}",
        wan_lora_seed=123,
    )

    assert isinstance(provider, swm.WanLoraFutureProvider)
    assert provider.name == "wan_lora"
    assert provider.frame_delta == 4
    assert provider.output_dir == tmp_path / "wan_live"
    config = constructed["config"]
    assert config.diffsynth_repo_dir == "/tmp/DiffSynth-Studio"
    assert config.checkpoint_dir == "/tmp/Wan2.2-TI2V-5B"
    assert config.lora_path == "/tmp/epoch-0.safetensors"
    assert config.height == 32
    assert config.width == 48
    assert config.num_frames == 9
    assert config.num_inference_steps == 5
    assert config.lora_alpha == 0.25
    assert config.tiled is False
    assert config.device == "cpu"
    assert config.frame_delta == 4
    assert config.future_frame_strategy == "first"
    assert config.prompt_template == "Robot task: {task}"
    assert config.base_seed == 123
    assert provider.seed == 123
    metadata_config = provider.metadata_config()
    assert metadata_config["lora_path"] == "/tmp/epoch-0.safetensors"
    assert metadata_config["checkpoint_dir"] == "/tmp/Wan2.2-TI2V-5B"
    assert metadata_config["height"] == 32
    assert metadata_config["width"] == 48
    assert metadata_config["num_frames"] == 9
    assert metadata_config["num_inference_steps"] == 5
    assert metadata_config["prompt_template"] == "Robot task: {task}"
    assert metadata_config["base_seed"] == 123
    assert metadata_config["seed"] == 123
    assert metadata_config["future_frame_strategy"] == "first"
    assert metadata_config["frame_delta"] == 4
    assert metadata_config["device"] == "cpu"
    policy_config = _tiny_model_config(num_views=1)
    policy = WorldModelPolicy(InverseDynamicsModel(policy_config), policy_config, future_provider=provider)
    assert policy.metadata["future_provider_config"] == metadata_config


def test_build_future_provider_rejects_lora_linspace_strategy(monkeypatch, tmp_path) -> None:
    class FakeDiffSynthWanLoraFutureGenerator:
        def __init__(self, config):
            self.config = config

    monkeypatch.setattr(swm, "DiffSynthWanLoraFutureGenerator", FakeDiffSynthWanLoraFutureGenerator)

    with pytest.raises(ValueError, match="future_frame_strategy must be one of"):
        build_future_provider(
            "wan_lora",
            image_size=16,
            frame_delta=4,
            diffsynth_repo_dir="/tmp/DiffSynth-Studio",
            wan_lora_checkpoint_dir="/tmp/Wan2.2-TI2V-5B",
            wan_lora_path="/tmp/epoch-0.safetensors",
            wan_lora_device="cpu",
            wan_lora_future_frame_strategy="linspace",
            wan_lora_output_dir=str(tmp_path / "wan_live"),
        )


def test_future_provider_kwargs_from_args_includes_wan_lora_prompt_template_and_seed() -> None:
    args = Args(
        future_provider="wan_lora",
        diffsynth_repo_dir="/tmp/DiffSynth-Studio",
        wan_lora_checkpoint_dir="/tmp/Wan2.2-TI2V-5B",
        wan_lora_path="/tmp/epoch-0.safetensors",
        wan_lora_prompt_template="Prompt for {task}",
        wan_lora_seed=77,
    )

    kwargs = swm._future_provider_kwargs_from_args(args)

    assert kwargs["wan_lora_prompt_template"] == "Prompt for {task}"
    assert kwargs["wan_lora_seed"] == 77


@pytest.mark.parametrize("prompt_template", ["", "   ", "Robot task"])
def test_build_future_provider_rejects_invalid_wan_lora_prompt_template(
    monkeypatch, tmp_path, prompt_template
) -> None:
    class UnusedDiffSynthWanLoraFutureGenerator:
        def __init__(self, config):
            del config
            raise AssertionError("generator should not be built with invalid prompt template")

    monkeypatch.setattr(swm, "DiffSynthWanLoraFutureGenerator", UnusedDiffSynthWanLoraFutureGenerator)

    with pytest.raises(ValueError, match="wan-lora-prompt-template"):
        build_future_provider(
            "wan_lora",
            image_size=16,
            frame_delta=4,
            diffsynth_repo_dir="/tmp/DiffSynth-Studio",
            wan_lora_checkpoint_dir="/tmp/Wan2.2-TI2V-5B",
            wan_lora_path="/tmp/epoch-0.safetensors",
            wan_lora_device="cpu",
            wan_lora_prompt_template=prompt_template,
            wan_lora_output_dir=str(tmp_path / "wan_live"),
        )


def test_build_future_provider_rejects_negative_wan_lora_seed(monkeypatch, tmp_path) -> None:
    class UnusedDiffSynthWanLoraFutureGenerator:
        def __init__(self, config):
            del config
            raise AssertionError("generator should not be built with invalid seed")

    monkeypatch.setattr(swm, "DiffSynthWanLoraFutureGenerator", UnusedDiffSynthWanLoraFutureGenerator)

    with pytest.raises(ValueError, match="wan-lora-seed"):
        build_future_provider(
            "wan_lora",
            image_size=16,
            frame_delta=4,
            diffsynth_repo_dir="/tmp/DiffSynth-Studio",
            wan_lora_checkpoint_dir="/tmp/Wan2.2-TI2V-5B",
            wan_lora_path="/tmp/epoch-0.safetensors",
            wan_lora_device="cpu",
            wan_lora_seed=-1,
            wan_lora_output_dir=str(tmp_path / "wan_live"),
        )


def test_wan_lora_future_provider_requires_prompt(tmp_path) -> None:
    class UnusedGenerator:
        def generate_future_stack(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("generator should not be called without prompt text")

    provider = swm.WanLoraFutureProvider(UnusedGenerator(), image_size=16, frame_delta=4, output_dir=tmp_path)
    current = torch.rand(1, 1, 3, 16, 16)

    with pytest.raises(ValueError, match="prompt"):
        provider(current, num_future_frames=1)
    with pytest.raises(ValueError, match="prompt"):
        provider(current, num_future_frames=1, prompts=[""])


def test_wan_lora_future_provider_accepts_generated_video_selected_frame_indices(tmp_path) -> None:
    calls: list[tuple[int, int]] = []

    class FakeGenerator:
        def generate_future_stack(
            self,
            current_images,
            *,
            task_text,
            output_dir,
            image_size,
            num_future_frames,
            view_index=0,
            seed=None,
        ):
            del current_images, task_text, output_dir, seed
            calls.append((view_index, num_future_frames))
            future_images = torch.full(
                (num_future_frames, 1, 3, image_size, image_size),
                fill_value=float(view_index) / 10.0,
            )
            return SimpleNamespace(
                future_images=future_images,
                selected_frame_indices=(1, 2),
            )

    provider = swm.WanLoraFutureProvider(FakeGenerator(), image_size=16, frame_delta=4, output_dir=tmp_path)
    current = torch.rand(1, 2, 3, 16, 16)

    future = provider(current, num_future_frames=2, prompts=["reach the goal"])

    assert future.shape == (1, 2, 2, 3, 16, 16)
    assert calls == [(0, 2), (1, 2)]


def test_wan_lora_future_provider_passes_configured_seed_to_generator(tmp_path) -> None:
    seeds: list[int | None] = []

    class FakeGenerator:
        config = SimpleNamespace(future_frame_strategy="first")

        def generate_future_stack(
            self,
            current_images,
            *,
            task_text,
            output_dir,
            image_size,
            num_future_frames,
            view_index=0,
            seed=None,
        ):
            del current_images, task_text, output_dir, view_index
            seeds.append(seed)
            return SimpleNamespace(
                future_images=torch.zeros(num_future_frames, 1, 3, image_size, image_size),
                selected_frame_indices=(1, 2),
            )

    provider = swm.WanLoraFutureProvider(
        FakeGenerator(),
        image_size=16,
        frame_delta=1,
        output_dir=tmp_path,
        seed=456,
    )
    current = torch.rand(1, 1, 3, 16, 16)

    future = provider(current, num_future_frames=2, prompts=["reach the goal"])

    assert future.shape == (1, 2, 1, 3, 16, 16)
    assert seeds == [456]


def test_wan_lora_future_provider_rejects_temporally_wrong_selected_frame_indices(tmp_path) -> None:
    class FakeGenerator:
        def generate_future_stack(
            self,
            current_images,
            *,
            task_text,
            output_dir,
            image_size,
            num_future_frames,
            view_index=0,
            seed=None,
        ):
            del current_images, task_text, output_dir, view_index, seed
            return SimpleNamespace(
                future_images=torch.zeros(num_future_frames, 1, 3, image_size, image_size),
                selected_frame_indices=(4, 8),
            )

    provider = swm.WanLoraFutureProvider(FakeGenerator(), image_size=16, frame_delta=4, output_dir=tmp_path)
    current = torch.rand(1, 1, 3, 16, 16)

    with pytest.raises(ValueError, match="selected_frame_indices") as excinfo:
        provider(current, num_future_frames=2, prompts=["reach the goal"])
    message = str(excinfo.value)
    assert "generated-video frame contract [1, 2]" in message
    assert "[4, 8]" in message


def test_wan_lora_future_provider_feeds_idm_and_returns_action_shape(tmp_path) -> None:
    calls: list[dict] = []

    class FakeGenerator:
        def generate_future_stack(
            self,
            current_images,
            *,
            task_text,
            output_dir,
            image_size,
            num_future_frames,
            view_index=0,
            seed=None,
        ):
            calls.append(
                {
                    "current_shape": tuple(current_images.shape),
                    "task_text": task_text,
                    "output_dir": output_dir,
                    "image_size": image_size,
                    "num_future_frames": num_future_frames,
                    "view_index": view_index,
                    "seed": seed,
                }
            )
            future_images = current_images[view_index].unsqueeze(0).repeat(num_future_frames, 1, 1, 1).unsqueeze(1)
            return SimpleNamespace(
                future_images=future_images,
                selected_frame_indices=tuple(range(1, num_future_frames + 1)),
            )

    config = _tiny_model_config(num_views=1)
    idm = InverseDynamicsModel(config).eval()
    provider = swm.WanLoraFutureProvider(
        FakeGenerator(),
        image_size=config.image_size,
        frame_delta=1,
        output_dir=tmp_path,
    )
    policy = WorldModelPolicy(idm, config, future_provider=provider)

    result = policy.infer(_make_single_view_obs(batch=2))

    assert result["actions"].shape == (2, config.action_horizon, config.action_dim)
    assert policy.metadata["future_provider"] == "wan_lora"
    assert [call["task_text"] for call in calls] == ["reach the goal", "reach the goal"]
    assert [call["current_shape"] for call in calls] == [(1, 3, config.image_size, config.image_size)] * 2
    assert [call["view_index"] for call in calls] == [0, 0]
    assert all(call["num_future_frames"] == config.num_future_frames for call in calls)


def test_cached_wan_vae_idm_receives_live_latents_from_serving(monkeypatch) -> None:
    config = _tiny_model_config(
        num_views=1,
        num_future_frames=2,
        idm_arch="flow_transformer",
        idm_visual_encoder="wan_vae",
        wan_vae_use_cached_latents=True,
        wan_vae_repo_dir="/tmp/DiffSynth-Studio",
        wan_vae_checkpoint_path="/tmp/Wan2.2_VAE.pth",
    )
    encoder = _RecordingWanVaeEncoder()
    build_calls: list[ModelConfig] = []

    def fake_build_frozen_wan_vae_encoder(model_config):
        build_calls.append(model_config)
        return encoder

    def future_provider(current_images, *, num_future_frames, prompts=None):
        del prompts
        assert num_future_frames == 2
        future = torch.empty(
            current_images.shape[0],
            num_future_frames,
            1,
            3,
            config.image_size,
            config.image_size,
            device=current_images.device,
            dtype=current_images.dtype,
        )
        future[:, 0].fill_(0.25)
        future[:, 1].fill_(0.75)
        return future

    monkeypatch.setattr(swm, "build_frozen_wan_vae_encoder", fake_build_frozen_wan_vae_encoder)
    idm = _RecordingWanLatentIdm(config)
    policy = WorldModelPolicy(
        idm,
        config,
        image_keys=("observation/image",),
        future_provider=future_provider,
        device="cpu",
    )
    obs = _make_single_view_obs(batch=2)
    obs["observation/image"] = np.zeros((2, config.image_size, config.image_size, 3), dtype=np.uint8)

    result = policy.infer(obs)

    assert result["actions"].shape == (2, config.action_horizon, config.action_dim)
    assert build_calls == [config]
    assert len(encoder.videos) == 1
    video = encoder.videos[0]
    assert tuple(video.shape) == (2, 3, 3, config.image_size, config.image_size)
    assert torch.allclose(video[:, :, 0], torch.full_like(video[:, :, 0], -1.0))
    assert torch.allclose(video[:, :, 1], torch.full_like(video[:, :, 1], -0.5))
    assert torch.allclose(video[:, :, 2], torch.full_like(video[:, :, 2], 0.5))
    assert len(idm.calls) == 1
    latents = idm.calls[0]["wan_vae_latents"]
    assert latents is not None
    assert tuple(latents.shape) == (2, 48, 1, 1, 1)
    assert latents.dtype == torch.float32
    assert torch.equal(latents, torch.full_like(latents, 7.0))
    assert policy.metadata["live_wan_vae_latents"] is True


def test_current_only_cached_wan_vae_idm_does_not_receive_live_latents(monkeypatch) -> None:
    config = _tiny_model_config(
        num_views=1,
        idm_arch="flow_transformer",
        idm_future_conditioning="current_only",
        idm_visual_encoder="wan_vae",
        wan_vae_use_cached_latents=True,
        wan_vae_repo_dir="/tmp/DiffSynth-Studio",
        wan_vae_checkpoint_path="/tmp/Wan2.2_VAE.pth",
    )

    def unexpected_build(model_config):
        del model_config
        raise AssertionError("current_only Wan-VAE IDMs must not build a live Wan VAE encoder")

    monkeypatch.setattr(swm, "build_frozen_wan_vae_encoder", unexpected_build)
    idm = _RecordingWanLatentIdm(config)
    policy = WorldModelPolicy(
        idm,
        config,
        image_keys=("observation/image",),
        device="cpu",
    )

    result = policy.infer(_make_single_view_obs(batch=1))

    assert result["actions"].shape == (1, config.action_horizon, config.action_dim)
    assert len(idm.calls) == 1
    assert idm.calls[0]["wan_vae_latents"] is None
    assert policy.metadata["live_wan_vae_latents"] is False


@pytest.mark.parametrize(
    "config",
    [
        _tiny_model_config(num_views=1, idm_visual_encoder="patch", wan_vae_use_cached_latents=True),
        _tiny_model_config(
            num_views=1,
            idm_arch="flow_transformer",
            idm_visual_encoder="wan_vae",
            wan_vae_use_cached_latents=False,
        ),
    ],
)
def test_policy_does_not_build_or_use_wan_vae_encoder_when_not_needed(monkeypatch, config) -> None:
    def unexpected_build(model_config):
        del model_config
        raise AssertionError("Wan VAE encoder should not be built for this config")

    monkeypatch.setattr(swm, "build_frozen_wan_vae_encoder", unexpected_build)
    idm = _RecordingWanLatentIdm(config)
    policy = WorldModelPolicy(
        idm,
        config,
        image_keys=("observation/image",),
        device="cpu",
    )

    result = policy.infer(_make_single_view_obs(batch=1))

    assert result["actions"].shape == (1, config.action_horizon, config.action_dim)
    assert idm.calls[0]["wan_vae_latents"] is None
    assert policy.metadata["live_wan_vae_latents"] is False


def test_cached_wan_vae_encoder_construction_errors_are_loud(monkeypatch) -> None:
    config = _tiny_model_config(
        num_views=1,
        idm_arch="flow_transformer",
        idm_visual_encoder="wan_vae",
        wan_vae_use_cached_latents=True,
        wan_vae_repo_dir="/tmp/DiffSynth-Studio",
        wan_vae_checkpoint_path="/tmp/Wan2.2_VAE.pth",
    )

    def failing_build(model_config):
        del model_config
        raise RuntimeError("boom loading Wan VAE")

    monkeypatch.setattr(swm, "build_frozen_wan_vae_encoder", failing_build)

    with pytest.raises(RuntimeError, match="boom loading Wan VAE"):
        WorldModelPolicy(
            _RecordingWanLatentIdm(config),
            config,
            image_keys=("observation/image",),
            device="cpu",
        )


def test_wan_prefix_action_expert_policy_unbatched_obs_returns_2d_chunk() -> None:
    policy, encoder = _wan_prefix_policy()
    obs = {
        "observation/image": np.zeros((16, 16, 3), dtype=np.uint8),
        "observation/state": np.zeros((4,), dtype=np.float32),
        "prompt": "reach the goal",
    }

    result = policy.infer(obs)

    assert result["actions"].shape == (2, 3)
    assert result["actions"].dtype == np.float32
    _assert_wan_prefix_timing_shape(result["server_timing"])
    assert len(encoder.calls) == 1
    assert tuple(encoder.calls[0]["current_images"].shape) == (1, 3, 16, 16)
    assert encoder.calls[0]["prompts"] == ["reach the goal"]


def test_wan_prefix_action_expert_policy_batched_obs_returns_3d_chunk() -> None:
    policy, encoder = _wan_prefix_policy()
    obs = _make_single_view_obs(batch=3, state_dim=4)

    result = policy.infer(obs)

    assert result["actions"].shape == (3, 2, 3)
    assert tuple(encoder.calls[0]["current_images"].shape) == (3, 3, 16, 16)
    assert encoder.calls[0]["prompts"] == ["reach the goal"] * 3
    metadata = policy.metadata
    assert metadata["policy"] == "pi05_wan_prefix_action_expert"
    assert metadata["action_horizon"] == 2
    assert metadata["action_dim"] == 3
    assert metadata["image_keys"] == ["observation/image"]
    assert metadata["state_key"] == "observation/state"
    assert metadata["checkpoint_path"] == "/tmp/fake_wan_pi05_action_expert.pt"
    assert metadata["wan_action_mode"] == "current_wan_prefix_action_expert"
    contract = metadata["wan_action_mode_contract"]
    assert contract["mode"] == "current_wan_prefix_action_expert"
    assert contract["runs_wan_generation"] is False
    assert contract["pi05_style_current_prefix_reuse"] is True
    assert contract["exposes_reusable_action_memory"] is True
    assert contract["native_wan_attention_kv_cache"] is False


def test_wan_prefix_action_expert_policy_requires_prompt() -> None:
    policy, _encoder = _wan_prefix_policy()
    obs = _make_single_view_obs(batch=1, state_dim=4)
    del obs["prompt"]

    with pytest.raises(KeyError, match="prompt"):
        policy.infer(obs)

    obs = _make_single_view_obs(batch=2, state_dim=4)
    obs["prompt"] = ["reach", ""]
    with pytest.raises(ValueError, match="non-empty prompt"):
        policy.infer(obs)


def test_wan_prefix_action_expert_policy_rejects_multi_image_keys() -> None:
    with pytest.raises(ValueError, match="exactly one image key"):
        WanPrefixActionExpertPolicy(
            _tiny_loaded_wan_action_expert(),
            _FakeWanPrefixEncoder(),
            image_keys=("observation/image", "observation/wrist_image"),
            image_size=16,
            device="cpu",
        )


def test_wan_prefix_action_expert_policy_rejects_non_current_mode_checkpoint() -> None:
    with pytest.raises(ValueError, match="current_wan_prefix_action_expert"):
        WanPrefixActionExpertPolicy(
            _tiny_loaded_wan_action_expert(wan_action_mode="partial_wan_prefix_action_expert"),
            _FakeWanPrefixEncoder(),
            image_size=16,
            device="cpu",
        )


def test_wan_prefix_action_expert_policy_reset_and_infer_many_are_compatible() -> None:
    policy, _encoder = _wan_prefix_policy()
    obs = _make_single_view_obs(batch=1, state_dim=4)

    policy.reset()
    policy.warmup_many(obs, batch_sizes=[1, 2])
    results = policy.infer_many([obs, obs])

    assert len(results) == 2
    assert all(result["actions"].shape == (1, 2, 3) for result in results)
    assert policy.new_history_state() is None


def test_wan_prefix_action_expert_policy_from_checkpoint_loads_tiny_checkpoint(tmp_path) -> None:
    model = WanPi05ActionExpert(**_tiny_wan_action_expert_kwargs()).eval()
    for parameter in model.parameters():
        parameter.data.zero_()
    checkpoint_path = tmp_path / "wan_pi05_action_expert.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_kwargs": _tiny_wan_action_expert_kwargs(),
            "args": {"wan_action_mode": "current_wan_prefix_action_expert"},
            "metrics": {"wan_action_mode": "current_wan_prefix_action_expert"},
            "action_normalization": {"enabled": False},
        },
        checkpoint_path,
    )
    encoder = _FakeWanPrefixEncoder()

    policy = WanPrefixActionExpertPolicy.from_checkpoint(
        checkpoint_path,
        prefix_encoder=encoder,
        image_size=16,
        device="cpu",
    )
    result = policy.infer(_make_single_view_obs(batch=2, state_dim=4))

    assert result["actions"].shape == (2, 2, 3)
    assert policy.metadata["checkpoint_path"] == str(checkpoint_path)
    assert policy.metadata["wan_action_mode"] == "current_wan_prefix_action_expert"


def test_main_decoded_policy_kind_calls_world_model_checkpoint(monkeypatch) -> None:
    captured: dict = {}
    policy = SimpleNamespace(metadata={"policy": "fake_decoded_video_idm"})

    def fake_from_checkpoint(cls, checkpoint_path, **kwargs):
        captured["checkpoint_path"] = checkpoint_path
        captured["kwargs"] = kwargs
        return policy

    def fake_run_websocket_server(served_policy, **kwargs):
        captured["served_policy"] = served_policy
        captured["server_kwargs"] = kwargs

    monkeypatch.setattr(WorldModelPolicy, "from_checkpoint", classmethod(fake_from_checkpoint))
    monkeypatch.setattr(swm, "run_websocket_server", fake_run_websocket_server)

    main(
        Args(
            policy_kind="decoded_video_idm",
            idm_checkpoint="/tmp/idm.pt",
            allow_repeat_current=True,
            device="cpu",
            flow_seed=123,
        )
    )

    assert captured["checkpoint_path"] == "/tmp/idm.pt"
    assert captured["kwargs"] == {
        "image_keys": ("observation/image",),
        "state_key": "observation/state",
        "prompt_key": "prompt",
        "future_provider": "repeat_current",
        "future_provider_kwargs": {},
        "device": "cpu",
        "flow_seed": 123,
    }
    assert captured["served_policy"] is policy
    assert captured["server_kwargs"]["metadata"] == {"policy": "fake_decoded_video_idm"}


def test_main_prefix_policy_kind_builds_encoder_and_action_expert(monkeypatch) -> None:
    captured: dict = {}
    encoder = _FakeWanPrefixEncoder(prefix_dim=12)
    policy = SimpleNamespace(metadata={"policy": "fake_prefix_action_expert"})

    def fake_build_wan_prefix_encoder(config):
        captured["encoder_config"] = config
        return encoder

    def fake_from_checkpoint(cls, checkpoint_path, **kwargs):
        captured["checkpoint_path"] = checkpoint_path
        captured["kwargs"] = kwargs
        return policy

    def fake_run_websocket_server(served_policy, **kwargs):
        captured["served_policy"] = served_policy
        captured["server_kwargs"] = kwargs

    monkeypatch.setattr(swm, "_pi05_checkpoint_prefix_dim", lambda checkpoint_path: 12)
    monkeypatch.setattr(swm, "build_wan_prefix_encoder", fake_build_wan_prefix_encoder)
    monkeypatch.setattr(WanPrefixActionExpertPolicy, "from_checkpoint", classmethod(fake_from_checkpoint))
    monkeypatch.setattr(swm, "run_websocket_server", fake_run_websocket_server)

    main(
        Args(
            policy_kind="current_wan_prefix_action_expert",
            pi05_checkpoint="/tmp/pi05.pt",
            image_keys=("observation/image",),
            pi05_image_size=96,
            pi05_num_steps=7,
            pi05_action_seed=42,
            prefix_backend="dit_hidden",
            wan_repo_dir="/fake/diffsynth",
            wan_checkpoint_dir="/fake/wan",
            wan_vae_checkpoint_path="/fake/vae.pth",
            wan_text_encoder_checkpoint_path="/fake/t5.pth",
            wan_tokenizer_dir="/fake/tokenizer",
            wan_dtype="float32",
            wan_tiled=True,
            dit_selected_layers=(1, 3),
            dit_hidden_pool="token_pool",
            dit_tokens_per_layer=4,
            dit_timestep=250.0,
            device="cuda:1",
        )
    )

    config = captured["encoder_config"]
    assert config.prefix_backend == "dit_hidden"
    assert config.prefix_dim == 12
    assert config.wan_repo_dir == "/fake/diffsynth"
    assert config.wan_checkpoint_dir == "/fake/wan"
    assert config.wan_vae_checkpoint_path == "/fake/vae.pth"
    assert config.wan_text_encoder_checkpoint_path == "/fake/t5.pth"
    assert config.wan_tokenizer_dir == "/fake/tokenizer"
    assert config.wan_dtype == "float32"
    assert config.wan_tiled is True
    assert config.dit_selected_layers == (1, 3)
    assert config.dit_hidden_pool == "token_pool"
    assert config.dit_tokens_per_layer == 4
    assert config.dit_num_latent_frames == 1
    assert config.dit_timestep == 250.0
    assert captured["checkpoint_path"] == "/tmp/pi05.pt"
    assert captured["kwargs"] == {
        "prefix_encoder": encoder,
        "image_keys": ("observation/image",),
        "state_key": "observation/state",
        "prompt_key": "prompt",
        "image_size": 96,
        "device": "cuda:1",
        "num_steps": 7,
        "action_seed": 42,
    }
    assert captured["served_policy"] is policy
    assert captured["server_kwargs"]["metadata"] == {"policy": "fake_prefix_action_expert"}


def test_main_prefix_policy_kind_requires_pi05_checkpoint() -> None:
    args = Args(policy_kind="current_wan_prefix_action_expert", image_keys=("observation/image",))

    with pytest.raises(ValueError, match="pi05-checkpoint"):
        main(args)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"idm_checkpoint": "/tmp/idm.pt"}, "does not load --idm-checkpoint"),
        ({"future_provider": "wan_lora"}, "does not use --future-provider"),
        ({"wan_lora_path": "/tmp/epoch-0.safetensors"}, "does not use Wan LoRA"),
        ({"image_keys": ("observation/image", "observation/wrist_image")}, "exactly one"),
    ],
)
def test_main_prefix_policy_kind_rejects_non_prefix_args(overrides, match) -> None:
    kwargs = {
        "policy_kind": "current_wan_prefix_action_expert",
        "pi05_checkpoint": "/tmp/pi05.pt",
        "image_keys": ("observation/image",),
    }
    kwargs.update(overrides)
    args = Args(**kwargs)

    with pytest.raises(ValueError, match=match):
        swm._validate_current_wan_prefix_args(args)


def test_prefix_server_rejects_partial_dit_future_slot_config() -> None:
    args = Args(
        policy_kind="current_wan_prefix_action_expert",
        pi05_checkpoint="/tmp/pi05.pt",
        image_keys=("observation/image",),
        prefix_backend="dit_hidden",
        dit_num_latent_frames=2,
    )

    with pytest.raises(ValueError, match="dit_num_latent_frames=1"):
        swm._wan_prefix_encoder_config_from_args(args, prefix_dim=8)


def test_main_decoded_policy_kind_rejects_prefix_only_args() -> None:
    args = Args(
        policy_kind="decoded_video_idm",
        idm_checkpoint="/tmp/idm.pt",
        allow_repeat_current=True,
        prefix_backend="raw_current",
    )

    with pytest.raises(ValueError, match="prefix args"):
        main(args)


def test_server_raw_current_prefix_encoder_smoke_without_wan_files() -> None:
    encoder = swm.build_wan_prefix_encoder(swm.WanPrefixEncoderConfig(prefix_dim=8, prefix_backend="raw_current"))

    prefix = encoder.encode_prefix(torch.zeros(2, 3, 16, 16), ["reach", "push"])

    assert prefix.shape == (2, 3, 8)
    assert torch.isfinite(prefix).all()


def test_infer_returns_batched_action_chunk_for_metaworld_style_obs() -> None:
    config = _tiny_model_config()
    policy = _build_policy(config)

    result = policy.infer(_make_obs(batch=3))

    actions = result["actions"]
    assert isinstance(actions, np.ndarray)
    assert actions.dtype == np.float32
    # MetaWorld driver asserts ndim == 3: (batch, action_horizon, action_dim).
    assert actions.ndim == 3
    assert actions.shape == (3, config.action_horizon, config.action_dim)
    _assert_server_timing_shape(result["server_timing"])


def test_infer_missing_required_key_fails_loudly() -> None:
    policy = _build_policy()
    obs = _make_obs()
    del obs["observation/state"]

    with pytest.raises(KeyError) as excinfo:
        policy.infer(obs)
    assert "observation/state" in str(excinfo.value)


def test_infer_state_dimension_mismatch_fails_loudly() -> None:
    policy = _build_policy(_tiny_model_config(state_dim=4))
    obs = _make_obs()
    obs["observation/state"] = np.zeros((2, 6), dtype=np.float32)

    with pytest.raises(ValueError) as excinfo:
        policy.infer(obs)
    assert "state" in str(excinfo.value).lower()


def test_constructing_policy_with_wrong_view_count_fails_loudly() -> None:
    config = _tiny_model_config(num_views=2)
    idm = InverseDynamicsModel(config).eval()

    with pytest.raises(ValueError) as excinfo:
        WorldModelPolicy(idm, config, image_keys=("observation/image",))
    assert "num_views" in str(excinfo.value) or "image_keys" in str(excinfo.value)


def test_infer_single_unbatched_observation_returns_2d_chunk() -> None:
    config = _tiny_model_config()
    policy = _build_policy(config)
    obs = {
        "observation/image": np.zeros((config.image_size, config.image_size, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((config.image_size, config.image_size, 3), dtype=np.uint8),
        "observation/state": np.zeros((config.state_dim,), dtype=np.float32),
        "prompt": "reach the goal",
    }

    actions = policy.infer(obs)["actions"]

    assert actions.ndim == 2
    assert actions.shape == (config.action_horizon, config.action_dim)


def test_action_normalizer_is_applied_to_outputs() -> None:
    config = _tiny_model_config()
    idm = InverseDynamicsModel(config).eval()
    obs = _make_obs(batch=2)

    raw = WorldModelPolicy(idm, config, image_keys=DEFAULT_IMAGE_KEYS).infer(obs)["actions"]
    normalizer = ActionNormalizer(mean=torch.full((config.action_dim,), 10.0), std=torch.ones(config.action_dim))
    normed = WorldModelPolicy(idm, config, image_keys=DEFAULT_IMAGE_KEYS, action_normalizer=normalizer).infer(obs)[
        "actions"
    ]

    # denormalize(x) = x * std + mean; with std=1, mean=10 the outputs shift by +10.
    assert np.allclose(normed, raw + 10.0, atol=1e-4)


def test_state_normalizer_is_applied_once_to_policy_inputs() -> None:
    class RecordingIdm(torch.nn.Module):
        def __init__(self, config: ModelConfig) -> None:
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(()))
            self.config = config
            self.seen_state: torch.Tensor | None = None

        def forward(self, current_images, future_images, state, task_id, *, sample_noise=None):
            del future_images, task_id, sample_noise
            self.seen_state = state.detach().cpu()
            return torch.zeros(
                current_images.shape[0],
                self.config.action_horizon,
                self.config.action_dim,
                device=current_images.device,
            )

    config = _tiny_model_config()
    idm = RecordingIdm(config).eval()
    normalizer = StateNormalizer(
        mean=torch.tensor([10.0, 20.0, 10.0, 20.0]),
        std=torch.tensor([2.0, 5.0, 2.0, 10.0]),
    )
    attach_state_normalizer(idm, normalizer, normalize_forward=True)
    policy = WorldModelPolicy(
        idm,
        config,
        image_keys=DEFAULT_IMAGE_KEYS,
        state_normalizer=normalizer,
        device="cpu",
    )
    obs = _make_obs(batch=2)
    obs["observation/state"] = np.array(
        [
            [12.0, 10.0, 14.0, 40.0],
            [8.0, 25.0, 6.0, 0.0],
        ],
        dtype=np.float32,
    )
    expected_state = torch.tensor(
        [
            [1.0, -2.0, 2.0, 2.0],
            [-1.0, 1.0, -2.0, -2.0],
        ]
    )

    policy.infer(obs)

    assert idm.seen_state is not None
    assert torch.allclose(idm.seen_state, expected_state)


def test_from_checkpoint_loads_idm_and_normalizer(tmp_path) -> None:
    config = _tiny_model_config()
    idm = InverseDynamicsModel(config)
    normalizer = ActionNormalizer(mean=torch.zeros(config.action_dim), std=torch.ones(config.action_dim))
    checkpoint_path = tmp_path / "idm.pt"
    save_idm_state_checkpoint(
        checkpoint_path,
        idm_state=module_state_dict_for_checkpoint(idm),
        model_config=config,
        train_config=TrainConfig(),
        metrics={},
        action_normalizer=normalizer,
    )

    policy = WorldModelPolicy.from_checkpoint(checkpoint_path, image_keys=DEFAULT_IMAGE_KEYS, device="cpu")
    result = policy.infer(_make_obs(batch=1))

    assert result["actions"].shape == (1, config.action_horizon, config.action_dim)
    assert policy.action_normalizer is not None


def test_from_checkpoint_accepts_history_conditioned_idm_checkpoint(tmp_path) -> None:
    config = _tiny_model_config(
        num_views=1,
        idm_arch="flow_transformer",
        idm_history_length=2,
        idm_transformer_patch_size=16,
        idm_transformer_heads=8,
    )
    idm = InverseDynamicsModel(config)
    checkpoint_path = tmp_path / "history_idm.pt"
    save_idm_state_checkpoint(
        checkpoint_path,
        idm_state=module_state_dict_for_checkpoint(idm),
        model_config=config,
        train_config=TrainConfig(),
        metrics={},
    )

    policy = WorldModelPolicy.from_checkpoint(
        checkpoint_path,
        image_keys=("observation/image",),
        device="cpu",
    )
    result = policy.infer(_make_single_view_obs(batch=1))

    assert policy.metadata["idm_history_length"] == 2
    assert result["actions"].shape == (1, config.action_horizon, config.action_dim)


def test_from_checkpoint_defaults_wan_lora_device_to_resolved_policy_device(monkeypatch) -> None:
    captured: dict = {}
    config = _tiny_model_config(num_views=1)

    def fake_load_idm_checkpoint(path, device):
        del path
        captured["load_device"] = device
        return InverseDynamicsModel(config), config

    def fake_get_action_normalizer(idm, device):
        del idm
        captured["normalizer_device"] = device
        return None

    def fake_build_future_provider(name, **kwargs):
        captured["provider_name"] = name
        captured["provider_kwargs"] = kwargs
        return RepeatCurrentFutureProvider()

    class CapturePolicy(WorldModelPolicy):
        def __init__(self, *args, **kwargs):
            del args
            captured["policy_device"] = kwargs["device"]

    monkeypatch.setattr(swm, "load_idm_checkpoint", fake_load_idm_checkpoint)
    monkeypatch.setattr(swm, "get_action_normalizer", fake_get_action_normalizer)
    monkeypatch.setattr(swm, "load_idm_training_frame_delta", lambda path: 4)
    monkeypatch.setattr(swm, "build_future_provider", fake_build_future_provider)

    CapturePolicy.from_checkpoint(
        "/tmp/idm.pt",
        future_provider="wan_lora",
        future_provider_kwargs=_wan_lora_checkpoint_kwargs(),
        device="cuda:1",
    )

    assert captured["load_device"] == torch.device("cuda:1")
    assert captured["normalizer_device"] == torch.device("cuda:1")
    assert captured["policy_device"] == torch.device("cuda:1")
    assert captured["provider_name"] == "wan_lora"
    assert captured["provider_kwargs"]["wan_lora_device"] == "cuda:1"
    assert captured["provider_kwargs"]["frame_delta"] == 4


def test_from_checkpoint_rejects_missing_wan_lora_training_frame_delta(monkeypatch) -> None:
    config = _tiny_model_config(num_views=1)

    def fake_load_idm_checkpoint(path, device):
        del path, device
        return InverseDynamicsModel(config), config

    def fake_build_future_provider(name, **kwargs):
        del name, kwargs
        raise AssertionError("wan_lora provider should not be built without checkpoint frame_delta")

    monkeypatch.setattr(swm, "load_idm_checkpoint", fake_load_idm_checkpoint)
    monkeypatch.setattr(swm, "get_action_normalizer", lambda idm, device: None)
    monkeypatch.setattr(swm, "load_idm_training_frame_delta", lambda path: None)
    monkeypatch.setattr(swm, "build_future_provider", fake_build_future_provider)

    with pytest.raises(ValueError, match="training frame_delta"):
        WorldModelPolicy.from_checkpoint(
            "/tmp/idm.pt",
            future_provider="wan_lora",
            future_provider_kwargs=_wan_lora_checkpoint_kwargs(),
            device="cpu",
        )


def test_from_checkpoint_propagates_wan_lora_training_frame_delta(monkeypatch) -> None:
    captured: dict = {}
    config = _tiny_model_config(num_views=1)

    def fake_load_idm_checkpoint(path, device):
        del path, device
        return InverseDynamicsModel(config), config

    def fake_build_future_provider(name, **kwargs):
        captured["provider_name"] = name
        captured["provider_kwargs"] = kwargs
        return RepeatCurrentFutureProvider()

    monkeypatch.setattr(swm, "load_idm_checkpoint", fake_load_idm_checkpoint)
    monkeypatch.setattr(swm, "get_action_normalizer", lambda idm, device: None)
    monkeypatch.setattr(swm, "load_idm_training_frame_delta", lambda path: 4)
    monkeypatch.setattr(swm, "build_future_provider", fake_build_future_provider)

    WorldModelPolicy.from_checkpoint(
        "/tmp/idm.pt",
        future_provider="wan_lora",
        future_provider_kwargs=_wan_lora_checkpoint_kwargs(),
        device="cpu",
    )

    assert captured["provider_name"] == "wan_lora"
    assert captured["provider_kwargs"]["frame_delta"] == 4


def test_from_checkpoint_keeps_explicit_wan_lora_device_override(monkeypatch) -> None:
    captured: dict = {}
    config = _tiny_model_config(num_views=1)

    def fake_load_idm_checkpoint(path, device):
        del path, device
        return InverseDynamicsModel(config), config

    def fake_build_future_provider(name, **kwargs):
        del name
        captured["provider_kwargs"] = kwargs
        return RepeatCurrentFutureProvider()

    class CapturePolicy(WorldModelPolicy):
        def __init__(self, *args, **kwargs):
            del args, kwargs

    monkeypatch.setattr(swm, "load_idm_checkpoint", fake_load_idm_checkpoint)
    monkeypatch.setattr(swm, "get_action_normalizer", lambda idm, device: None)
    monkeypatch.setattr(swm, "load_idm_training_frame_delta", lambda path: 4)
    monkeypatch.setattr(swm, "build_future_provider", fake_build_future_provider)

    CapturePolicy.from_checkpoint(
        "/tmp/idm.pt",
        future_provider="wan_lora",
        future_provider_kwargs=_wan_lora_checkpoint_kwargs(wan_lora_device="cpu"),
        device="cuda:1",
    )

    assert captured["provider_kwargs"]["wan_lora_device"] == "cpu"
    assert captured["provider_kwargs"]["frame_delta"] == 4


def test_custom_future_provider_is_injected_and_receives_current_frames() -> None:
    config = _tiny_model_config()
    idm = InverseDynamicsModel(config).eval()
    calls: dict = {}

    def stub_provider(current_images, *, num_future_frames, prompts=None):
        calls["shape"] = tuple(current_images.shape)
        calls["num_future_frames"] = num_future_frames
        calls["prompts"] = prompts
        return current_images.unsqueeze(1).repeat(1, num_future_frames, 1, 1, 1, 1)

    policy = WorldModelPolicy(idm, config, image_keys=DEFAULT_IMAGE_KEYS, future_provider=stub_provider)
    policy.infer(_make_obs(batch=2))

    assert calls["shape"] == (2, config.num_views, 3, config.image_size, config.image_size)
    assert calls["num_future_frames"] == config.num_future_frames
    assert calls["prompts"] == ["reach the goal", "reach the goal"]


def test_build_future_provider_repeat_current_and_errors() -> None:
    provider = build_future_provider("repeat_current")
    current = torch.rand(1, 2, 3, 8, 8)
    assert provider(current, num_future_frames=1).shape == (1, 1, 2, 3, 8, 8)

    with pytest.raises(ValueError, match="diffsynth-repo-dir"):
        build_future_provider("wan_lora", image_size=16)
    with pytest.raises(ValueError):
        build_future_provider("not-a-provider")


def test_infer_many_returns_list_of_results() -> None:
    policy = _build_policy()

    results = policy.infer_many([_make_obs(batch=1), _make_obs(batch=1)])

    assert isinstance(results, list)
    assert len(results) == 2
    for result in results:
        assert result["actions"].shape[0] == 1


def test_infer_with_flow_transformer_idm_is_deterministic() -> None:
    config = _tiny_model_config(
        idm_arch="flow_transformer",
        idm_transformer_patch_size=16,
        idm_transformer_heads=8,
        idm_flow_sampling_steps=2,
    )
    policy = _build_policy(config, flow_seed=0)
    obs = _make_obs(batch=2)

    first = policy.infer(obs)["actions"]
    second = policy.infer(obs)["actions"]

    assert first.shape == (2, config.action_horizon, config.action_dim)
    assert np.allclose(first, second)


# --------------------------------------------------------------------------------------
# Fix #1: default image_keys must match the default one-view IDM (corner4 -> image).
# --------------------------------------------------------------------------------------
def test_default_image_keys_is_single_corner4_view() -> None:
    # The default IDM (train_idm.py) is trained with image_keys=("corner4.image",), i.e.
    # num_views=1, and examples/metaworld/main.py sends corner4 as "observation/image".
    assert swm.DEFAULT_IMAGE_KEYS == ("observation/image",)


def test_policy_with_defaults_serves_one_view_idm() -> None:
    # Constructing the policy with the default image_keys must work against the default
    # one-view IDM and consume only observation/image.
    policy = _single_view_policy()  # no explicit image_keys -> uses module default

    result = policy.infer(_make_single_view_obs(batch=2))

    assert result["actions"].shape == (2, 4, 4)
    assert policy.metadata["image_keys"] == ["observation/image"]
    assert policy.metadata["wan_action_mode"] == "decoded_video_idm"
    contract = policy.metadata["wan_action_mode_contract"]
    assert contract["mode"] == "decoded_video_idm"
    assert contract["runs_wan_generation"] is True
    assert contract["generates_video"] is True
    assert contract["consumes_future_pixels"] is True
    assert contract["native_wan_attention_kv_cache"] is False
    assert policy.metadata["future_provider"] == "repeat_current"
    assert policy.metadata["future_provider_smoke"] is True


def test_one_view_policy_ignores_extra_wrist_image_key() -> None:
    # The MetaWorld driver also sends observation/wrist_image; a one-view policy must
    # ignore it rather than fail.
    policy = _single_view_policy()
    obs = _make_single_view_obs(batch=2)
    obs["observation/wrist_image"] = np.zeros((2, 16, 16, 3), dtype=np.uint8)

    result = policy.infer(obs)

    assert result["actions"].shape == (2, 4, 4)


# --------------------------------------------------------------------------------------
# Fix #2: non-scalar prompts must match the batch size and fail loudly otherwise.
# --------------------------------------------------------------------------------------
def test_prompt_list_length_mismatch_fails_loudly() -> None:
    policy = _build_policy()
    obs = _make_obs(batch=2)
    obs["prompt"] = ["a", "b", "c"]  # length 3 != batch 2

    with pytest.raises(ValueError) as excinfo:
        policy.infer(obs)
    message = str(excinfo.value).lower()
    assert "prompt" in message
    assert "3" in message and "2" in message


def test_prompt_list_matching_length_is_accepted() -> None:
    policy = _build_policy()
    obs = _make_obs(batch=2)
    obs["prompt"] = ["left", "right"]

    result = policy.infer(obs)

    assert result["actions"].shape[0] == 2


def test_scalar_prompt_is_broadcast_to_batch() -> None:
    assert swm._normalize_prompts("reach", 3) == ["reach", "reach", "reach"]
    assert swm._normalize_prompts(None, 3) is None


# --------------------------------------------------------------------------------------
# Fix #3: future_provider output is validated at the server boundary before the IDM.
# --------------------------------------------------------------------------------------
def _future_provider_returning(value):
    def provider(current_images, *, num_future_frames, prompts=None):
        del current_images, num_future_frames, prompts
        return value

    return provider


def _good_future(batch: int = 2, num_future_frames: int = 1, num_views: int = 2, size: int = 16) -> torch.Tensor:
    return torch.rand(batch, num_future_frames, num_views, 3, size, size)


def test_future_provider_non_tensor_output_fails_loudly() -> None:
    bad = _good_future().numpy()  # ndarray, not a torch.Tensor
    policy = _build_policy(future_provider=_future_provider_returning(bad))

    with pytest.raises((TypeError, ValueError)) as excinfo:
        policy.infer(_make_obs(batch=2))
    assert "tensor" in str(excinfo.value).lower()


def test_future_provider_wrong_shape_output_fails_loudly() -> None:
    bad = torch.rand(2, 1, 2, 3, 8, 8)  # wrong image_size (8 != 16)
    policy = _build_policy(future_provider=_future_provider_returning(bad))

    with pytest.raises(ValueError) as excinfo:
        policy.infer(_make_obs(batch=2))
    assert "shape" in str(excinfo.value).lower()


def test_future_provider_non_floating_dtype_fails_loudly() -> None:
    bad = torch.zeros(2, 1, 2, 3, 16, 16, dtype=torch.int64)
    policy = _build_policy(future_provider=_future_provider_returning(bad))

    with pytest.raises(ValueError) as excinfo:
        policy.infer(_make_obs(batch=2))
    assert "float" in str(excinfo.value).lower() or "dtype" in str(excinfo.value).lower()


def test_future_provider_non_finite_output_fails_loudly() -> None:
    bad = _good_future()
    bad[0, 0, 0, 0, 0, 0] = float("nan")
    policy = _build_policy(future_provider=_future_provider_returning(bad))

    with pytest.raises(ValueError) as excinfo:
        policy.infer(_make_obs(batch=2))
    assert "finite" in str(excinfo.value).lower() or "nan" in str(excinfo.value).lower()


def test_future_provider_out_of_range_output_fails_loudly() -> None:
    bad = _good_future() + 1.5  # values exceed 1.0
    policy = _build_policy(future_provider=_future_provider_returning(bad))

    with pytest.raises(ValueError) as excinfo:
        policy.infer(_make_obs(batch=2))
    assert "[0, 1]" in str(excinfo.value) or "range" in str(excinfo.value).lower()


# --------------------------------------------------------------------------------------
# Fix #4: future_provider in metadata + explicit (non-silent) repeat_current.
# --------------------------------------------------------------------------------------
def test_metadata_reports_repeat_current_provider_by_default() -> None:
    policy = _build_policy()
    assert policy.metadata["future_provider"] == "repeat_current"


def test_metadata_reports_custom_future_provider_name() -> None:
    def my_wan_provider(current_images, *, num_future_frames, prompts=None):
        return current_images.unsqueeze(1).repeat(1, num_future_frames, 1, 1, 1, 1)

    policy = _build_policy(future_provider=my_wan_provider)
    assert policy.metadata["future_provider"] == "my_wan_provider"


def test_args_allow_repeat_current_defaults_to_false() -> None:
    assert Args.allow_repeat_current is False


def test_main_rejects_repeat_current_without_allow_flag() -> None:
    # The CLI must NOT silently serve the non-physical repeat_current provider. The gate
    # fires before any checkpoint loading, so a missing checkpoint path is irrelevant.
    args = Args(idm_checkpoint="/nonexistent/idm.pt")  # future_provider defaults to repeat_current

    with pytest.raises(ValueError) as excinfo:
        main(args)
    assert "repeat_current" in str(excinfo.value).lower()
    assert "allow-repeat-current" in str(excinfo.value).lower()


def test_main_rejects_wan_lora_missing_paths_before_checkpoint_load() -> None:
    args = Args(idm_checkpoint="/nonexistent/idm.pt", future_provider="wan_lora")

    with pytest.raises(ValueError, match="diffsynth-repo-dir"):
        main(args)


def test_require_explicit_repeat_current_allows_flag_and_other_providers() -> None:
    # Explicit opt-in is allowed (tests / smoke runs set it explicitly).
    swm._require_explicit_repeat_current("repeat_current", allow_repeat_current=True)
    # Real providers never require the smoke flag.
    swm._require_explicit_repeat_current("wan_lora", allow_repeat_current=False)


# --------------------------------------------------------------------------------------
# Fix #5: end-to-end websocket round-trip with the real openpi_client.
# --------------------------------------------------------------------------------------
def _start_server_on_ephemeral_port(policy: WorldModelPolicy):
    """Start ``policy``'s websocket server on an OS-assigned port in a daemon thread.

    Returns ``(port, thread)``. The thread is a daemon so it does not block teardown.
    """
    import threading

    ready = threading.Event()
    state: dict = {}

    def _on_ready(port: int) -> None:
        state["port"] = port
        ready.set()

    def _serve() -> None:
        run_websocket_server(policy, host="127.0.0.1", port=0, metadata=policy.metadata, on_ready=_on_ready)

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    if not ready.wait(timeout=15):
        raise TimeoutError("World-model websocket server did not start within 15s.")
    return state["port"], thread


def test_websocket_roundtrip_metadata_actions_and_error_propagation() -> None:
    # Lazy import: only needed for the integration path (matches server's lazy deps).
    from openpi_client import websocket_client_policy as _wcp

    config = _tiny_model_config(num_views=1)
    idm = InverseDynamicsModel(config).eval()
    policy = WorldModelPolicy(idm, config, image_keys=("observation/image",))

    port, _thread = _start_server_on_ephemeral_port(policy)

    client = _wcp.WebsocketClientPolicy("127.0.0.1", port)

    # 1) Metadata handshake.
    metadata = client.get_server_metadata()
    assert metadata["policy"] == "world_model_idm"
    assert metadata["future_provider"] == "repeat_current"
    assert metadata["image_keys"] == ["observation/image"]

    # 2) Batched action shape and server timing round-trip through msgpack.
    result = client.infer(_make_single_view_obs(batch=3))
    actions = result["actions"]
    assert actions.shape == (3, config.action_horizon, config.action_dim)
    timing = result["server_timing"]
    assert isinstance(timing, dict)
    _assert_server_timing_shape(timing)

    # 3) Server-side errors propagate to the client as a raised error.
    bad_obs = _make_single_view_obs(batch=2)
    del bad_obs["observation/state"]
    with pytest.raises(RuntimeError) as excinfo:
        client.infer(bad_obs)
    assert "observation/state" in str(excinfo.value)


# --------------------------------------------------------------------------------------
# History-conditioned IDM serving (idm_history_length > 0).
# --------------------------------------------------------------------------------------
def test_metadata_reports_zero_history_length_for_default_idm() -> None:
    policy = _build_policy()
    assert policy.metadata["idm_history_length"] == 0


def test_history_first_request_passes_zero_history_with_zero_mask() -> None:
    config = _history_model_config(history_length=2)
    policy, idm = _history_policy_with_capturing_idm(config)

    policy.infer(_make_single_view_obs(batch=2))

    assert len(idm.calls) == 1
    call = idm.calls[0]
    assert call["prev_state_history"].shape == (2, 2, config.state_dim)
    assert call["prev_action_history"].shape == (2, 2, config.action_dim)
    assert call["history_mask"].shape == (2, 2)
    # First request: nothing buffered yet -> all-zero history tensors and zero mask.
    assert torch.count_nonzero(call["history_mask"]) == 0
    assert torch.count_nonzero(call["prev_state_history"]) == 0
    assert torch.count_nonzero(call["prev_action_history"]) == 0


def test_history_rolls_forward_with_prev_state_and_first_action() -> None:
    config = _history_model_config(history_length=2)
    policy, idm = _history_policy_with_capturing_idm(config)

    obs = _make_single_view_obs(batch=1)
    obs["observation/state"] = np.full((1, config.state_dim), 1.0, dtype=np.float32)
    policy.infer(obs)  # call 0 -> first action 0.0
    obs["observation/state"] = np.full((1, config.state_dim), 2.0, dtype=np.float32)
    policy.infer(obs)  # call 1 -> first action 10.0
    obs["observation/state"] = np.full((1, config.state_dim), 3.0, dtype=np.float32)
    policy.infer(obs)  # call 2

    # Call 1 sees only the call-0 step in the newest (last) slot.
    call1 = idm.calls[1]
    assert torch.equal(call1["history_mask"], torch.tensor([[0.0, 1.0]]))
    assert torch.allclose(call1["prev_state_history"][0, -1], torch.full((config.state_dim,), 1.0))
    assert torch.allclose(call1["prev_action_history"][0, -1], torch.zeros(config.action_dim))

    # Call 2 sees [call0, call1] ordered oldest -> newest, both valid.
    call2 = idm.calls[2]
    assert torch.equal(call2["history_mask"], torch.tensor([[1.0, 1.0]]))
    assert torch.allclose(call2["prev_state_history"][0, 0], torch.full((config.state_dim,), 1.0))
    assert torch.allclose(call2["prev_state_history"][0, 1], torch.full((config.state_dim,), 2.0))
    # Buffered actions are the FIRST action of each chunk (0.0 then 10.0), not later steps.
    assert torch.allclose(call2["prev_action_history"][0, 0], torch.zeros(config.action_dim))
    assert torch.allclose(call2["prev_action_history"][0, 1], torch.full((config.action_dim,), 10.0))


def test_reset_clears_history_state() -> None:
    config = _history_model_config(history_length=2)
    policy, idm = _history_policy_with_capturing_idm(config)

    policy.infer(_make_single_view_obs(batch=1))
    policy.infer(_make_single_view_obs(batch=1))
    assert torch.count_nonzero(idm.calls[-1]["history_mask"]) > 0

    policy.reset()
    policy.infer(_make_single_view_obs(batch=1))
    assert torch.count_nonzero(idm.calls[-1]["history_mask"]) == 0
    assert torch.count_nonzero(idm.calls[-1]["prev_state_history"]) == 0


def test_batched_history_maintains_independent_per_row_state() -> None:
    config = _history_model_config(history_length=2)
    policy, idm = _history_policy_with_capturing_idm(config)

    obs = _make_single_view_obs(batch=2)
    obs["observation/state"] = np.array([[1, 1, 1, 1], [5, 5, 5, 5]], dtype=np.float32)
    policy.infer(obs)
    obs = _make_single_view_obs(batch=2)
    obs["observation/state"] = np.array([[2, 2, 2, 2], [6, 6, 6, 6]], dtype=np.float32)
    policy.infer(obs)

    call1 = idm.calls[1]
    # Each row's newest history entry is that row's own previous observation state.
    assert torch.allclose(call1["prev_state_history"][0, -1], torch.full((config.state_dim,), 1.0))
    assert torch.allclose(call1["prev_state_history"][1, -1], torch.full((config.state_dim,), 5.0))


def test_supplied_history_overrides_server_side_fallback() -> None:
    config = _history_model_config(history_length=2)
    policy, idm = _history_policy_with_capturing_idm(config)

    obs = _make_single_view_obs(batch=1)
    obs["observation/state"] = np.full((1, config.state_dim), 3.0, dtype=np.float32)
    policy.infer(obs)

    supplied_state_history = np.full((1, 2, config.state_dim), 7.0, dtype=np.float32)
    supplied_action_history = np.full((1, 2, config.action_dim), 8.0, dtype=np.float32)
    supplied_mask = np.array([[1.0, 0.0]], dtype=np.float32)
    obs = _make_single_view_obs(batch=1)
    obs.update(
        {
            "prev_state_history": supplied_state_history,
            "prev_action_history": supplied_action_history,
            "history_mask": supplied_mask,
        }
    )

    policy.infer(obs)

    call = idm.calls[-1]
    assert torch.allclose(call["prev_state_history"], torch.from_numpy(supplied_state_history))
    assert torch.allclose(call["prev_action_history"], torch.from_numpy(supplied_action_history))
    assert torch.allclose(call["history_mask"], torch.from_numpy(supplied_mask))


def test_supplied_history_does_not_update_server_side_fallback() -> None:
    config = _history_model_config(history_length=2)
    policy, idm = _history_policy_with_capturing_idm(config)
    supplied_state_history = np.full((1, 2, config.state_dim), 7.0, dtype=np.float32)
    supplied_action_history = np.full((1, 2, config.action_dim), 8.0, dtype=np.float32)
    supplied_mask = np.ones((1, 2), dtype=np.float32)
    obs = _make_single_view_obs(batch=1)
    obs.update(
        {
            "prev_state_history": supplied_state_history,
            "prev_action_history": supplied_action_history,
            "history_mask": supplied_mask,
        }
    )

    policy.infer(obs)
    policy.infer(_make_single_view_obs(batch=1))

    # The second request uses server-side fallback. Since the first request supplied
    # authoritative client history, it must not have advanced the fallback buffer.
    call = idm.calls[-1]
    assert torch.count_nonzero(call["history_mask"]) == 0
    assert torch.count_nonzero(call["prev_state_history"]) == 0
    assert torch.count_nonzero(call["prev_action_history"]) == 0


def test_supplied_history_shape_validation_fails_loudly() -> None:
    config = _history_model_config(history_length=2)
    policy, _idm = _history_policy_with_capturing_idm(config)
    obs = _make_single_view_obs(batch=2)
    obs.update(
        {
            "prev_state_history": np.zeros((2, 2, config.state_dim), dtype=np.float32),
            "prev_action_history": np.zeros((2, 1, config.action_dim), dtype=np.float32),
            "history_mask": np.zeros((2, 2), dtype=np.float32),
        }
    )

    with pytest.raises(ValueError, match="prev_action_history.*shape"):
        policy.infer(obs)


def test_unbatched_observation_rejects_batched_supplied_history() -> None:
    config = _history_model_config(history_length=2)
    policy, _idm = _history_policy_with_capturing_idm(config)
    obs = {
        "observation/image": np.zeros((config.image_size, config.image_size, 3), dtype=np.uint8),
        "observation/state": np.zeros((config.state_dim,), dtype=np.float32),
        "prompt": "reach the goal",
        "prev_state_history": np.zeros((1, 2, config.state_dim), dtype=np.float32),
        "prev_action_history": np.zeros((1, 2, config.action_dim), dtype=np.float32),
        "history_mask": np.zeros((1, 2), dtype=np.float32),
    }

    with pytest.raises(ValueError, match="unbatched observations"):
        policy.infer(obs)


def test_supplied_history_requires_all_history_keys() -> None:
    config = _history_model_config(history_length=2)
    policy, _idm = _history_policy_with_capturing_idm(config)
    obs = _make_single_view_obs(batch=1)
    obs["prev_state_history"] = np.zeros((1, 2, config.state_dim), dtype=np.float32)

    with pytest.raises(ValueError, match="missing required key"):
        policy.infer(obs)


def test_zero_history_model_rejects_supplied_history() -> None:
    config = _tiny_model_config(num_views=1, idm_history_length=0)
    policy = WorldModelPolicy(InverseDynamicsModel(config), config, image_keys=("observation/image",), device="cpu")
    obs = _make_single_view_obs(batch=1)
    obs.update(
        {
            "prev_state_history": np.zeros((1, 1, config.state_dim), dtype=np.float32),
            "prev_action_history": np.zeros((1, 1, config.action_dim), dtype=np.float32),
            "history_mask": np.zeros((1, 1), dtype=np.float32),
        }
    )

    with pytest.raises(ValueError, match="idm_history_length=0"):
        policy.infer(obs)


def test_supplied_history_kwargs_are_normalized_consistently_with_training() -> None:
    config = _history_model_config(history_length=2)
    state_normalizer = StateNormalizer(
        mean=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        std=torch.tensor([2.0, 2.0, 2.0, 2.0]),
    )
    action_normalizer = ActionNormalizer(
        mean=torch.full((config.action_dim,), 100.0),
        std=torch.full((config.action_dim,), 2.0),
    )
    policy, idm = _history_policy_with_capturing_idm(
        config,
        state_normalizer=state_normalizer,
        action_normalizer=action_normalizer,
    )
    raw_state_history = np.array([[[3.0, 4.0, 5.0, 6.0], [5.0, 6.0, 7.0, 8.0]]], dtype=np.float32)
    raw_action_history = np.full((1, 2, config.action_dim), 102.0, dtype=np.float32)
    obs = _make_single_view_obs(batch=1)
    obs.update(
        {
            "prev_state_history": raw_state_history,
            "prev_action_history": raw_action_history,
            "history_mask": np.ones((1, 2), dtype=np.float32),
        }
    )

    policy.infer(obs)

    call = idm.calls[-1]
    assert torch.allclose(
        call["prev_state_history"],
        state_normalizer.normalize(torch.from_numpy(raw_state_history)),
        atol=1e-5,
    )
    assert torch.allclose(
        call["prev_action_history"],
        action_normalizer.normalize(torch.from_numpy(raw_action_history)),
        atol=1e-5,
    )


def test_history_buffer_resets_when_batch_size_changes() -> None:
    config = _history_model_config(history_length=2)
    policy, idm = _history_policy_with_capturing_idm(config)

    policy.infer(_make_single_view_obs(batch=2))
    policy.infer(_make_single_view_obs(batch=2))
    assert torch.count_nonzero(idm.calls[-1]["history_mask"]) > 0

    policy.infer(_make_single_view_obs(batch=3))
    call = idm.calls[-1]
    assert call["history_mask"].shape == (3, 2)
    assert torch.count_nonzero(call["history_mask"]) == 0
    assert torch.count_nonzero(call["prev_state_history"]) == 0


def test_history_kwargs_are_normalized_consistently_with_training() -> None:
    config = _history_model_config(history_length=2)
    state_normalizer = StateNormalizer(
        mean=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        std=torch.tensor([2.0, 2.0, 2.0, 2.0]),
    )
    action_normalizer = ActionNormalizer(
        mean=torch.full((config.action_dim,), 100.0),
        std=torch.full((config.action_dim,), 2.0),
    )
    policy, idm = _history_policy_with_capturing_idm(
        config,
        state_normalizer=state_normalizer,
        action_normalizer=action_normalizer,
    )

    obs = _make_single_view_obs(batch=1)
    raw_state0 = np.array([[3.0, 4.0, 5.0, 6.0]], dtype=np.float32)
    obs["observation/state"] = raw_state0
    policy.infer(obs)  # call 0 -> model first action 0.0 (normalized space)
    obs = _make_single_view_obs(batch=1)
    obs["observation/state"] = np.array([[7.0, 8.0, 9.0, 10.0]], dtype=np.float32)
    policy.infer(obs)  # call 1 sees normalized history of call 0

    call1 = idm.calls[1]
    # State history is normalized exactly like idm_history_kwargs / normalize_state_for_idm.
    expected_state = state_normalizer.normalize(torch.tensor(raw_state0))
    assert torch.allclose(call1["prev_state_history"][:, -1], expected_state, atol=1e-5)
    # The buffered action is denormalize(model_action); re-normalizing round-trips to it.
    assert torch.allclose(call1["prev_action_history"][:, -1], torch.zeros(config.action_dim), atol=1e-4)


def test_history_state_is_normalized_when_current_state_uses_forward_hook() -> None:
    config = _history_model_config(history_length=2)
    state_normalizer = StateNormalizer(
        mean=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        std=torch.tensor([2.0, 2.0, 2.0, 2.0]),
    )
    action_normalizer = ActionNormalizer(
        mean=torch.full((config.action_dim,), 100.0),
        std=torch.full((config.action_dim,), 2.0),
    )
    idm = _CapturingHistoryIdm(config)
    attach_state_normalizer(idm, state_normalizer, normalize_forward=True)
    policy = WorldModelPolicy(
        idm,
        config,
        image_keys=("observation/image",),
        state_normalizer=state_normalizer,
        action_normalizer=action_normalizer,
        device="cpu",
    )

    raw_state0 = np.array([[3.0, 4.0, 5.0, 6.0]], dtype=np.float32)
    obs = _make_single_view_obs(batch=1)
    obs["observation/state"] = raw_state0
    policy.infer(obs)  # current state is normalized by the IDM forward pre-hook.

    raw_state1 = np.array([[7.0, 8.0, 9.0, 10.0]], dtype=np.float32)
    obs = _make_single_view_obs(batch=1)
    obs["observation/state"] = raw_state1
    policy.infer(obs)  # history from call 0 must be normalized before kwargs reach IDM.

    expected_state0 = state_normalizer.normalize(torch.tensor(raw_state0))
    expected_state1 = state_normalizer.normalize(torch.tensor(raw_state1))
    assert torch.allclose(idm.calls[0]["state"], expected_state0, atol=1e-5)
    assert torch.allclose(idm.calls[1]["state"], expected_state1, atol=1e-5)
    assert torch.allclose(idm.calls[1]["prev_state_history"][:, -1], expected_state0, atol=1e-5)
    assert torch.allclose(idm.calls[1]["prev_action_history"][:, -1], torch.zeros(config.action_dim), atol=1e-4)


def test_real_flow_transformer_history_idm_serves_closed_loop() -> None:
    config = _history_model_config(history_length=2)
    idm = InverseDynamicsModel(config).eval()
    policy = WorldModelPolicy(idm, config, image_keys=("observation/image",), device="cpu")

    first = policy.infer(_make_single_view_obs(batch=2))["actions"]
    second = policy.infer(_make_single_view_obs(batch=2))["actions"]

    assert first.shape == (2, config.action_horizon, config.action_dim)
    assert second.shape == (2, config.action_horizon, config.action_dim)


def test_direct_infer_uses_explicit_history_state_when_provided() -> None:
    config = _history_model_config(history_length=2)
    policy, idm = _history_policy_with_capturing_idm(config)

    session = policy.new_history_state()
    policy.infer(_make_single_view_obs(batch=1), history_state=session)
    policy.infer(_make_single_view_obs(batch=1), history_state=session)
    # The explicit session rolled forward independently of the policy default.
    assert torch.count_nonzero(idm.calls[-1]["history_mask"]) > 0

    policy.infer(_make_single_view_obs(batch=1))  # policy default state is still fresh
    assert torch.count_nonzero(idm.calls[-1]["history_mask"]) == 0


def test_history_state_is_independent_per_websocket_connection() -> None:
    from openpi_client import websocket_client_policy as _wcp

    config = _history_model_config(history_length=2)
    policy, idm = _history_policy_with_capturing_idm(config)
    port, _thread = _start_server_on_ephemeral_port(policy)

    client_a = _wcp.WebsocketClientPolicy("127.0.0.1", port)
    client_a.infer(_make_single_view_obs(batch=1))
    client_a.infer(_make_single_view_obs(batch=1))

    client_b = _wcp.WebsocketClientPolicy("127.0.0.1", port)
    client_b.infer(_make_single_view_obs(batch=1))

    assert len(idm.calls) == 3
    # client_a's second request rolled its own history forward.
    assert torch.count_nonzero(idm.calls[1]["history_mask"]) > 0
    # client_b opens a fresh connection -> its own empty history, not client_a's.
    assert torch.count_nonzero(idm.calls[2]["history_mask"]) == 0
    assert torch.count_nonzero(idm.calls[2]["prev_state_history"]) == 0
