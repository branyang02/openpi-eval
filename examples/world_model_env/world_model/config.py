from __future__ import annotations

import dataclasses
from typing import Literal

DatasetSource = Literal["synthetic", "lerobot"]
IdmTargetSource = Literal["ground_truth", "generated"]
IdmArchitecture = Literal["stacked", "delta", "transformer", "flow_transformer"]
IdmVisualEncoder = Literal["patch", "wan_vae"]
IdmFlowContextConditioning = Literal["token", "additive"]
IdmFutureConditioning = Literal["full", "current_only", "future_only"]
IdmFutureRankingScoreMode = Literal["teacher_forced_endpoint", "sampled_action"]
IdmFlowVisualTokenConditioningMode = Literal["prefix", "cross_attention"]
IdmFlowVisualTokenScope = Literal["all", "future_only"]
IdmFlowVisualTokenRepresentation = Literal["encoded", "future_delta"]
WanVaeLatentNoiseTimeMode = Literal["all", "future_only"]
WorldModelSource = Literal["conv_baseline", "wan2_2"]
FutureFrameStrategy = Literal["first", "source_offsets"]

FLOW_DIT_350M_LATENT_DIM = 1024
FLOW_DIT_350M_LAYERS = 14
FLOW_DIT_350M_HEADS = 16
FLOW_DIT_350M_FF_DIM = 4096
FLOW_DIT_350M_SAMPLING_STEPS = 16
FLOW_DIT_DEFAULT_ENDPOINT_LOSS_WEIGHT = 0.1


def validate_future_frame_strategy(strategy: str) -> FutureFrameStrategy:
    if strategy == "first":
        return "first"
    if strategy == "source_offsets":
        return "source_offsets"
    raise ValueError(
        "future_frame_strategy must be one of {'first', 'source_offsets'}, "
        f"got {strategy!r}. 'first' selects generated-video frames [1..K]; "
        "'source_offsets' selects generated-video frames matching dataset source offsets "
        "[frame_delta, 2*frame_delta, ...]."
    )


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    source: DatasetSource = "synthetic"
    repo_id: str = "brandonyang/metaworld_ml45"
    image_keys: tuple[str, ...] = ("corner.image", "corner4.image", "gripperPOV.image")
    state_key: str = "observation.state"
    action_key: str = "actions"
    task_key: str = "task"
    prompt_from_task: bool = True
    frame_delta: int = 4
    num_future_frames: int = 1
    action_horizon: int = 32
    idm_history_length: int = 0
    image_size: int = 64
    max_samples: int | None = None
    samples_per_episode: int | None = None
    episodes: tuple[int, ...] | None = None
    synthetic_samples: int = 128
    task_vocab_size: int = 4096
    seed: int = 7

    def __post_init__(self) -> None:
        if self.idm_history_length < 0:
            raise ValueError(f"idm_history_length must be non-negative, got {self.idm_history_length}.")
        if self.samples_per_episode is None:
            return
        if self.samples_per_episode <= 0:
            raise ValueError(f"samples_per_episode must be positive, got {self.samples_per_episode}.")
        if self.max_samples is not None:
            raise ValueError("samples_per_episode cannot be used together with max_samples.")
        if self.source != "lerobot":
            raise ValueError("samples_per_episode is only supported for source='lerobot'.")


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    num_views: int = 3
    image_size: int = 64
    state_dim: int = 4
    action_dim: int = 4
    action_horizon: int = 32
    idm_history_length: int = 0
    num_future_frames: int = 1
    task_vocab_size: int = 4096
    task_embed_dim: int = 64
    latent_dim: int = 256
    idm_arch: IdmArchitecture = "stacked"
    idm_visual_encoder: IdmVisualEncoder = "patch"
    idm_transformer_layers: int = 4
    idm_transformer_heads: int = 8
    idm_transformer_patch_size: int = 16
    idm_transformer_dropout: float = 0.1
    idm_transformer_ff_dim: int | None = None
    idm_flow_sampling_steps: int = 8
    idm_flow_num_samples: int = 1
    idm_flow_sample_noise_scale: float = 1.0
    idm_flow_time_scale: float = 1000.0
    idm_flow_endpoint_loss_weight: float = 0.0
    idm_flow_endpoint_consistency_loss_weight: float = 0.0
    idm_flow_zero_start_endpoint_loss_weight: float = 0.0
    idm_flow_sampled_action_loss_weight: float = 0.0
    idm_flow_context_conditioning: IdmFlowContextConditioning = "token"
    idm_future_conditioning: IdmFutureConditioning = "full"
    idm_flow_visual_token_conditioning: bool = False
    idm_flow_visual_token_conditioning_mode: IdmFlowVisualTokenConditioningMode = "prefix"
    idm_flow_visual_token_scope: IdmFlowVisualTokenScope = "all"
    idm_flow_visual_token_representation: IdmFlowVisualTokenRepresentation = "encoded"
    idm_flow_train_time_min: float = 0.0
    idm_flow_train_time_max: float = 1.0
    wan_vae_repo_dir: str | None = None
    wan_vae_checkpoint_path: str | None = None
    wan_vae_dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    wan_vae_tiled: bool = False
    wan_vae_latent_channels: int = 48
    wan_vae_spatial_stride: int = 16
    wan_vae_use_cached_latents: bool = False

    def __post_init__(self) -> None:
        if self.idm_history_length < 0:
            raise ValueError(f"idm_history_length must be non-negative, got {self.idm_history_length}.")
        if self.idm_history_length > 0 and self.idm_arch != "flow_transformer":
            raise ValueError("idm_history_length is supported only with idm_arch='flow_transformer'.")
        if self.idm_flow_sample_noise_scale < 0.0:
            raise ValueError(
                "idm_flow_sample_noise_scale must be non-negative, "
                f"got {self.idm_flow_sample_noise_scale}."
            )
        if self.idm_flow_context_conditioning not in ("token", "additive"):
            raise ValueError(
                "idm_flow_context_conditioning must be one of {'token', 'additive'}, "
                f"got {self.idm_flow_context_conditioning!r}."
            )
        if self.idm_future_conditioning not in ("full", "current_only", "future_only"):
            raise ValueError(
                "idm_future_conditioning must be one of {'full', 'current_only', 'future_only'}, "
                f"got {self.idm_future_conditioning!r}."
            )
        if self.idm_future_conditioning in ("current_only", "future_only") and self.idm_arch != "flow_transformer":
            raise ValueError(
                f"idm_future_conditioning={self.idm_future_conditioning!r} is supported only with "
                "idm_arch='flow_transformer'."
            )
        if self.idm_flow_endpoint_consistency_loss_weight < 0.0:
            raise ValueError(
                "idm_flow_endpoint_consistency_loss_weight must be non-negative, "
                f"got {self.idm_flow_endpoint_consistency_loss_weight}."
            )
        if self.idm_flow_zero_start_endpoint_loss_weight < 0.0:
            raise ValueError(
                "idm_flow_zero_start_endpoint_loss_weight must be non-negative, "
                f"got {self.idm_flow_zero_start_endpoint_loss_weight}."
            )
        if self.idm_flow_sampled_action_loss_weight < 0.0:
            raise ValueError(
                "idm_flow_sampled_action_loss_weight must be non-negative, "
                f"got {self.idm_flow_sampled_action_loss_weight}."
            )
        if self.idm_flow_visual_token_conditioning_mode not in ("prefix", "cross_attention"):
            raise ValueError(
                "idm_flow_visual_token_conditioning_mode must be one of {'prefix', 'cross_attention'}, "
                f"got {self.idm_flow_visual_token_conditioning_mode!r}."
            )
        if self.idm_flow_visual_token_scope not in ("all", "future_only"):
            raise ValueError(
                "idm_flow_visual_token_scope must be one of {'all', 'future_only'}, "
                f"got {self.idm_flow_visual_token_scope!r}."
            )
        if self.idm_flow_visual_token_representation not in ("encoded", "future_delta"):
            raise ValueError(
                "idm_flow_visual_token_representation must be one of {'encoded', 'future_delta'}, "
                f"got {self.idm_flow_visual_token_representation!r}."
            )
        if (
            self.idm_flow_visual_token_conditioning_mode == "cross_attention"
            and not self.idm_flow_visual_token_conditioning
        ):
            raise ValueError(
                "idm_flow_visual_token_conditioning_mode='cross_attention' requires "
                "idm_flow_visual_token_conditioning=True."
            )
        if self.idm_flow_visual_token_conditioning:
            if self.idm_future_conditioning == "current_only":
                raise ValueError(
                    "idm_flow_visual_token_conditioning cannot be used with "
                    "idm_future_conditioning='current_only'."
                )
            if self.idm_arch != "flow_transformer":
                raise ValueError(
                    "idm_flow_visual_token_conditioning is supported only with idm_arch='flow_transformer'."
                )
            if self.idm_visual_encoder not in ("patch", "wan_vae"):
                raise ValueError(
                    "idm_flow_visual_token_conditioning is supported only with "
                    "idm_visual_encoder in {'patch', 'wan_vae'}."
                )
        if self.idm_flow_visual_token_scope == "future_only":
            if not self.idm_flow_visual_token_conditioning:
                raise ValueError(
                    "idm_flow_visual_token_scope='future_only' requires "
                    "idm_flow_visual_token_conditioning=True."
                )
            if self.idm_arch != "flow_transformer":
                raise ValueError(
                    "idm_flow_visual_token_scope='future_only' is supported only with "
                    "idm_arch='flow_transformer'."
                )
            if self.idm_visual_encoder not in ("patch", "wan_vae"):
                raise ValueError(
                    "idm_flow_visual_token_scope='future_only' is supported only with "
                    "idm_visual_encoder in {'patch', 'wan_vae'}."
                )
            if self.idm_visual_encoder == "wan_vae":
                latent_frames = (1 + self.num_future_frames + 3) // 4
                if latent_frames < 2:
                    raise ValueError(
                        "idm_flow_visual_token_scope='future_only' requires at least one future Wan VAE latent "
                        "time slice; configured num_future_frames="
                        f"{self.num_future_frames} gives latent_frames={latent_frames} (need >= 2)."
                    )
            elif self.num_future_frames < 1:
                raise ValueError(
                    "idm_flow_visual_token_scope='future_only' requires at least one future patch frame; "
                    f"got num_future_frames={self.num_future_frames}."
                )
        if self.idm_flow_visual_token_representation == "future_delta":
            if not self.idm_flow_visual_token_conditioning:
                raise ValueError(
                    "idm_flow_visual_token_representation='future_delta' requires "
                    "idm_flow_visual_token_conditioning=True."
                )
            if self.idm_arch != "flow_transformer":
                raise ValueError(
                    "idm_flow_visual_token_representation='future_delta' is supported only with "
                    "idm_arch='flow_transformer'."
                )
            if self.idm_visual_encoder not in ("patch", "wan_vae"):
                raise ValueError(
                    "idm_flow_visual_token_representation='future_delta' is supported only with "
                    "idm_visual_encoder in {'patch', 'wan_vae'}."
                )
            if self.idm_flow_visual_token_scope != "future_only":
                raise ValueError(
                    "idm_flow_visual_token_representation='future_delta' requires "
                    "idm_flow_visual_token_scope='future_only'."
                )
            if self.idm_visual_encoder == "wan_vae":
                latent_frames = (1 + self.num_future_frames + 3) // 4
                if latent_frames < 2:
                    raise ValueError(
                        "idm_flow_visual_token_representation='future_delta' requires at least one future Wan VAE "
                        "latent time slice; configured num_future_frames="
                        f"{self.num_future_frames} gives latent_frames={latent_frames} (need >= 2)."
                    )
            elif self.num_future_frames < 1:
                raise ValueError(
                    "idm_flow_visual_token_representation='future_delta' requires at least one future patch frame; "
                    f"got num_future_frames={self.num_future_frames}."
                )
        if not 0.0 <= self.idm_flow_train_time_min <= 1.0:
            raise ValueError(f"idm_flow_train_time_min must be in [0, 1], got {self.idm_flow_train_time_min}.")
        if not 0.0 <= self.idm_flow_train_time_max <= 1.0:
            raise ValueError(f"idm_flow_train_time_max must be in [0, 1], got {self.idm_flow_train_time_max}.")
        if self.idm_flow_train_time_min > self.idm_flow_train_time_max:
            raise ValueError(
                "idm_flow_train_time_min must be <= idm_flow_train_time_max, "
                f"got {self.idm_flow_train_time_min} > {self.idm_flow_train_time_max}."
            )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    dataset: DatasetConfig = dataclasses.field(default_factory=DatasetConfig)
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    output_dir: str = "output/smoke"
    epochs: int = 2
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    wm_loss_weight: float = 1.0
    idm_loss_weight: float = 1.0
    action_smoothness_weight: float = 0.01
    idm_target_source: IdmTargetSource = "ground_truth"
    eval_fraction: float = 0.1
    split_gap: int = 1
    num_workers: int = 0
    device: str = "auto"
    data_parallel: bool = False
    normalize_actions: bool = True
    idm_state_dropout: float = 0.0
    idm_current_frame_dropout: float = 0.0
    idm_future_noise_std: float = 0.0
    idm_future_frame_dropout: float = 0.0
    idm_wan_vae_current_latent_dropout: float = 0.0
    idm_wan_vae_latent_noise_prob: float = 0.0
    idm_wan_vae_latent_noise_s_min: float = 0.5
    idm_wan_vae_latent_noise_s_max: float = 1.0
    idm_wan_vae_latent_noise_time_mode: WanVaeLatentNoiseTimeMode = "all"
    idm_context_action_loss_weight: float = 0.0
    idm_future_contrastive_weight: float = 0.0
    idm_future_contrastive_margin: float = 0.1
    idm_future_ranking_weight: float = 0.0
    idm_future_ranking_start_epoch: int | None = None
    idm_future_ranking_ramp_epochs: int = 0
    idm_future_ranking_temperature: float = 0.1
    idm_future_ranking_noise_std: float = 1.0
    idm_future_ranking_repeated_current_negative: bool = False
    idm_future_ranking_shuffled_future_negative: bool = False
    idm_future_ranking_noisy_future_negative: bool = False
    idm_future_ranking_zero_future_negative: bool = False
    idm_future_ranking_same_task_negative: bool = False
    idm_future_ranking_score_mode: IdmFutureRankingScoreMode = "teacher_forced_endpoint"
    idm_future_usage_eval: bool = False
    idm_future_usage_rank_accuracy_min: float = 0.55
    idm_future_usage_gap_min: float = 0.0
    idm_future_usage_degradation_min: float = 1e-4
    idm_future_usage_output_delta_mse_min: float = 1e-4
    idm_future_usage_score_mode: IdmFutureRankingScoreMode = "teacher_forced_endpoint"
    idm_same_task_batching: bool = False
    idm_same_task_future_delta_weight: float = 0.0
    idm_same_task_future_delta_time_value: float = 0.5
    idm_same_task_future_delta_max_state_distance: float | None = None
    idm_same_task_future_delta_min_action_delta_mse: float = 0.0
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    seed: int = 7

    def __post_init__(self) -> None:
        for field_name in ("idm_current_frame_dropout", "idm_wan_vae_current_latent_dropout"):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1], got {value}.")
        if not 0.0 <= self.idm_wan_vae_latent_noise_prob <= 1.0:
            raise ValueError(
                f"idm_wan_vae_latent_noise_prob must be in [0, 1], got {self.idm_wan_vae_latent_noise_prob}."
            )
        if not 0.0 <= self.idm_wan_vae_latent_noise_s_min <= 1.0:
            raise ValueError(
                f"idm_wan_vae_latent_noise_s_min must be in [0, 1], got {self.idm_wan_vae_latent_noise_s_min}."
            )
        if not 0.0 <= self.idm_wan_vae_latent_noise_s_max <= 1.0:
            raise ValueError(
                f"idm_wan_vae_latent_noise_s_max must be in [0, 1], got {self.idm_wan_vae_latent_noise_s_max}."
            )
        if self.idm_wan_vae_latent_noise_s_min > self.idm_wan_vae_latent_noise_s_max:
            raise ValueError(
                "idm_wan_vae_latent_noise_s_min must be <= idm_wan_vae_latent_noise_s_max, "
                f"got {self.idm_wan_vae_latent_noise_s_min} > {self.idm_wan_vae_latent_noise_s_max}."
            )
        if self.idm_wan_vae_latent_noise_time_mode not in ("all", "future_only"):
            raise ValueError(
                "idm_wan_vae_latent_noise_time_mode must be one of {'all', 'future_only'}, "
                f"got {self.idm_wan_vae_latent_noise_time_mode!r}."
            )
        if self.idm_future_contrastive_weight < 0.0:
            raise ValueError(
                f"idm_future_contrastive_weight must be non-negative, got {self.idm_future_contrastive_weight}."
            )
        if self.idm_future_contrastive_margin < 0.0:
            raise ValueError(
                f"idm_future_contrastive_margin must be non-negative, got {self.idm_future_contrastive_margin}."
            )
        if self.idm_future_ranking_weight < 0.0:
            raise ValueError(f"idm_future_ranking_weight must be non-negative, got {self.idm_future_ranking_weight}.")
        if self.idm_future_ranking_start_epoch is not None and self.idm_future_ranking_start_epoch < 0:
            raise ValueError(
                f"idm_future_ranking_start_epoch must be non-negative, got {self.idm_future_ranking_start_epoch}."
            )
        if self.idm_future_ranking_ramp_epochs < 0:
            raise ValueError(
                f"idm_future_ranking_ramp_epochs must be non-negative, got {self.idm_future_ranking_ramp_epochs}."
            )
        if self.idm_future_ranking_temperature <= 0.0:
            raise ValueError(
                f"idm_future_ranking_temperature must be positive, got {self.idm_future_ranking_temperature}."
            )
        if self.idm_future_ranking_noise_std < 0.0:
            raise ValueError(
                f"idm_future_ranking_noise_std must be non-negative, got {self.idm_future_ranking_noise_std}."
            )
        if self.idm_future_ranking_score_mode not in ("teacher_forced_endpoint", "sampled_action"):
            raise ValueError(
                "idm_future_ranking_score_mode must be one of {'teacher_forced_endpoint', 'sampled_action'}, "
                f"got {self.idm_future_ranking_score_mode!r}."
            )
        if not 0.0 <= self.idm_future_usage_rank_accuracy_min <= 1.0:
            raise ValueError(
                "idm_future_usage_rank_accuracy_min must be in [0, 1], "
                f"got {self.idm_future_usage_rank_accuracy_min}."
            )
        for field_name in (
            "idm_future_usage_gap_min",
            "idm_future_usage_degradation_min",
            "idm_future_usage_output_delta_mse_min",
        ):
            value = getattr(self, field_name)
            if value < 0.0:
                raise ValueError(f"{field_name} must be non-negative, got {value}.")
        if self.idm_future_usage_score_mode not in ("teacher_forced_endpoint", "sampled_action"):
            raise ValueError(
                "idm_future_usage_score_mode must be one of {'teacher_forced_endpoint', 'sampled_action'}, "
                f"got {self.idm_future_usage_score_mode!r}."
            )
        if self.idm_same_task_future_delta_weight < 0.0:
            raise ValueError(
                "idm_same_task_future_delta_weight must be non-negative, "
                f"got {self.idm_same_task_future_delta_weight}."
            )
        if not 0.0 <= self.idm_same_task_future_delta_time_value < 1.0:
            raise ValueError(
                "idm_same_task_future_delta_time_value must be in [0, 1), "
                f"got {self.idm_same_task_future_delta_time_value}."
            )
        if (
            self.idm_same_task_future_delta_max_state_distance is not None
            and self.idm_same_task_future_delta_max_state_distance < 0.0
        ):
            raise ValueError(
                "idm_same_task_future_delta_max_state_distance must be non-negative or None, "
                f"got {self.idm_same_task_future_delta_max_state_distance}."
            )
        if self.idm_same_task_future_delta_min_action_delta_mse < 0.0:
            raise ValueError(
                "idm_same_task_future_delta_min_action_delta_mse must be non-negative, "
                f"got {self.idm_same_task_future_delta_min_action_delta_mse}."
            )
        if self.idm_context_action_loss_weight < 0.0:
            raise ValueError(
                f"idm_context_action_loss_weight must be non-negative, got {self.idm_context_action_loss_weight}."
            )
        if self.idm_context_action_loss_weight > 0.0 and self.model.idm_arch != "flow_transformer":
            raise ValueError("idm_context_action_loss_weight is supported only with idm_arch='flow_transformer'.")
        if self.idm_future_contrastive_weight > 0.0 and self.model.idm_arch != "flow_transformer":
            raise ValueError("idm_future_contrastive_weight is supported only with idm_arch='flow_transformer'.")
        if self.idm_future_ranking_weight > 0.0 and self.model.idm_arch != "flow_transformer":
            raise ValueError("idm_future_ranking_weight is supported only with idm_arch='flow_transformer'.")
        if self.idm_future_usage_eval and self.model.idm_arch != "flow_transformer":
            raise ValueError("idm_future_usage_eval is supported only with idm_arch='flow_transformer'.")
        if self.idm_same_task_future_delta_weight > 0.0 and self.model.idm_arch != "flow_transformer":
            raise ValueError(
                "idm_same_task_future_delta_weight is supported only with idm_arch='flow_transformer'."
            )
        ranking_negatives_enabled = (
            self.idm_future_ranking_repeated_current_negative
            or self.idm_future_ranking_shuffled_future_negative
            or self.idm_future_ranking_noisy_future_negative
            or self.idm_future_ranking_zero_future_negative
            or self.idm_future_ranking_same_task_negative
        )
        if self.idm_future_ranking_weight > 0.0 and not ranking_negatives_enabled:
            raise ValueError(
                "idm_future_ranking_weight requires at least one enabled negative: "
                "idm_future_ranking_repeated_current_negative, "
                "idm_future_ranking_shuffled_future_negative, "
                "idm_future_ranking_noisy_future_negative, "
                "idm_future_ranking_zero_future_negative, or "
                "idm_future_ranking_same_task_negative."
            )
        if self.dataset.idm_history_length > 0 and self.model.idm_arch != "flow_transformer":
            raise ValueError("idm_history_length is supported only with idm_arch='flow_transformer'.")
        if self.model.idm_history_length > 0 and self.model.idm_history_length != self.dataset.idm_history_length:
            raise ValueError(
                "model.idm_history_length must match dataset.idm_history_length when explicitly set; "
                f"got model={self.model.idm_history_length}, dataset={self.dataset.idm_history_length}."
            )


@dataclasses.dataclass(frozen=True)
class Wan22Config:
    repo_dir: str
    checkpoint_dir: str
    task: str = "ti2v-5B"
    size: str = "1280*704"
    frame_num: int = 17
    sample_steps: int | None = None
    sample_shift: float | None = None
    sample_guide_scale: float | None = None
    offload_model: bool = False
    convert_model_dtype: bool = False
    t5_cpu: bool = False
    base_seed: int = 7
    python_executable: str = "python"
    frame_delta: int = 1
    future_frame_strategy: FutureFrameStrategy = "first"
    prompt_template: str = (
        "Robot manipulation in MetaWorld. Task: {task}. "
        "Generate the near future scene after the robot continues the task."
    )

    def __post_init__(self) -> None:
        if self.frame_delta <= 0:
            raise ValueError(f"frame_delta must be positive, got {self.frame_delta}.")
        validate_future_frame_strategy(self.future_frame_strategy)
