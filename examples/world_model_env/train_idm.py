from __future__ import annotations

import dataclasses

import tyro

from world_model.config import (
    FLOW_DIT_350M_FF_DIM,
    FLOW_DIT_350M_HEADS,
    FLOW_DIT_350M_LATENT_DIM,
    FLOW_DIT_350M_LAYERS,
    FLOW_DIT_350M_SAMPLING_STEPS,
    FLOW_DIT_DEFAULT_ENDPOINT_LOSS_WEIGHT,
    DatasetConfig,
    DatasetSource,
    IdmArchitecture,
    IdmFlowContextConditioning,
    IdmFlowVisualTokenConditioningMode,
    IdmFlowVisualTokenRepresentation,
    IdmFlowVisualTokenScope,
    IdmFutureConditioning,
    IdmFutureRankingScoreMode,
    IdmVisualEncoder,
    ModelConfig,
    TrainConfig,
    WanVaeLatentNoiseTimeMode,
)
from world_model.train_lib import run_idm_training

CACHED_FUTURES_TRAINING_ERROR = (
    "Generated/cached futures are for eval/ranking only, not IDM training. "
    "train_idm.py trains only on ground-truth dataset futures."
)


@dataclasses.dataclass
class Args:
    dataset_source: DatasetSource = "synthetic"
    repo_id: str = "brandonyang/metaworld_ml45"
    image_keys: tuple[str, ...] = ("corner4.image",)
    output_dir: str = "output/idm"
    cached_future_dir: str | None = None
    include_gt_futures_with_cache: bool = False
    wan_vae_latent_cache_dir: str | None = None
    epochs: int = 2
    batch_size: int = 16
    learning_rate: float = 1e-4
    image_size: int = 64
    frame_delta: int = 4
    num_future_frames: int = 1
    action_horizon: int = 32
    idm_history_length: int = 0
    max_samples: int | None = None
    samples_per_episode: int | None = None
    episodes: tuple[int, ...] | None = None
    synthetic_samples: int = 128
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
    idm_context_action_warmup_epochs: int | None = None
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
    idm_arch: IdmArchitecture = "flow_transformer"
    idm_visual_encoder: IdmVisualEncoder = "patch"
    latent_dim: int = FLOW_DIT_350M_LATENT_DIM
    idm_transformer_layers: int = FLOW_DIT_350M_LAYERS
    idm_transformer_heads: int = FLOW_DIT_350M_HEADS
    idm_transformer_patch_size: int = 16
    idm_transformer_dropout: float = 0.1
    idm_transformer_ff_dim: int | None = FLOW_DIT_350M_FF_DIM
    idm_flow_sampling_steps: int = FLOW_DIT_350M_SAMPLING_STEPS
    idm_flow_num_samples: int = 1
    idm_flow_sample_noise_scale: float = 1.0
    idm_flow_time_scale: float = 1000.0
    idm_flow_endpoint_loss_weight: float = FLOW_DIT_DEFAULT_ENDPOINT_LOSS_WEIGHT
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
    wan_vae_dtype: str = "bfloat16"
    wan_vae_tiled: bool = False
    wan_vae_latent_channels: int = 48
    wan_vae_spatial_stride: int = 16
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    seed: int = 7


def main(args: Args) -> None:
    if args.cached_future_dir is not None or args.include_gt_futures_with_cache:
        raise ValueError(CACHED_FUTURES_TRAINING_ERROR)
    if args.wan_vae_latent_cache_dir is not None and args.idm_visual_encoder != "wan_vae":
        raise ValueError("--wan-vae-latent-cache-dir requires --idm-visual-encoder wan_vae.")

    dataset = DatasetConfig(
        source=args.dataset_source,
        repo_id=args.repo_id,
        image_keys=args.image_keys,
        frame_delta=args.frame_delta,
        num_future_frames=args.num_future_frames,
        action_horizon=args.action_horizon,
        idm_history_length=args.idm_history_length,
        image_size=args.image_size,
        max_samples=args.max_samples,
        samples_per_episode=args.samples_per_episode,
        episodes=args.episodes,
        synthetic_samples=args.synthetic_samples,
        seed=args.seed,
    )
    config = TrainConfig(
        dataset=dataset,
        model=ModelConfig(
            num_views=len(args.image_keys),
            image_size=args.image_size,
            action_horizon=args.action_horizon,
            num_future_frames=args.num_future_frames,
            idm_arch=args.idm_arch,
            idm_visual_encoder=args.idm_visual_encoder,
            idm_history_length=args.idm_history_length,
            latent_dim=args.latent_dim,
            idm_transformer_layers=args.idm_transformer_layers,
            idm_transformer_heads=args.idm_transformer_heads,
            idm_transformer_patch_size=args.idm_transformer_patch_size,
            idm_transformer_dropout=args.idm_transformer_dropout,
            idm_transformer_ff_dim=args.idm_transformer_ff_dim,
            idm_flow_sampling_steps=args.idm_flow_sampling_steps,
            idm_flow_num_samples=args.idm_flow_num_samples,
            idm_flow_sample_noise_scale=args.idm_flow_sample_noise_scale,
            idm_flow_time_scale=args.idm_flow_time_scale,
            idm_flow_endpoint_loss_weight=args.idm_flow_endpoint_loss_weight,
            idm_flow_endpoint_consistency_loss_weight=args.idm_flow_endpoint_consistency_loss_weight,
            idm_flow_zero_start_endpoint_loss_weight=args.idm_flow_zero_start_endpoint_loss_weight,
            idm_flow_sampled_action_loss_weight=args.idm_flow_sampled_action_loss_weight,
            idm_flow_context_conditioning=args.idm_flow_context_conditioning,
            idm_future_conditioning=args.idm_future_conditioning,
            idm_flow_visual_token_conditioning=args.idm_flow_visual_token_conditioning,
            idm_flow_visual_token_conditioning_mode=args.idm_flow_visual_token_conditioning_mode,
            idm_flow_visual_token_scope=args.idm_flow_visual_token_scope,
            idm_flow_visual_token_representation=args.idm_flow_visual_token_representation,
            idm_flow_train_time_min=args.idm_flow_train_time_min,
            idm_flow_train_time_max=args.idm_flow_train_time_max,
            wan_vae_repo_dir=args.wan_vae_repo_dir,
            wan_vae_checkpoint_path=args.wan_vae_checkpoint_path,
            wan_vae_dtype=args.wan_vae_dtype,
            wan_vae_tiled=args.wan_vae_tiled,
            wan_vae_latent_channels=args.wan_vae_latent_channels,
            wan_vae_spatial_stride=args.wan_vae_spatial_stride,
            wan_vae_use_cached_latents=args.wan_vae_latent_cache_dir is not None,
        ),
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        device=args.device,
        data_parallel=args.data_parallel,
        normalize_actions=args.normalize_actions,
        idm_state_dropout=args.idm_state_dropout,
        idm_current_frame_dropout=args.idm_current_frame_dropout,
        idm_future_noise_std=args.idm_future_noise_std,
        idm_future_frame_dropout=args.idm_future_frame_dropout,
        idm_wan_vae_current_latent_dropout=args.idm_wan_vae_current_latent_dropout,
        idm_wan_vae_latent_noise_prob=args.idm_wan_vae_latent_noise_prob,
        idm_wan_vae_latent_noise_s_min=args.idm_wan_vae_latent_noise_s_min,
        idm_wan_vae_latent_noise_s_max=args.idm_wan_vae_latent_noise_s_max,
        idm_wan_vae_latent_noise_time_mode=args.idm_wan_vae_latent_noise_time_mode,
        idm_context_action_loss_weight=args.idm_context_action_loss_weight,
        idm_future_contrastive_weight=args.idm_future_contrastive_weight,
        idm_future_contrastive_margin=args.idm_future_contrastive_margin,
        idm_future_ranking_weight=args.idm_future_ranking_weight,
        idm_future_ranking_start_epoch=args.idm_future_ranking_start_epoch,
        idm_future_ranking_ramp_epochs=args.idm_future_ranking_ramp_epochs,
        idm_future_ranking_temperature=args.idm_future_ranking_temperature,
        idm_future_ranking_noise_std=args.idm_future_ranking_noise_std,
        idm_future_ranking_repeated_current_negative=args.idm_future_ranking_repeated_current_negative,
        idm_future_ranking_shuffled_future_negative=args.idm_future_ranking_shuffled_future_negative,
        idm_future_ranking_noisy_future_negative=args.idm_future_ranking_noisy_future_negative,
        idm_future_ranking_zero_future_negative=args.idm_future_ranking_zero_future_negative,
        idm_future_ranking_same_task_negative=args.idm_future_ranking_same_task_negative,
        idm_future_ranking_score_mode=args.idm_future_ranking_score_mode,
        idm_future_usage_eval=args.idm_future_usage_eval,
        idm_future_usage_rank_accuracy_min=args.idm_future_usage_rank_accuracy_min,
        idm_future_usage_gap_min=args.idm_future_usage_gap_min,
        idm_future_usage_degradation_min=args.idm_future_usage_degradation_min,
        idm_future_usage_output_delta_mse_min=args.idm_future_usage_output_delta_mse_min,
        idm_future_usage_score_mode=args.idm_future_usage_score_mode,
        idm_same_task_batching=args.idm_same_task_batching,
        idm_same_task_future_delta_weight=args.idm_same_task_future_delta_weight,
        idm_same_task_future_delta_time_value=args.idm_same_task_future_delta_time_value,
        idm_same_task_future_delta_max_state_distance=args.idm_same_task_future_delta_max_state_distance,
        idm_same_task_future_delta_min_action_delta_mse=args.idm_same_task_future_delta_min_action_delta_mse,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        seed=args.seed,
    )
    run_kwargs = {}
    if args.wan_vae_latent_cache_dir is not None:
        run_kwargs["wan_vae_latent_cache_dir"] = args.wan_vae_latent_cache_dir
    if args.idm_context_action_warmup_epochs is not None:
        run_kwargs["idm_context_action_warmup_epochs"] = args.idm_context_action_warmup_epochs
    run_idm_training(config, **run_kwargs)


if __name__ == "__main__":
    main(tyro.cli(Args))
