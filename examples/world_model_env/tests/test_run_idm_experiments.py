from __future__ import annotations

import json

import pytest

import run_idm_experiments
from run_idm_experiments import Args, main


def test_run_idm_experiments_writes_summary_and_checkpoints(tmp_path) -> None:
    main(
        Args(
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            max_samples=16,
            synthetic_samples=16,
            image_size=32,
            num_future_frames=4,
            action_horizons=(4,),
            epochs=1,
            batch_size=4,
            learning_rates=(3e-4,),
            seeds=(5,),
            latent_dim=32,
            idm_transformer_layers=1,
            idm_transformer_heads=4,
            idm_transformer_ff_dim=64,
            idm_flow_sampling_steps=2,
            device="cpu",
            diagnostics=False,
        )
    )

    summary = json.loads((tmp_path / "experiment_summary.json").read_text())
    row = summary["runs"][0]

    assert summary["num_runs"] == 1
    assert row["run_name"] == "idm_h4_lr0.0003_seed5"
    assert row["status"] == "success"
    assert row["train_best"]["idm_mse"] >= 0.0
    assert row["final_full_eval"]["idm_mse"] >= 0.0
    assert row["best_full_eval"]["idm_mse"] >= 0.0
    assert (tmp_path / row["run_name"] / "idm_checkpoint.pt").exists()
    assert (tmp_path / row["run_name"] / "best_idm_checkpoint.pt").exists()
    assert summary["best_by_full_eval"]["run_name"] == row["run_name"]


def test_run_idm_experiments_records_failed_config_and_keeps_summary_incremental(tmp_path, monkeypatch) -> None:
    train_configs = []

    def fake_run_idm_training(config):
        train_configs.append(config)
        if len(train_configs) == 2:
            partial_summary = json.loads((tmp_path / "experiment_summary.json").read_text())
            assert partial_summary["num_runs"] == 1
            assert partial_summary["runs"][0]["status"] == "success"
            raise RuntimeError("second config exploded")
        return {
            "final": {"idm_mse": 0.25},
            "best": {"idm_mse": 0.125},
        }

    def fake_run_full_dataset_eval(**kwargs):
        del kwargs
        return {"idm_mse": 0.05}

    monkeypatch.setattr(run_idm_experiments, "run_idm_training", fake_run_idm_training)
    monkeypatch.setattr(run_idm_experiments, "run_full_dataset_eval", fake_run_full_dataset_eval)

    main(
        Args(
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            max_samples=16,
            synthetic_samples=16,
            image_size=32,
            num_future_frames=4,
            action_horizons=(4, 8),
            epochs=1,
            batch_size=4,
            learning_rates=(3e-4,),
            seeds=(5,),
            device="cpu",
            diagnostics=False,
        )
    )

    summary = json.loads((tmp_path / "experiment_summary.json").read_text())
    success, failure = summary["runs"]

    assert len(train_configs) == 2
    assert summary["num_runs"] == 2
    assert success["run_name"] == "idm_h4_lr0.0003_seed5"
    assert success["status"] == "success"
    assert failure == {
        "run_name": "idm_h8_lr0.0003_seed5",
        "status": "failed",
        "error": "second config exploded",
    }
    assert summary["best_by_full_eval"]["run_name"] == success["run_name"]


def test_run_idm_experiments_forwards_future_ranking_options(tmp_path, monkeypatch) -> None:
    train_configs = []

    def fake_run_idm_training(config):
        train_configs.append(config)
        return {
            "final": {"idm_mse": 0.25},
            "best": {"idm_mse": 0.125},
        }

    def fake_run_full_dataset_eval(**kwargs):
        del kwargs
        return {"idm_mse": 0.05}

    monkeypatch.setattr(run_idm_experiments, "run_idm_training", fake_run_idm_training)
    monkeypatch.setattr(run_idm_experiments, "run_full_dataset_eval", fake_run_full_dataset_eval)

    main(
        Args(
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            max_samples=16,
            synthetic_samples=16,
            image_size=32,
            num_future_frames=4,
            action_horizons=(4,),
            epochs=1,
            batch_size=4,
            learning_rates=(3e-4,),
            seeds=(5,),
            device="cpu",
            diagnostics=False,
            idm_visual_encoder="wan_vae",
            idm_flow_visual_token_conditioning=True,
            idm_flow_visual_token_conditioning_mode="cross_attention",
            idm_flow_visual_token_scope="future_only",
            idm_flow_visual_token_representation="future_delta",
            idm_flow_endpoint_consistency_loss_weight=0.2,
            idm_flow_zero_start_endpoint_loss_weight=0.3,
            idm_flow_sampled_action_loss_weight=0.35,
            idm_future_ranking_weight=0.5,
            idm_future_ranking_start_epoch=2,
            idm_future_ranking_ramp_epochs=3,
            idm_future_ranking_temperature=0.25,
            idm_future_ranking_noise_std=0.75,
            idm_future_ranking_repeated_current_negative=True,
            idm_future_ranking_shuffled_future_negative=True,
            idm_future_ranking_noisy_future_negative=True,
            idm_future_ranking_zero_future_negative=True,
            idm_future_ranking_same_task_negative=True,
            idm_future_ranking_score_mode="sampled_action",
            idm_same_task_batching=True,
            idm_same_task_future_delta_weight=0.5,
            idm_same_task_future_delta_time_value=0.25,
            idm_same_task_future_delta_max_state_distance=0.75,
            idm_same_task_future_delta_min_action_delta_mse=0.01,
        )
    )

    assert len(train_configs) == 1
    train_config = train_configs[0]
    assert train_config.model.idm_visual_encoder == "wan_vae"
    assert train_config.model.idm_flow_visual_token_conditioning is True
    assert train_config.model.idm_flow_visual_token_conditioning_mode == "cross_attention"
    assert train_config.model.idm_flow_visual_token_scope == "future_only"
    assert train_config.model.idm_flow_visual_token_representation == "future_delta"
    assert train_config.model.idm_flow_endpoint_consistency_loss_weight == pytest.approx(0.2)
    assert train_config.model.idm_flow_zero_start_endpoint_loss_weight == pytest.approx(0.3)
    assert train_config.model.idm_flow_sampled_action_loss_weight == pytest.approx(0.35)
    assert train_config.idm_future_ranking_weight == pytest.approx(0.5)
    assert train_config.idm_future_ranking_start_epoch == 2
    assert train_config.idm_future_ranking_ramp_epochs == 3
    assert train_config.idm_future_ranking_temperature == pytest.approx(0.25)
    assert train_config.idm_future_ranking_noise_std == pytest.approx(0.75)
    assert train_config.idm_future_ranking_repeated_current_negative
    assert train_config.idm_future_ranking_shuffled_future_negative
    assert train_config.idm_future_ranking_noisy_future_negative
    assert train_config.idm_future_ranking_zero_future_negative
    assert train_config.idm_future_ranking_same_task_negative
    assert train_config.idm_future_ranking_score_mode == "sampled_action"
    assert train_config.idm_same_task_batching
    assert train_config.idm_same_task_future_delta_weight == pytest.approx(0.5)
    assert train_config.idm_same_task_future_delta_time_value == pytest.approx(0.25)
    assert train_config.idm_same_task_future_delta_max_state_distance == pytest.approx(0.75)
    assert train_config.idm_same_task_future_delta_min_action_delta_mse == pytest.approx(0.01)


def test_run_idm_experiments_forwards_future_usage_eval_options(tmp_path, monkeypatch) -> None:
    train_configs = []

    def fake_run_idm_training(config):
        train_configs.append(config)
        return {
            "final": {"idm_mse": 0.25},
            "best": {"idm_mse": 0.125},
        }

    def fake_run_full_dataset_eval(**kwargs):
        del kwargs
        return {"idm_mse": 0.05}

    monkeypatch.setattr(run_idm_experiments, "run_idm_training", fake_run_idm_training)
    monkeypatch.setattr(run_idm_experiments, "run_full_dataset_eval", fake_run_full_dataset_eval)

    main(
        Args(
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            max_samples=16,
            synthetic_samples=16,
            image_size=32,
            num_future_frames=4,
            action_horizons=(4,),
            epochs=1,
            batch_size=4,
            learning_rates=(3e-4,),
            seeds=(5,),
            device="cpu",
            diagnostics=False,
            idm_future_usage_eval=True,
            idm_future_usage_rank_accuracy_min=0.8,
            idm_future_usage_gap_min=0.05,
            idm_future_usage_degradation_min=0.02,
            idm_future_usage_output_delta_mse_min=0.03,
            idm_future_usage_score_mode="sampled_action",
        )
    )

    assert len(train_configs) == 1
    train_config = train_configs[0]
    assert train_config.idm_future_usage_eval is True
    assert train_config.idm_future_usage_rank_accuracy_min == pytest.approx(0.8)
    assert train_config.idm_future_usage_gap_min == pytest.approx(0.05)
    assert train_config.idm_future_usage_degradation_min == pytest.approx(0.02)
    assert train_config.idm_future_usage_output_delta_mse_min == pytest.approx(0.03)
    assert train_config.idm_future_usage_score_mode == "sampled_action"


def test_run_idm_experiments_forwards_idm_future_conditioning(tmp_path, monkeypatch) -> None:
    train_configs = []

    def fake_run_idm_training(config):
        train_configs.append(config)
        return {
            "final": {"idm_mse": 0.25},
            "best": {"idm_mse": 0.125},
        }

    def fake_run_full_dataset_eval(**kwargs):
        del kwargs
        return {"idm_mse": 0.05}

    monkeypatch.setattr(run_idm_experiments, "run_idm_training", fake_run_idm_training)
    monkeypatch.setattr(run_idm_experiments, "run_full_dataset_eval", fake_run_full_dataset_eval)

    main(
        Args(
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            max_samples=16,
            synthetic_samples=16,
            image_size=32,
            num_future_frames=4,
            action_horizons=(4,),
            epochs=1,
            batch_size=4,
            learning_rates=(3e-4,),
            seeds=(5,),
            device="cpu",
            diagnostics=False,
            idm_future_conditioning="current_only",
        )
    )

    assert len(train_configs) == 1
    assert train_configs[0].model.idm_future_conditioning == "current_only"


def test_run_idm_experiments_forwards_flow_sample_noise_scale(tmp_path, monkeypatch) -> None:
    train_configs = []

    def fake_run_idm_training(config):
        train_configs.append(config)
        return {
            "final": {"idm_mse": 0.25},
            "best": {"idm_mse": 0.125},
        }

    def fake_run_full_dataset_eval(**kwargs):
        del kwargs
        return {"idm_mse": 0.05}

    monkeypatch.setattr(run_idm_experiments, "run_idm_training", fake_run_idm_training)
    monkeypatch.setattr(run_idm_experiments, "run_full_dataset_eval", fake_run_full_dataset_eval)

    main(
        Args(
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            max_samples=16,
            synthetic_samples=16,
            image_size=32,
            num_future_frames=4,
            action_horizons=(4,),
            epochs=1,
            batch_size=4,
            learning_rates=(3e-4,),
            seeds=(5,),
            device="cpu",
            diagnostics=False,
            idm_flow_sample_noise_scale=0.0,
        )
    )

    assert len(train_configs) == 1
    assert train_configs[0].model.idm_flow_sample_noise_scale == pytest.approx(0.0)


def test_run_idm_experiments_forwards_current_conditioning_dropout(tmp_path, monkeypatch) -> None:
    train_configs = []

    def fake_run_idm_training(config):
        train_configs.append(config)
        return {
            "final": {"idm_mse": 0.25},
            "best": {"idm_mse": 0.125},
        }

    def fake_run_full_dataset_eval(**kwargs):
        del kwargs
        return {"idm_mse": 0.05}

    monkeypatch.setattr(run_idm_experiments, "run_idm_training", fake_run_idm_training)
    monkeypatch.setattr(run_idm_experiments, "run_full_dataset_eval", fake_run_full_dataset_eval)

    main(
        Args(
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            max_samples=16,
            synthetic_samples=16,
            image_size=32,
            num_future_frames=4,
            action_horizons=(4,),
            epochs=1,
            batch_size=4,
            learning_rates=(3e-4,),
            seeds=(5,),
            device="cpu",
            diagnostics=False,
            idm_current_frame_dropout=0.25,
            idm_wan_vae_current_latent_dropout=0.5,
        )
    )

    assert len(train_configs) == 1
    assert train_configs[0].idm_current_frame_dropout == pytest.approx(0.25)
    assert train_configs[0].idm_wan_vae_current_latent_dropout == pytest.approx(0.5)


def test_run_idm_experiments_refuses_cached_future_dir_for_training(tmp_path, monkeypatch) -> None:
    def fail_if_training_starts(config):
        del config
        raise AssertionError("training should not start when cached_future_dir is set")

    monkeypatch.setattr(run_idm_experiments, "run_idm_training", fail_if_training_starts)

    with pytest.raises(ValueError, match="Generated/cached futures are for eval/ranking only"):
        main(
            Args(
                dataset_source="synthetic",
                output_dir=str(tmp_path),
                cached_future_dir=str(tmp_path / "cached_futures"),
                max_samples=16,
                synthetic_samples=16,
                image_size=32,
                num_future_frames=4,
                action_horizons=(4,),
                epochs=1,
                batch_size=4,
                learning_rates=(3e-4,),
                seeds=(5,),
                device="cpu",
                diagnostics=False,
            )
        )

    assert not (tmp_path / "experiment_summary.json").exists()
