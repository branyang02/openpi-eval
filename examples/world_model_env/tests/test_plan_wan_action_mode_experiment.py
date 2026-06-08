from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

WORLD_MODEL_ENV_DIR = Path(__file__).resolve().parents[1]
if str(WORLD_MODEL_ENV_DIR) not in sys.path:
    sys.path.insert(0, str(WORLD_MODEL_ENV_DIR))

from plan_wan_action_mode_experiment import PlannerConfig, main, planned_commands, render_plan  # noqa: E402


def _one_line(text: str) -> str:
    return " ".join(text.replace("\\\n", " ").split())


def test_render_plan_contains_matched_paths_flags_matrix_and_kv_caveat() -> None:
    config = PlannerConfig(
        output_root=Path("output/matched_32"),
        sample_count=32,
        idm_checkpoint="output/idm.pt",
        current_action_expert_checkpoint="output/current/checkpoint.pt",
        partial_action_expert_checkpoint="output/partial/checkpoint.pt",
        wan_lora_path="output/wan_lora/epoch-3.safetensors",
    )

    rendered = render_plan(config)
    one_line = _one_line(rendered)

    assert "Sample fingerprint:" in rendered
    assert "dataset_source=lerobot" in rendered
    assert "image_key=corner4.image" in rendered
    assert "episodes=[16, 17, 18, 19, 20, 21, 22, 23]" in rendered
    assert "samples_per_episode=32" in rendered
    assert "planned_samples=256" in rendered
    assert "action_horizon=4" in rendered
    assert "prefix_dim=3072" in rendered
    assert "selected_layers=[0, 14, 29]" in rendered
    assert "hidden_pool=mean" in rendered
    assert "hidden_pool_config=model_fn_wan_video_block_hooks_mean_pool_selected_layers" in rendered
    assert "tokens_per_layer=1" in rendered
    assert "dit_timestep=500.0" in rendered
    assert "partial_num_latent_frames=2" in rendered
    assert "True native Wan attention KV cache is not implemented here" in rendered
    assert "not cached Wan attention KV" in rendered

    assert "CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=/tmp/uv-cache uv run python cache_future_rollouts.py" in one_line
    assert "CUDA_VISIBLE_DEVICES=1 UV_CACHE_DIR=/tmp/uv-cache uv run python cache_pi05_wan_prefix_tokens.py" in one_line
    assert "--future-source wan_lora" in one_line
    assert "--cached-future-dir output/matched_32/decoded_video_idm/future_cache" in one_line
    assert "--samples-per-episode 32" in one_line
    assert "--max-samples 32" not in one_line
    assert "--max-samples 256" not in one_line
    assert "--prefix-dim 3072" in one_line
    assert "--prefix-backend dit_hidden" in one_line
    assert "--dit-selected-layers 0 14 29" in one_line
    assert "--dit-hidden-pool mean" in one_line
    assert "--dit-tokens-per-layer 1" in one_line
    assert "--dit-timestep 500.0" in one_line
    assert "--dit-num-latent-frames 1" in one_line
    assert "--dit-num-latent-frames 2" in one_line
    assert "--dit-future-latent-fill noise" in one_line
    assert "--dit-future-latent-seed 0" in one_line
    assert "--dit-num-latent-frames 5" not in one_line
    assert "--prefix-dim 48" not in one_line

    assert "run_wan_action_mode_matrix.py" in one_line
    assert "--decoded-video-idm output/matched_32/decoded_video_idm/eval" in one_line
    assert "--current-wan-prefix-action-expert output/matched_32/current_wan_prefix_action_expert/eval" in one_line
    assert "--partial-wan-prefix-action-expert output/matched_32/partial_wan_prefix_action_expert/eval" in one_line
    assert "--output-json output/matched_32/wan_action_mode_matrix.json" in one_line
    assert "sample_sets_match=true" in rendered


def test_planned_commands_reject_small_or_non_matching_contracts() -> None:
    with pytest.raises(ValueError, match="larger than the current matched n=8"):
        planned_commands(PlannerConfig(output_root=Path("output/small"), sample_count=8, samples_per_episode=None))

    with pytest.raises(ValueError, match="larger than the current matched n=8"):
        planned_commands(PlannerConfig(output_root=Path("output/spe1"), samples_per_episode=1))

    with pytest.raises(ValueError, match="keep frame_delta=1"):
        planned_commands(PlannerConfig(output_root=Path("output/fd2"), frame_delta=2))


def test_main_prints_plan_with_overrides() -> None:
    out = io.StringIO()

    status = main(
        [
            "--output-root",
            "output/custom",
            "--sample-count",
            "64",
            "--episodes",
            "20",
            "21",
            "--current-action-expert-checkpoint",
            "output/current64/checkpoint.pt",
            "--partial-action-expert-checkpoint",
            "output/partial64/checkpoint.pt",
        ],
        out=out,
    )

    rendered = out.getvalue()
    one_line = _one_line(rendered)
    assert status == 0
    assert "episodes=[20, 21]" in rendered
    assert "samples_per_episode=32" in rendered
    assert "planned_samples=64" in rendered
    assert "--checkpoint output/current64/checkpoint.pt" in one_line
    assert "--checkpoint output/partial64/checkpoint.pt" in one_line
    assert "--output-json output/custom/wan_action_mode_matrix.json" in one_line


def test_main_defaults_use_existing_matched_eval_checkpoints() -> None:
    out = io.StringIO()

    status = main([], out=out)

    rendered = out.getvalue()
    one_line = _one_line(rendered)
    assert status == 0
    assert "planned_samples=256" in rendered
    assert "--checkpoint output/idm_flow_patch_decoded_wan_smoke_ep0_15_h4/best_idm_checkpoint.pt" in one_line
    assert (
        "--checkpoint "
        "output/pi05_wan_action_expert_dit_train2_eval1_spe4_h4_prefixonly_norm_seed109_e300_h512_l6/checkpoint.pt"
        in one_line
    )
    assert (
        "--checkpoint "
        "output/pi05_wan_partial_action_expert_train2_eval1_spe4_h4_lat2_noise_seed0_prefixonly_norm_seed109_e300_h512_l6/checkpoint.pt"
        in one_line
    )
