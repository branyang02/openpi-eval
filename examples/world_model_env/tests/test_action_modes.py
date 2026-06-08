from __future__ import annotations

import io
import json

from describe_wan_action_modes import main, render_markdown
from world_model.action_modes import WanActionMode, get_action_mode_spec, iter_action_mode_specs

EXPECTED_CONTRACTS = {
    WanActionMode.DECODED_VIDEO_IDM: {
        "runs_wan_generation": True,
        "generates_video": True,
        "produces_future_latents": True,
        "consumes_future_pixels": True,
        "uses_wan_prefix_tokens": False,
        "can_decode_for_visual_debugging": True,
        "pi05_style_current_prefix_reuse": False,
        "exposes_reusable_action_memory": False,
        "native_wan_attention_kv_cache": False,
    },
    WanActionMode.CURRENT_WAN_PREFIX_ACTION_EXPERT: {
        "runs_wan_generation": False,
        "generates_video": False,
        "produces_future_latents": False,
        "consumes_future_pixels": False,
        "uses_wan_prefix_tokens": True,
        "can_decode_for_visual_debugging": False,
        "pi05_style_current_prefix_reuse": True,
        "exposes_reusable_action_memory": True,
        "native_wan_attention_kv_cache": False,
    },
    WanActionMode.PARTIAL_WAN_PREFIX_ACTION_EXPERT: {
        "runs_wan_generation": True,
        "generates_video": False,
        "produces_future_latents": True,
        "consumes_future_pixels": False,
        "uses_wan_prefix_tokens": True,
        "can_decode_for_visual_debugging": True,
        "pi05_style_current_prefix_reuse": False,
        "exposes_reusable_action_memory": True,
        "native_wan_attention_kv_cache": False,
    },
}


def test_all_modes_have_expected_contract_flags() -> None:
    specs = iter_action_mode_specs()

    assert tuple(spec.mode for spec in specs) == tuple(WanActionMode)
    for mode, expected in EXPECTED_CONTRACTS.items():
        spec = get_action_mode_spec(mode)
        for field, value in expected.items():
            assert getattr(spec, field) is value


def test_decoded_video_idm_contract_mentions_future_pixel_idm_path() -> None:
    spec = get_action_mode_spec("decoded_video_idm")

    assert spec.wan_inputs == ("current_image", "task_text")
    assert "decoded_future_pixels" in spec.action_inputs
    assert "IDM" in spec.action_decoder
    assert "required action input" in spec.debug_decode


def test_current_prefix_contract_is_current_only_and_reusable() -> None:
    spec = get_action_mode_spec(WanActionMode.CURRENT_WAN_PREFIX_ACTION_EXPERT)

    assert spec.future_latent_slots == "none"
    assert spec.frozen_wan is True
    assert "cached_wan_prefix_memory" in spec.action_inputs
    assert "run once" in spec.wan_process
    assert "pi0.5-style" in spec.summary
    assert "across flow denoising steps" in spec.memory_contract


def test_hybrid_contract_uses_future_latents_without_future_pixels() -> None:
    spec = get_action_mode_spec(WanActionMode.PARTIAL_WAN_PREFIX_ACTION_EXPERT)

    assert "future_latent_noise_or_slots" in spec.wan_inputs
    assert "incomplete denoising" in spec.wan_process
    assert "optional inspection artifact" in spec.debug_decode
    assert "true native Wan attention KV" in spec.memory_contract
    assert "decoded_future_pixels" not in spec.action_inputs


def test_json_cli_outputs_machine_readable_specs() -> None:
    out = io.StringIO()

    assert main(["--format", "json"], out=out) == 0

    payload = json.loads(out.getvalue())
    modes = {row["mode"]: row for row in payload["modes"]}
    assert set(modes) == {mode.value for mode in WanActionMode}
    assert modes["decoded_video_idm"]["generates_video"] is True
    assert modes["current_wan_prefix_action_expert"]["pi05_style_current_prefix_reuse"] is True
    assert modes["current_wan_prefix_action_expert"]["exposes_reusable_action_memory"] is True
    assert modes["partial_wan_prefix_action_expert"]["runs_wan_generation"] is True
    assert modes["partial_wan_prefix_action_expert"]["can_decode_for_visual_debugging"] is True
    assert modes["partial_wan_prefix_action_expert"]["native_wan_attention_kv_cache"] is False


def test_cli_can_filter_markdown_to_one_mode() -> None:
    out = io.StringIO()

    assert main(["--mode", "partial_wan_prefix_action_expert"], out=out) == 0

    rendered = out.getvalue()
    assert "# Wan Action Modes" in rendered
    assert "`partial_wan_prefix_action_expert`" in rendered
    assert "`decoded_video_idm`" not in rendered


def test_markdown_table_renders_all_contract_flags() -> None:
    rendered = render_markdown(iter_action_mode_specs())

    assert (
        "| Mode | Wan generation | Decoded action video | Future pixels | Reusable memory | Native Wan KV |" in rendered
    )
    assert "| `decoded_video_idm` | yes | yes | yes | no | no |" in rendered
    assert "| `current_wan_prefix_action_expert` | no | no | no | yes | no |" in rendered
    assert "| `partial_wan_prefix_action_expert` | yes | no | no | yes | no |" in rendered
    assert "Memory contract:" in rendered
