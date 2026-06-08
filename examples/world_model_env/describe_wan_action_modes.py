"""Describe supported Wan action inference modes."""

from __future__ import annotations

import argparse
import json
import sys
from typing import TextIO

from world_model.action_modes import WanActionMode, WanActionModeSpec, get_action_mode_spec, iter_action_mode_specs


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def render_json(specs: tuple[WanActionModeSpec, ...]) -> str:
    """Render mode specs as stable JSON."""
    return json.dumps({"modes": [spec.to_dict() for spec in specs]}, indent=2)


def render_markdown(specs: tuple[WanActionModeSpec, ...]) -> str:
    """Render mode specs as concise Markdown."""
    lines = [
        "# Wan Action Modes",
        "",
        "| Mode | Wan generation | Decoded action video | Future pixels | Reusable memory | Native Wan KV |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for spec in specs:
        lines.append(
            "| "
            f"`{spec.mode.value}` | "
            f"{_yes_no(spec.runs_wan_generation)} | "
            f"{_yes_no(spec.generates_video)} | "
            f"{_yes_no(spec.consumes_future_pixels)} | "
            f"{_yes_no(spec.exposes_reusable_action_memory)} | "
            f"{_yes_no(spec.native_wan_attention_kv_cache)} |"
        )

    for spec in specs:
        lines.extend(
            [
                "",
                f"## {spec.name}",
                "",
                spec.summary,
                "",
                f"- Mode: `{spec.mode.value}`",
                f"- Wan inputs: {', '.join(spec.wan_inputs)}",
                f"- Wan process: {spec.wan_process}",
                f"- Action decoder: {spec.action_decoder}",
                f"- Action inputs: {', '.join(spec.action_inputs)}",
                f"- Produces future latents: {_yes_no(spec.produces_future_latents)}",
                f"- Debug decode: {spec.debug_decode}",
                f"- pi0.5 current-prefix reuse: {_yes_no(spec.pi05_style_current_prefix_reuse)}",
                f"- Future latent slots: {spec.future_latent_slots}",
                f"- Memory contract: {spec.memory_contract}",
            ]
        )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Describe supported Wan action inference modes.")
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format.",
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=[mode.value for mode in WanActionMode],
        help="Limit output to one mode. May be passed more than once.",
    )
    return parser


def _select_specs(mode_values: list[str] | None) -> tuple[WanActionModeSpec, ...]:
    if not mode_values:
        return iter_action_mode_specs()
    return tuple(get_action_mode_spec(mode) for mode in mode_values)


def main(argv: list[str] | None = None, *, out: TextIO | None = None) -> int:
    args = _build_parser().parse_args(argv)
    specs = _select_specs(args.mode)
    rendered = render_json(specs) if args.format == "json" else render_markdown(specs)
    print(rendered, file=out or sys.stdout, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
