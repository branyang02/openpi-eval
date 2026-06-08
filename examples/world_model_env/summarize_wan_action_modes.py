"""Summarize Wan action-mode experiment results as Markdown."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from world_model.action_modes import WanActionMode, WanActionModeSpec, get_action_mode_spec, iter_action_mode_specs

MODE_KEYS = ("wan_action_mode", "action_mode", "mode")
DECODER_ARCH_KEY = "decoder_arch"
DECODER_ARCH_CONTAINERS = ("metadata", "result_metadata", "metrics", "args", "model_kwargs")
CHECKPOINT_PATH_KEYS = ("checkpoint", "checkpoint_path")
MSE_KEYS = (
    "dataset_action_mse",
    "mse",
    "val_model_sample_mse",
    "val_model_zero_noise_mse",
    "model_sample_mse",
    "model_zero_noise_mse",
)
PER_DIM_MSE_KEYS = (
    "dataset_action_mse_per_action_dim",
    "per_dim_mse",
    "val_model_zero_noise_mse_per_action_dim",
    "model_zero_noise_mse_per_action_dim",
    "per_action_dim_mse",
)
MODE_INFERENCE_PATHS = {
    WanActionMode.DECODED_VIDEO_IDM: "full Wan video generation -> IDM",
    WanActionMode.CURRENT_WAN_PREFIX_ACTION_EXPERT: "current Wan prefix run once -> action expert",
    WanActionMode.PARTIAL_WAN_PREFIX_ACTION_EXPERT: "hybrid Wan future latents/memory -> action expert",
}


@dataclass(frozen=True, slots=True)
class WanActionModeResult:
    """One action-mode result row with its resolved contract."""

    mode: WanActionMode
    spec: WanActionModeSpec
    mse: Any
    per_dim_mse: Any
    decoder_arch: str | None
    path: Path


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def inference_path_for_mode(mode: WanActionMode) -> str:
    """Return the concise reporting label for a Wan action-mode concept."""
    return MODE_INFERENCE_PATHS[mode]


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_fmt(item) for item in value) + "]"
    return str(value)


def _validate_mode(mode_value: str) -> WanActionMode:
    try:
        return WanActionMode(mode_value)
    except ValueError as exc:
        valid = ", ".join(mode.value for mode in WanActionMode)
        raise ValueError(f"unknown Wan action mode {mode_value!r}; expected one of: {valid}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except OSError as exc:
        raise ValueError(f"failed to read result JSON {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"failed to parse result JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"result JSON must contain an object: {path}")
    return data


def _first_present(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _decoder_arch_fields(data: dict[str, Any]) -> list[tuple[str, Any]]:
    fields = []
    if DECODER_ARCH_KEY in data:
        fields.append((DECODER_ARCH_KEY, data[DECODER_ARCH_KEY]))
    for container_key in DECODER_ARCH_CONTAINERS:
        container = data.get(container_key)
        if isinstance(container, dict) and DECODER_ARCH_KEY in container:
            fields.append((f"{container_key}.{DECODER_ARCH_KEY}", container[DECODER_ARCH_KEY]))
    return fields


def _decoder_arch_from_mapping(data: dict[str, Any], path: Path) -> str | None:
    values = []
    for source_field, value in _decoder_arch_fields(data):
        if value is None:
            continue
        if not isinstance(value, str) or not value:
            raise ValueError(f"decoder_arch field {source_field!r} in {path} must be a non-empty string")
        values.append((source_field, value))
    if not values:
        return None

    unique_values = {value for _source_field, value in values}
    if len(unique_values) != 1:
        details = ", ".join(f"{source_field}={value!r}" for source_field, value in values)
        raise ValueError(f"decoder_arch metadata in {path} is inconsistent: {details}")
    return values[0][1]


def _checkpoint_path_value(data: dict[str, Any], path: Path) -> str | None:
    for key in CHECKPOINT_PATH_KEYS:
        if key not in data:
            continue
        value = data[key]
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise ValueError(f"checkpoint path field {key!r} in {path} must be a non-empty string or null")
        return value
    return None


def _existing_checkpoint_path(value: str, result_path: Path) -> Path | None:
    checkpoint_path = Path(value).expanduser()
    if checkpoint_path.is_absolute():
        return checkpoint_path if checkpoint_path.is_file() else None

    candidates = [checkpoint_path, result_path.parent / checkpoint_path]
    candidates.extend(parent / checkpoint_path for parent in result_path.parents)
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate
    return None


def _read_checkpoint(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ValueError(f"failed to read checkpoint {path}: PyTorch is not installed") from exc

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"failed to read checkpoint {path}: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint must contain a mapping: {path}")
    return checkpoint


def extract_decoder_arch(data: dict[str, Any], path: Path, *, follow_checkpoint: bool = True) -> str | None:
    """Return explicitly recorded decoder architecture metadata, when present."""
    decoder_arch = _decoder_arch_from_mapping(data, path)
    if decoder_arch is not None:
        return decoder_arch

    if not follow_checkpoint:
        return None
    checkpoint_value = _checkpoint_path_value(data, path)
    if checkpoint_value is None:
        return None
    checkpoint_path = _existing_checkpoint_path(checkpoint_value, path)
    if checkpoint_path is None:
        return None
    return _decoder_arch_from_mapping(_read_checkpoint(checkpoint_path), checkpoint_path)


def _extract_mode(data: dict[str, Any], path: Path, explicit_mode: str | None) -> WanActionMode:
    if explicit_mode is not None:
        return _validate_mode(explicit_mode)

    mode_value = _first_present(data, MODE_KEYS)
    if mode_value is None:
        keys = ", ".join(MODE_KEYS)
        raise ValueError(f"missing Wan action mode for {path}; pass --mode or include one of: {keys}")
    if not isinstance(mode_value, str):
        raise ValueError(f"Wan action mode in {path} must be a string")
    return _validate_mode(mode_value)


def _resolve_explicit_modes(paths: list[Path], modes: list[str] | None) -> list[str | None]:
    if not modes:
        return [None] * len(paths)
    for mode in modes:
        _validate_mode(mode)
    if len(modes) == 1:
        return modes * len(paths)
    if len(modes) != len(paths):
        raise ValueError(f"--mode must be passed once or once per result path; got {len(modes)} modes for {len(paths)} paths")
    return modes


def load_results(paths: list[str | Path], modes: list[str] | None = None) -> tuple[WanActionModeResult, ...]:
    """Load result JSON paths and resolve their Wan action modes."""
    result_paths = [Path(path) for path in paths]
    explicit_modes = _resolve_explicit_modes(result_paths, modes)
    rows: list[WanActionModeResult] = []
    for path, explicit_mode in zip(result_paths, explicit_modes, strict=True):
        data = _read_json(path)
        mode = _extract_mode(data, path, explicit_mode)
        rows.append(
            WanActionModeResult(
                mode=mode,
                spec=get_action_mode_spec(mode),
                mse=_first_present(data, MSE_KEYS),
                per_dim_mse=_first_present(data, PER_DIM_MSE_KEYS),
                decoder_arch=extract_decoder_arch(data, path, follow_checkpoint=mode != WanActionMode.DECODED_VIDEO_IDM),
                path=path,
            )
        )
    return tuple(rows)


def render_contract_table(specs: tuple[WanActionModeSpec, ...] | None = None) -> str:
    """Render the built-in Wan action-mode contracts as Markdown."""
    specs = specs or iter_action_mode_specs()
    lines = [
        "| mode | inference_path | runs_wan_generation | decoded_action_video | consumes_future_pixels | reusable_action_memory | native_wan_kv |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for spec in specs:
        lines.append(
            f"| `{spec.mode.value}` | {inference_path_for_mode(spec.mode)} | "
            f"{_yes_no(spec.runs_wan_generation)} | {_yes_no(spec.generates_video)} | "
            f"{_yes_no(spec.consumes_future_pixels)} | {_yes_no(spec.exposes_reusable_action_memory)} | "
            f"{_yes_no(spec.native_wan_attention_kv_cache)} |"
        )
    return "\n".join(lines) + "\n"


def render_result_table(rows: tuple[WanActionModeResult, ...]) -> str:
    """Render result rows as a concise Markdown table."""
    lines = [
        "| mode | decoder_arch | inference_path | runs_wan_generation | decoded_action_video | consumes_future_pixels | reusable_action_memory | native_wan_kv | mse | per_dim_mse | result path |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row.mode.value}` | {_fmt(row.decoder_arch)} | {inference_path_for_mode(row.mode)} | "
            f"{_yes_no(row.spec.runs_wan_generation)} | {_yes_no(row.spec.generates_video)} | "
            f"{_yes_no(row.spec.consumes_future_pixels)} | {_yes_no(row.spec.exposes_reusable_action_memory)} | "
            f"{_yes_no(row.spec.native_wan_attention_kv_cache)} | {_fmt(row.mse)} | "
            f"{_fmt(row.per_dim_mse)} | `{row.path}` |"
        )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Wan action-mode experiment results.")
    parser.add_argument("results", nargs="*", help="Result JSON path(s) to summarize.")
    parser.add_argument(
        "--mode",
        action="append",
        choices=[mode.value for mode in WanActionMode],
        help="Wan action mode label. Pass once for all result paths or once per path.",
    )
    parser.add_argument(
        "--list-modes",
        action="store_true",
        help="Print the built-in Wan action-mode contract table.",
    )
    return parser


def main(argv: list[str] | None = None, *, out: TextIO | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        rendered = render_contract_table() if args.list_modes else render_result_table(load_results(args.results, args.mode))
    except ValueError as exc:
        parser.error(str(exc))

    print(rendered, file=out or sys.stdout, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
