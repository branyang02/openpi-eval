"""Summarize world-model experiment artifacts found under an ``output/`` tree.

This is a small, dependency-free reporting helper. It recursively scans a
directory for the JSON artifacts emitted by the training/eval/ranking scripts
and distills them into a single concise JSON document (optionally rendered as
Markdown). It never imports the training or model code, so it is safe to run in
a lightweight environment.

Recognized artifacts:

* ``metrics.json``               -- IDM/WM training runs (``train_lib.py``)
* ``ranking_summary.json``       -- Wan LoRA rankings (``validate_wan_lora.py``)
* ``eval_metrics.json``          -- evaluation metrics (``eval_idm.py`` / ``eval.py``)
* ``idm_diagnostics.json``       -- IDM diagnostics (``diagnose_idm.py``)
* ``future_cache_metrics.json``  -- future-rollout cache quality (``evaluate_future_cache.py``)

Usage::

    python summarize_experiments.py --root output
    python summarize_experiments.py --root output --json-out summary.json --markdown-out summary.md
    python summarize_experiments.py --root output --markdown
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

VISUAL_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".mp4", ".webm", ".mov")

# Category order is also the section order used for output and rendering.
CATEGORIES = ("idm_train", "wan_ranking", "eval", "diagnostics", "future_cache")


def summarize_idm_train(data: dict[str, Any]) -> dict[str, Any]:
    """Distill an IDM/WM ``metrics.json`` training record."""
    history = data.get("history") or []
    final = data.get("final") or {}
    entry: dict[str, Any] = {
        "training_target": data.get("training_target"),
        "epochs": len(history) or final.get("epoch"),
        "idm_arch": (data.get("model_config") or {}).get("idm_arch"),
        "final": data.get("final"),
    }
    if "best" in data:
        entry["best"] = data["best"]
    if "stopped_early" in data:
        entry["stopped_early"] = data["stopped_early"]
    return entry


def summarize_wan_ranking(data: dict[str, Any]) -> dict[str, Any]:
    """Distill a Wan LoRA ``ranking_summary.json`` record."""
    runs = data.get("runs") or []
    ranked = data.get("ranked") or runs
    compact = []
    for run in ranked:
        pixel = run.get("pixel") or {}
        idm = run.get("idm") or {}
        compact.append(
            {
                "label": run.get("label"),
                "idm_decodability_gap": run.get("idm_decodability_gap"),
                "idm_mse": idm.get("idm_mse"),
                "future_mse": pixel.get("future_mse"),
                "future_psnr": pixel.get("future_psnr"),
            }
        )
    return {
        "rank_by": data.get("rank_by"),
        "best": data.get("best"),
        "num_runs": len(runs),
        "ground_truth_reference": data.get("ground_truth_reference"),
        "ranked": compact,
    }


def summarize_eval(data: dict[str, Any]) -> dict[str, Any]:
    """Distill an ``eval_metrics.json`` record (eval_idm.py or eval.py shape)."""
    metric_keys = ("idm_mse", "idm_smooth_l1", "idm_generated_mse", "wm_mse", "wm_psnr")
    entry: dict[str, Any] = {"metrics": {k: data[k] for k in metric_keys if k in data}}
    for key in ("checkpoint", "cached_future_dir", "flow_eval_seed"):
        if key in data:
            entry[key] = data[key]
    return entry


def summarize_diagnostics(data: dict[str, Any]) -> dict[str, Any]:
    """Distill an ``idm_diagnostics.json`` record."""
    baseline = data.get("mean_action_baseline") or {}
    sensitivity = data.get("future_sensitivity") or {}
    entry: dict[str, Any] = {
        "idm_mse": data.get("idm_mse"),
        "idm_smooth_l1": data.get("idm_smooth_l1"),
        "num_samples": data.get("num_samples"),
        "num_valid_actions": data.get("num_valid_actions"),
        "idm_arch": (data.get("model_config") or {}).get("idm_arch"),
        "mean_action_baseline": {
            "idm_mse": baseline.get("idm_mse"),
            "idm_smooth_l1": baseline.get("idm_smooth_l1"),
        },
        "future_sensitivity": {name: (vals or {}).get("target_mse") for name, vals in sensitivity.items()},
    }
    if "error_stats" in data:
        entry["error_stats"] = data["error_stats"]
    return entry


def summarize_future_cache(data: dict[str, Any]) -> dict[str, Any]:
    """Distill a ``future_cache_metrics.json`` record."""
    metric_keys = ("num_samples", "future_mse", "future_mae", "future_psnr", "max_abs_error")
    entry: dict[str, Any] = {key: data.get(key) for key in metric_keys}
    entry["cache_dir"] = data.get("cache_dir")
    if data.get("contact_sheet"):
        entry["contact_sheet"] = data["contact_sheet"]
    return entry


# filename -> (category, summarizer)
ARTIFACTS: dict[str, tuple[str, Any]] = {
    "metrics.json": ("idm_train", summarize_idm_train),
    "ranking_summary.json": ("wan_ranking", summarize_wan_ranking),
    "eval_metrics.json": ("eval", summarize_eval),
    "idm_diagnostics.json": ("diagnostics", summarize_diagnostics),
    "future_cache_metrics.json": ("future_cache", summarize_future_cache),
}


def _referenced_visuals(obj: Any) -> list[str]:
    """Recursively collect string values that look like visual artifact paths."""
    found: list[str] = []
    if isinstance(obj, dict):
        for value in obj.values():
            found.extend(_referenced_visuals(value))
    elif isinstance(obj, list):
        for value in obj:
            found.extend(_referenced_visuals(value))
    elif isinstance(obj, str) and obj.lower().endswith(VISUAL_SUFFIXES):
        found.append(obj)
    return found


def collect_visual_artifacts(metric_path: Path, data: Any, root: Path) -> list[str]:
    """Return visual artifacts for a metric file.

    Combines paths referenced inside the JSON (verbatim) with image/video files
    physically present in the metric file's directory (relative to ``root``).
    """
    artifacts = set(_referenced_visuals(data))
    for child in metric_path.parent.iterdir():
        if child.is_file() and child.suffix.lower() in VISUAL_SUFFIXES:
            artifacts.add(child.relative_to(root).as_posix())
    return sorted(artifacts)


def build_summary(root: str | Path) -> dict[str, Any]:
    """Scan ``root`` for known artifacts and return a concise summary dict."""
    root = Path(root)
    buckets: dict[str, list[dict[str, Any]]] = {category: [] for category in CATEGORIES}
    errors: list[dict[str, str]] = []

    for filename, (category, summarizer) in ARTIFACTS.items():
        for path in sorted(root.rglob(filename)):
            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError) as exc:
                errors.append({"path": path.relative_to(root).as_posix(), "error": str(exc)})
                continue
            entry: dict[str, Any] = {
                "experiment": path.parent.relative_to(root).as_posix(),
                "path": path.relative_to(root).as_posix(),
            }
            entry.update(summarizer(data))
            entry["visual_artifacts"] = collect_visual_artifacts(path, data, root)
            buckets[category].append(entry)

    summary: dict[str, Any] = {"root": str(root), "counts": {}}
    for category in CATEGORIES:
        items = sorted(buckets[category], key=lambda e: (e["experiment"], e["path"]))
        summary[category] = items
        summary["counts"][category] = len(items)
    summary["errors"] = sorted(errors, key=lambda e: e["path"])
    return summary


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def render_markdown(summary: dict[str, Any]) -> str:
    """Render the summary dict as a concise Markdown report."""
    lines: list[str] = ["# Experiment Summary", ""]
    lines.append(f"Scanned `{summary['root']}`")
    counts = summary["counts"]
    lines.append("")
    lines.append("Counts: " + ", ".join(f"{cat}={counts[cat]}" for cat in CATEGORIES))
    lines.append("")

    lines.append(f"## IDM Training Runs ({counts['idm_train']})")
    for entry in summary["idm_train"]:
        final = entry.get("final") or {}
        best = entry.get("best") or {}
        lines.append(
            f"- **{entry['experiment']}** — arch={_fmt(entry.get('idm_arch'))}, "
            f"epochs={_fmt(entry.get('epochs'))}, "
            f"final idm_mse={_fmt(final.get('idm_mse'))}, "
            f"best idm_mse={_fmt(best.get('idm_mse'))}"
        )
    lines.append("")

    lines.append(f"## Wan Rankings ({counts['wan_ranking']})")
    for entry in summary["wan_ranking"]:
        best = entry.get("best") or {}
        lines.append(f"### {entry['experiment']} (rank_by={_fmt(entry.get('rank_by'))})")
        lines.append(f"Best: {_fmt(best)}")
        lines.append("")
        lines.append("| label | decodability_gap | idm_mse | future_mse | future_psnr |")
        lines.append("| --- | --- | --- | --- | --- |")
        for run in entry.get("ranked", []):
            lines.append(
                f"| {_fmt(run.get('label'))} | {_fmt(run.get('idm_decodability_gap'))} "
                f"| {_fmt(run.get('idm_mse'))} | {_fmt(run.get('future_mse'))} "
                f"| {_fmt(run.get('future_psnr'))} |"
            )
        lines.append("")

    lines.append(f"## Eval Metrics ({counts['eval']})")
    for entry in summary["eval"]:
        metrics = entry.get("metrics") or {}
        rendered = ", ".join(f"{k}={_fmt(v)}" for k, v in metrics.items()) or "n/a"
        lines.append(f"- **{entry['experiment']}** — {rendered}")
    lines.append("")

    lines.append(f"## Diagnostics ({counts['diagnostics']})")
    for entry in summary["diagnostics"]:
        baseline = entry.get("mean_action_baseline") or {}
        sensitivity = entry.get("future_sensitivity") or {}
        sens = ", ".join(f"{k}={_fmt(v)}" for k, v in sensitivity.items()) or "n/a"
        lines.append(
            f"- **{entry['experiment']}** — idm_mse={_fmt(entry.get('idm_mse'))} "
            f"(baseline {_fmt(baseline.get('idm_mse'))}); future_sensitivity target_mse: {sens}"
        )
    lines.append("")

    lines.append(f"## Future Cache ({counts['future_cache']})")
    for entry in summary["future_cache"]:
        lines.append(
            f"- **{entry['experiment']}** — future_mse={_fmt(entry.get('future_mse'))}, "
            f"future_psnr={_fmt(entry.get('future_psnr'))}, n={_fmt(entry.get('num_samples'))}"
        )
    lines.append("")

    lines.append("## Visual Artifacts")
    for category in CATEGORIES:
        for entry in summary[category]:
            artifacts = entry.get("visual_artifacts") or []
            if artifacts:
                lines.append(f"- **{entry['experiment']}**: " + ", ".join(artifacts))
    lines.append("")

    errors = summary.get("errors") or []
    if errors:
        lines.append(f"## Unreadable Files ({len(errors)})")
        for error in errors:
            lines.append(f"- {error['path']}: {error['error']}")
        lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize world-model experiment artifacts.")
    parser.add_argument("--root", default="output", help="Directory to scan (default: output)")
    parser.add_argument("--json-out", default=None, help="Write the JSON summary to this path")
    parser.add_argument("--markdown-out", default=None, help="Write the Markdown summary to this path")
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Print Markdown to stdout instead of JSON",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indent (default: 2)")
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        parser.error(f"root directory does not exist: {root}")

    summary = build_summary(root)
    json_text = json.dumps(summary, indent=args.indent)
    markdown_text = render_markdown(summary)

    if args.json_out:
        Path(args.json_out).write_text(json_text + "\n")
    if args.markdown_out:
        Path(args.markdown_out).write_text(markdown_text + "\n")

    print(markdown_text if args.markdown else json_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
