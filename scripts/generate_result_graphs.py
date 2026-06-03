"""Generate README result charts from local release evaluation outputs."""

from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "assets" / "results"

BLUE = "#2563eb"
GREEN = "#16a34a"
ORANGE = "#f97316"
PURPLE = "#7c3aed"
GRID = "#e5e7eb"
TEXT = "#111827"
MUTED = "#6b7280"


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def load_mean(path: Path) -> float:
    return float(load_json(path)["mean_success_rate"])


def load_per_task(path: Path) -> list[tuple[str, float]]:
    data = load_json(path)
    per_task = data["per_task"]
    if isinstance(per_task, dict):
        return [(name, float(value)) for name, value in per_task.items()]

    rows = []
    for item in per_task:
        label = item.get("task_description") or item["task_name"]
        rows.append((label, float(item["success_rate"])))
    return rows


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def short_label(label: str, max_len: int = 58) -> str:
    if len(label) <= max_len:
        return label
    return f"{label[: max_len - 3]}..."


def render_bar_chart(
    *,
    title: str,
    subtitle: str,
    rows: list[tuple[str, float, str]],
    output_path: Path,
    width: int = 980,
    label_width: int = 360,
    row_height: int = 28,
) -> None:
    top = 88
    bottom = 46
    chart_x = label_width
    chart_width = width - chart_x - 96
    height = top + len(rows) * row_height + bottom
    title_id = output_path.stem.replace("_", "-")

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="{title_id}-title {title_id}-desc">',
        f'<title id="{title_id}-title">{html.escape(title)}</title>',
        f'<desc id="{title_id}-desc">{html.escape(subtitle)}</desc>',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="24" y="34" font-family="Arial, Helvetica, sans-serif" font-size="20" '
        f'font-weight="700" fill="{TEXT}">{html.escape(title)}</text>',
        f'<text x="24" y="58" font-family="Arial, Helvetica, sans-serif" font-size="13" '
        f'fill="{MUTED}">{html.escape(subtitle)}</text>',
    ]

    for tick in (0, 25, 50, 75, 100):
        x = chart_x + chart_width * tick / 100
        parts.extend(
            [
                f'<line x1="{x:.1f}" y1="72" x2="{x:.1f}" y2="{height - bottom + 8}" '
                f'stroke="{GRID}" stroke-width="1"/>',
                f'<text x="{x:.1f}" y="{height - 18}" text-anchor="middle" '
                f'font-family="Arial, Helvetica, sans-serif" font-size="11" fill="{MUTED}">{tick}</text>',
            ]
        )

    for index, (label, value, color) in enumerate(rows):
        y = top + index * row_height
        bar_width = max(0, min(chart_width, chart_width * value))
        label_text = html.escape(short_label(label))
        parts.extend(
            [
                f'<text x="{chart_x - 16}" y="{y + 17}" text-anchor="end" '
                f'font-family="Arial, Helvetica, sans-serif" font-size="12" fill="{TEXT}">{label_text}</text>',
                f'<rect x="{chart_x}" y="{y + 6}" width="{chart_width}" height="14" rx="2" fill="#f3f4f6"/>',
                f'<rect x="{chart_x}" y="{y + 6}" width="{bar_width:.1f}" height="14" rx="2" fill="{color}"/>',
                f'<text x="{chart_x + bar_width + 8:.1f}" y="{y + 17}" '
                f'font-family="Arial, Helvetica, sans-serif" font-size="12" fill="{TEXT}">{pct(value)}</text>',
            ]
        )

    parts.append(
        f'<text x="{chart_x + chart_width / 2:.1f}" y="{height - 4}" text-anchor="middle" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="11" fill="{MUTED}">success rate (%)</text>'
    )
    parts.append("</svg>")
    output_path.write_text("\n".join(parts) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    metaworld_path = ROOT / "examples" / "metaworld" / "output" / "pi05_metaworld-ML45-train" / "results.json"
    libero_paths = {
        "pi05 spatial": ROOT / "examples" / "libero_env" / "output" / "pi05_libero-libero_spatial" / "results.json",
        "pi05 object": ROOT / "examples" / "libero_env" / "output" / "pi05_libero-libero_object" / "results.json",
        "pi05 goal": ROOT / "examples" / "libero_env" / "output" / "pi05_libero-libero_goal" / "results.json",
        "pi05 10": ROOT / "examples" / "libero_env" / "output" / "pi05_libero-libero_10" / "results.json",
        "pi0-FAST spatial": ROOT
        / "examples"
        / "libero_env"
        / "output"
        / "pi0_fast_libero-libero_spatial"
        / "results.json",
        "pi0-FAST object": ROOT
        / "examples"
        / "libero_env"
        / "output"
        / "pi0_fast_libero-libero_object"
        / "results.json",
        "pi0-FAST goal": ROOT / "examples" / "libero_env" / "output" / "pi0_fast_libero-libero_goal" / "results.json",
        "pi0-FAST 10": ROOT / "examples" / "libero_env" / "output" / "pi0_fast_libero-libero_10" / "results.json",
    }
    robocasa_paths = {
        "atomic_seen": ROOT
        / "examples"
        / "robocasa_env"
        / "output"
        / "pi05_robocasa-atomic_seen-pretrain"
        / "results.json",
        "composite_seen": ROOT
        / "examples"
        / "robocasa_env"
        / "output"
        / "pi05_robocasa-composite_seen-pretrain"
        / "results.json",
        "composite_unseen": ROOT
        / "examples"
        / "robocasa_env"
        / "output"
        / "pi05_robocasa-composite_unseen-pretrain"
        / "results.json",
    }

    metaworld_mean = load_mean(metaworld_path)
    libero_means = {name: load_mean(path) for name, path in libero_paths.items()}
    robocasa_means = {name: load_mean(path) for name, path in robocasa_paths.items()}

    overview_rows = [("MetaWorld pi05 ML45 train", metaworld_mean, BLUE)]
    overview_rows.extend(
        (f"LIBERO {name}", value, BLUE if name.startswith("pi05") else GREEN) for name, value in libero_means.items()
    )
    overview_rows.extend((f"RoboCasa pi05 {name}", value, ORANGE) for name, value in robocasa_means.items())
    render_bar_chart(
        title="Release Evaluation Results",
        subtitle="Mean success rates from current results.json outputs; released checkpoints only.",
        rows=overview_rows,
        output_path=OUT_DIR / "release_evaluation_overview.svg",
        label_width=310,
    )

    metaworld_tasks = sorted(load_per_task(metaworld_path), key=lambda item: (-item[1], item[0]))
    render_bar_chart(
        title="MetaWorld pi05 ML45 Train",
        subtitle=f"Mean success rate: {pct(metaworld_mean)} across {len(metaworld_tasks)} tasks.",
        rows=[(name, value, BLUE) for name, value in metaworld_tasks],
        output_path=OUT_DIR / "metaworld_pi05_ml45_tasks.svg",
        label_width=300,
        row_height=24,
    )

    render_bar_chart(
        title="LIBERO Suite Success",
        subtitle="Mean success rates for released pi05 and pi0-FAST checkpoints.",
        rows=[(name, value, BLUE if name.startswith("pi05") else GREEN) for name, value in libero_means.items()],
        output_path=OUT_DIR / "libero_suite_success.svg",
        label_width=220,
    )

    render_bar_chart(
        title="RoboCasa pi05 Task Sets",
        subtitle="Mean success rates for atomic and composite pretrain evaluations.",
        rows=[(name, value, ORANGE) for name, value in robocasa_means.items()],
        output_path=OUT_DIR / "robocasa_task_set_success.svg",
        label_width=220,
    )

    atomic_rows = sorted(load_per_task(robocasa_paths["atomic_seen"]), key=lambda item: (-item[1], item[0]))
    render_bar_chart(
        title="RoboCasa pi05 Atomic Seen Tasks",
        subtitle=f"Mean success rate: {pct(robocasa_means['atomic_seen'])} across {len(atomic_rows)} tasks.",
        rows=[(name, value, PURPLE) for name, value in atomic_rows],
        output_path=OUT_DIR / "robocasa_atomic_seen_tasks.svg",
        label_width=270,
    )


if __name__ == "__main__":
    main()
