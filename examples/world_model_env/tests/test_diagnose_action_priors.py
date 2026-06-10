from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from diagnose_action_priors import diagnose_action_priors


def _constant_actions(value: float) -> torch.Tensor:
    return torch.full((2, 2), value, dtype=torch.float32)


def _write_row(
    cache_dir: Path,
    index: int,
    *,
    state: list[float],
    actions: torch.Tensor,
    task: str,
    metadata: dict[str, Any] | None = None,
    action_mask: list[float] | None = None,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "prefix_tokens": torch.zeros((3, 4), dtype=torch.float32),
            "state": torch.tensor(state, dtype=torch.float32),
            "actions": actions.to(dtype=torch.float32),
            "action_mask": torch.tensor(action_mask or [1.0, 1.0], dtype=torch.float32),
            "task": task,
            "metadata": metadata or {},
        },
        cache_dir / f"sample_{index:06d}.pt",
    )


def test_action_prior_diagnostic_reports_metrics_fallbacks_coverage_and_files(tmp_path: Path) -> None:
    train_cache = tmp_path / "train"
    eval_cache = tmp_path / "eval"
    output_dir = tmp_path / "out"

    _write_row(
        train_cache,
        0,
        state=[0.0, 0.0],
        actions=_constant_actions(0.0),
        task="task zero",
        metadata={"task_index": 0, "frame_index": 0, "task": "task zero"},
    )
    _write_row(
        train_cache,
        1,
        state=[2.0, 0.0],
        actions=_constant_actions(2.0),
        task="task zero",
        metadata={"task_index": 0, "frame_index": 1, "task": "task zero"},
    )
    _write_row(
        train_cache,
        2,
        state=[10.0, 0.0],
        actions=_constant_actions(10.0),
        task="task one",
        metadata={"task_index": 1, "frame_index": 0, "task": "task one"},
    )
    _write_row(
        eval_cache,
        0,
        state=[0.1, 0.0],
        actions=_constant_actions(0.0),
        task="task zero",
        metadata={"task_index": 0, "frame_index": 0, "task": "task zero"},
    )
    _write_row(
        eval_cache,
        1,
        state=[2.1, 0.0],
        actions=_constant_actions(2.0),
        task="task zero",
        metadata={"task_index": 0, "frame_index": 99, "task": "task zero"},
    )
    _write_row(
        eval_cache,
        2,
        state=[10.1, 0.0],
        actions=_constant_actions(4.0),
        task="new task",
        metadata={"frame_index": 5, "task": "new task"},
    )

    report = diagnose_action_priors(
        train_cache_path=train_cache,
        eval_cache_path=eval_cache,
        output_dir=output_dir,
    )
    baselines = report["baselines"]

    assert report["num_train_rows"] == 3
    assert report["num_eval_rows"] == 3
    assert baselines["global_mean"]["mse"] == pytest.approx(20.0 / 3.0)
    assert baselines["global_mean"]["per_action_dim_mse"] == pytest.approx([20.0 / 3.0, 20.0 / 3.0])
    assert baselines["task_mean"]["mse"] == pytest.approx(2.0 / 3.0)
    assert baselines["task_mean"]["prediction_source_counts"] == {"task_mean": 2, "global_mean": 1}
    assert baselines["task_frame_mean"]["mse"] == pytest.approx(1.0 / 3.0)
    assert baselines["task_frame_mean"]["coverage_count"] == 1
    assert baselines["task_frame_mean"]["coverage_fraction"] == pytest.approx(1.0 / 3.0)
    assert baselines["task_frame_mean"]["prediction_source_counts"] == {
        "task_frame_mean": 1,
        "task_mean": 1,
        "global_mean": 1,
    }
    assert baselines["nearest_state"]["mse"] == pytest.approx(12.0)
    assert baselines["nearest_state_same_task"]["mse"] == pytest.approx(12.0)
    assert baselines["nearest_state_same_task"]["prediction_source_counts"] == {
        "same_task": 2,
        "nearest_state": 1,
    }
    assert baselines["nearest_state_same_task"]["fallback_fraction"] == pytest.approx(1.0 / 3.0)

    metrics_path = output_dir / "action_prior_metrics.json"
    summary_path = output_dir / "action_prior_summary.md"
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary = summary_path.read_text(encoding="utf-8")

    assert metrics_path.exists()
    assert summary_path.exists()
    assert saved["baselines"]["task_frame_mean"]["coverage_fraction"] == pytest.approx(1.0 / 3.0)
    assert "| task_frame_mean | 0.333333 | [0.333333, 0.333333] | exact 1/3 (0.333)" in summary


def test_task_mean_uses_task_string_when_task_index_is_missing(tmp_path: Path) -> None:
    train_cache = tmp_path / "train"
    eval_cache = tmp_path / "eval"

    _write_row(
        train_cache,
        0,
        state=[0.0],
        actions=_constant_actions(5.0),
        task="open drawer",
        metadata={},
    )
    _write_row(
        train_cache,
        1,
        state=[10.0],
        actions=_constant_actions(1.0),
        task="close drawer",
        metadata={},
    )
    _write_row(
        eval_cache,
        0,
        state=[3.0],
        actions=_constant_actions(5.0),
        task="open drawer",
        metadata={},
    )

    report = diagnose_action_priors(
        train_cache_path=train_cache,
        eval_cache_path=eval_cache,
        output_dir=tmp_path / "out",
    )

    assert report["baselines"]["global_mean"]["mse"] == pytest.approx(4.0)
    assert report["baselines"]["task_mean"]["mse"] == pytest.approx(0.0)
    assert report["baselines"]["task_mean"]["prediction_source_counts"] == {"task_mean": 1, "global_mean": 0}


def test_action_masks_control_train_means_and_eval_metrics(tmp_path: Path) -> None:
    train_cache = tmp_path / "train"
    eval_cache = tmp_path / "eval"

    _write_row(
        train_cache,
        0,
        state=[0.0],
        actions=torch.tensor([[0.0, 0.0], [100.0, 100.0]]),
        action_mask=[1.0, 0.0],
        task="a",
        metadata={"task": "a"},
    )
    _write_row(
        train_cache,
        1,
        state=[10.0],
        actions=torch.tensor([[2.0, 4.0], [2.0, 4.0]]),
        action_mask=[1.0, 1.0],
        task="b",
        metadata={"task": "b"},
    )
    _write_row(
        eval_cache,
        0,
        state=[5.0],
        actions=torch.tensor([[1.0, 2.0], [999.0, 999.0]]),
        action_mask=[1.0, 0.0],
        task="heldout",
        metadata={"task": "heldout"},
    )

    report = diagnose_action_priors(
        train_cache_path=train_cache,
        eval_cache_path=eval_cache,
        output_dir=tmp_path / "out",
    )

    assert report["baselines"]["global_mean"]["mse"] == pytest.approx(0.0)
    assert report["baselines"]["global_mean"]["per_action_dim_mse"] == pytest.approx([0.0, 0.0])
    assert report["baselines"]["global_mean"]["num_valid_action_steps"] == 1
    assert report["baselines"]["global_mean"]["num_valid_action_elements"] == 2
