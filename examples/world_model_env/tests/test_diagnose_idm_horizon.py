from __future__ import annotations

import json

import pytest
import torch

from diagnose_idm_horizon import Args, horizon_action_diagnostics
from diagnose_idm_horizon import main as horizon_main
from world_model.config import ModelConfig


class _ZeroIdm(torch.nn.Module):
    def forward(self, current_images, future_images, state, task_id, *, sample_noise=None):
        del future_images, state, task_id, sample_noise
        return torch.zeros((current_images.shape[0], 3, 2), device=current_images.device)


class _HistoryRequiredIdm(torch.nn.Module):
    uses_flow_matching = False

    def __init__(self):
        super().__init__()
        self.seen_history: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def forward(
        self,
        current_images,
        future_images,
        state,
        task_id,
        *,
        sample_noise=None,
        prev_state_history=None,
        prev_action_history=None,
        history_mask=None,
    ):
        del future_images, state, task_id, sample_noise
        if prev_state_history is None or prev_action_history is None or history_mask is None:
            raise AssertionError("history kwargs were not forwarded")
        self.seen_history = (
            prev_state_history.detach().cpu(),
            prev_action_history.detach().cpu(),
            history_mask.detach().cpu(),
        )
        return torch.zeros((current_images.shape[0], 3, 2), device=current_images.device)


def _dataset_and_config() -> tuple[list[dict[str, torch.Tensor]], ModelConfig]:
    dataset = [
        {
            "current_images": torch.zeros(1, 3, 8, 8),
            "future_images": torch.zeros(1, 1, 3, 8, 8),
            "state": torch.zeros(4),
            "task_id": torch.tensor(0),
            "action_chunk": torch.tensor([[1.0, 3.0], [2.0, 4.0], [5.0, 7.0]]),
            "action_mask": torch.tensor([1.0, 1.0, 0.0]),
        },
        {
            "current_images": torch.zeros(1, 3, 8, 8),
            "future_images": torch.zeros(1, 1, 3, 8, 8),
            "state": torch.zeros(4),
            "task_id": torch.tensor(0),
            "action_chunk": torch.tensor([[10.0, 20.0], [6.0, 8.0], [9.0, 11.0]]),
            "action_mask": torch.tensor([1.0, 0.0, 0.0]),
        },
    ]
    model_config = ModelConfig(
        num_views=1,
        image_size=8,
        state_dim=4,
        action_dim=2,
        action_horizon=3,
        num_future_frames=1,
    )
    return dataset, model_config


def test_horizon_action_diagnostics_weights_valid_action_elements() -> None:
    predicted = torch.zeros(2, 3, 2)
    target = torch.tensor(
        [
            [[1.0, 3.0], [2.0, 4.0], [5.0, 7.0]],
            [[10.0, 20.0], [6.0, 8.0], [9.0, 11.0]],
        ]
    )
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    diag = horizon_action_diagnostics(predicted, target, mask)

    assert diag["num_valid_actions"] == 3
    assert diag["idm_mse"] == pytest.approx((1.0 + 9.0 + 4.0 + 16.0 + 100.0 + 400.0) / 6.0)
    assert diag["idm_mae"] == pytest.approx((1.0 + 3.0 + 2.0 + 4.0 + 10.0 + 20.0) / 6.0)
    assert diag["per_action_step_valid_count"] == [2, 1, 0]
    assert diag["per_action_step_mse"][:2] == pytest.approx([127.5, 10.0])
    assert diag["per_action_step_mse"][2] is None
    assert diag["per_action_step_mae"][:2] == pytest.approx([8.5, 3.0])
    assert diag["per_action_step_mae"][2] is None
    assert diag["per_action_step_dim_mse"][0] == pytest.approx([50.5, 204.5])
    assert diag["per_action_step_dim_mse"][1] == pytest.approx([4.0, 16.0])
    assert diag["per_action_step_dim_mse"][2] == [None, None]


def test_horizon_action_diagnostics_reports_first_later_and_zero_count_fields() -> None:
    predicted = torch.zeros(2, 3, 2)
    target = torch.tensor(
        [
            [[1.0, 3.0], [2.0, 4.0], [5.0, 7.0]],
            [[10.0, 20.0], [6.0, 8.0], [9.0, 11.0]],
        ]
    )
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    diag = horizon_action_diagnostics(predicted, target, mask)

    assert diag["first_action_mse"] == pytest.approx(127.5)
    assert diag["first_action_mae"] == pytest.approx(8.5)
    assert diag["later_action_mse"] == pytest.approx(10.0)
    assert diag["later_action_mae"] == pytest.approx(3.0)
    assert diag["first_vs_later_mse_ratio"] == pytest.approx(12.75)
    assert diag["last_action_mse"] is None
    assert diag["last_action_mae"] is None
    assert diag["per_action_step_dim_mae"][2] == [None, None]


def test_horizon_action_diagnostics_all_invalid_outputs_null_metrics() -> None:
    diag = horizon_action_diagnostics(
        predicted=torch.zeros(1, 2, 1),
        target=torch.ones(1, 2, 1),
        mask=torch.zeros(1, 2),
    )

    assert diag["num_valid_actions"] == 0
    assert diag["idm_mse"] is None
    assert diag["idm_mae"] is None
    assert diag["per_action_step_mse"] == [None, None]
    assert diag["later_action_mse"] is None
    assert diag["first_vs_later_mse_ratio"] is None


def test_diagnose_idm_horizon_writes_json(tmp_path, monkeypatch) -> None:
    dataset, model_config = _dataset_and_config()
    monkeypatch.setattr("diagnose_idm_horizon.create_dataset_with_optional_cache", lambda config, cache: dataset)
    monkeypatch.setattr(
        "diagnose_idm_horizon.load_idm_checkpoint",
        lambda path, device: (_ZeroIdm().to(device), model_config),
    )
    monkeypatch.setattr("diagnose_idm_horizon.enforce_idm_frame_delta_contract", lambda path, frame_delta: None)

    horizon_main(
        Args(
            checkpoint="fake.pt",
            output_dir=str(tmp_path),
            image_size=8,
            action_horizon=3,
            batch_size=1,
            device="cpu",
        )
    )

    metrics_path = tmp_path / "idm_horizon_diagnostics.json"
    metrics = json.loads(metrics_path.read_text())

    assert metrics_path.exists()
    assert metrics["checkpoint"] == "fake.pt"
    assert metrics["cached_future_dir"] is None
    assert metrics["dataset_config"]["action_horizon"] == 3
    assert metrics["num_samples"] == 2
    assert metrics["num_valid_actions"] == 3
    assert metrics["idm_mse"] == pytest.approx((1.0 + 9.0 + 4.0 + 16.0 + 100.0 + 400.0) / 6.0)
    assert metrics["first_action_mse"] == pytest.approx(127.5)
    assert metrics["later_action_mse"] == pytest.approx(10.0)
    assert metrics["last_action_mse"] is None


def test_diagnose_idm_horizon_forwards_history_kwargs_from_dataset(tmp_path, monkeypatch) -> None:
    item = {
        "current_images": torch.zeros(1, 3, 8, 8),
        "future_images": torch.zeros(1, 1, 3, 8, 8),
        "state": torch.zeros(4),
        "task_id": torch.tensor(0),
        "action_chunk": torch.zeros(3, 2),
        "action_mask": torch.ones(3),
        "prev_state_history": torch.ones(2, 4),
        "prev_action_history": torch.ones(2, 2) * 2.0,
        "history_mask": torch.tensor([1.0, 0.0]),
    }
    idm = _HistoryRequiredIdm()
    model_config = ModelConfig(
        num_views=1,
        image_size=8,
        state_dim=4,
        action_dim=2,
        action_horizon=3,
        num_future_frames=1,
        idm_arch="flow_transformer",
        idm_history_length=2,
    )
    captured_configs = []
    monkeypatch.setattr(
        "diagnose_idm_horizon.create_dataset_with_optional_cache",
        lambda config, cache: captured_configs.append(config) or [item],
    )
    monkeypatch.setattr(
        "diagnose_idm_horizon.load_idm_checkpoint",
        lambda path, device: (idm.to(device), model_config),
    )
    monkeypatch.setattr("diagnose_idm_horizon.enforce_idm_frame_delta_contract", lambda path, frame_delta: None)

    horizon_main(
        Args(
            checkpoint="fake.pt",
            output_dir=str(tmp_path),
            image_size=8,
            action_horizon=3,
            batch_size=1,
            device="cpu",
        )
    )

    assert captured_configs[0].idm_history_length == 2
    assert idm.seen_history is not None
    assert torch.allclose(idm.seen_history[0], item["prev_state_history"].unsqueeze(0))
    assert torch.allclose(idm.seen_history[1], item["prev_action_history"].unsqueeze(0))
    assert torch.allclose(idm.seen_history[2], item["history_mask"].unsqueeze(0))
