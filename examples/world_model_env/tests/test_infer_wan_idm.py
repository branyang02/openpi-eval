from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from infer_wan_idm import Args, main
from world_model.config import DatasetConfig, ModelConfig


def test_infer_wan_idm_defaults_to_raw_wan_frame_delta_one(tmp_path) -> None:
    args = Args(
        idm_checkpoint=str(tmp_path / "idm.pt"),
        wan_repo_dir=str(tmp_path / "Wan2.2"),
        wan_checkpoint_dir=str(tmp_path / "Wan2.2-TI2V-5B"),
    )

    assert args.frame_delta == 1


def test_infer_wan_idm_rejects_raw_wan_frame_delta_above_one_before_loading_checkpoint(tmp_path) -> None:
    with pytest.raises(ValueError, match="Raw Wan2.2 only supports frame_delta=1"):
        main(
            Args(
                idm_checkpoint=str(tmp_path / "missing_idm.pt"),
                wan_repo_dir=str(tmp_path / "missing_wan_repo"),
                wan_checkpoint_dir=str(tmp_path / "missing_wan_checkpoint"),
                frame_delta=4,
            )
        )


def test_infer_wan_idm_forwards_history_kwargs_from_dataset(tmp_path, monkeypatch) -> None:
    class HistoryRequiredIdm(torch.nn.Module):
        uses_flow_matching = False

        def __init__(self):
            super().__init__()
            self.seen_history = None

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
            return torch.zeros((current_images.shape[0], 2, 1), device=current_images.device)

    class TinyDataset:
        def __init__(self):
            self.item = {
                "current_images": torch.zeros(1, 3, 8, 8),
                "future_images": torch.zeros(1, 1, 3, 8, 8),
                "state": torch.zeros(4),
                "task_id": torch.tensor(0),
                "prev_state_history": torch.ones(1, 4),
                "prev_action_history": torch.ones(1, 1) * 3.0,
                "history_mask": torch.ones(1),
            }

        def __getitem__(self, index):
            if index != 0:
                raise IndexError(index)
            return self.item

        def task_text(self, index):
            del index
            return "reach"

    class FakeWan:
        def __init__(self, config):
            self.config = config

        def generate_future_stack(self, current_images, *, task_text, output_dir, image_size, num_future_frames, seed):
            del current_images, task_text, output_dir, image_size, seed
            return SimpleNamespace(
                prompt="reach",
                seed=7,
                input_image_path=tmp_path / "input.png",
                video_path=tmp_path / "video.mp4",
                future_images=torch.zeros(num_future_frames, 1, 3, 8, 8),
                selected_frame_indices=(1,),
                total_video_frames=2,
            )

    dataset = TinyDataset()
    idm = HistoryRequiredIdm()
    model_config = ModelConfig(
        num_views=1,
        image_size=8,
        state_dim=4,
        action_dim=1,
        action_horizon=2,
        num_future_frames=1,
        idm_arch="flow_transformer",
        idm_history_length=1,
    )
    captured_configs: list[DatasetConfig] = []
    monkeypatch.setattr(
        "infer_wan_idm.create_dataset",
        lambda config: captured_configs.append(config) or dataset,
    )
    monkeypatch.setattr("infer_wan_idm.load_idm_checkpoint", lambda path, device: (idm.to(device), model_config))
    monkeypatch.setattr("infer_wan_idm.enforce_idm_frame_delta_contract", lambda path, frame_delta: None)
    monkeypatch.setattr("infer_wan_idm.Wan22FutureGenerator", FakeWan)

    main(
        Args(
            idm_checkpoint="fake.pt",
            wan_repo_dir=str(tmp_path / "Wan2.2"),
            wan_checkpoint_dir=str(tmp_path / "Wan2.2-TI2V-5B"),
            output_dir=str(tmp_path / "out"),
            device="cpu",
        )
    )

    output = json.loads((tmp_path / "out" / "wan_idm_action.json").read_text())

    assert captured_configs[0].idm_history_length == 1
    assert output["action_chunk"] == [[0.0], [0.0]]
    assert idm.seen_history is not None
    assert torch.allclose(idm.seen_history[0], dataset.item["prev_state_history"].unsqueeze(0))
    assert torch.allclose(idm.seen_history[1], dataset.item["prev_action_history"].unsqueeze(0))
    assert torch.allclose(idm.seen_history[2], dataset.item["history_mask"].unsqueeze(0))
