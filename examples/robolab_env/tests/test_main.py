from pathlib import Path

import torch

import main


def test_robolab_runner_path_resolves() -> None:
    repo_root = Path(__file__).resolve().parents[3]

    runner = main._robolab_runner(repo_root)

    assert (
        runner
        == repo_root / "third_party" / "robolab" / "policies" / "pi0_family" / "run.py"
    )
    assert runner.exists()


def test_uint16_slice_uses_numpy_advanced_indexing() -> None:
    value = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint16)

    sliced = main._slice_uint16_tensor(value, [1])

    assert sliced.dtype == torch.uint16
    assert sliced.tolist() == [[3, 4]]
