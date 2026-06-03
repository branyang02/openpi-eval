import importlib
import sys
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


def test_recorder_patch_hooks_future_import(tmp_path, monkeypatch) -> None:
    logging_pkg = tmp_path / "robolab" / "core" / "logging"
    logging_pkg.mkdir(parents=True)
    for package in (
        tmp_path / "robolab",
        tmp_path / "robolab" / "core",
        logging_pkg,
    ):
        (package / "__init__.py").write_text("")
    (logging_pkg / "recorder_manager.py").write_text(
        "def _slice_to_envs(value, env_ids):\n" "    return 'unpatched'\n"
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    for module_name in list(sys.modules):
        if module_name == "robolab" or module_name.startswith("robolab."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setattr(
        sys,
        "meta_path",
        [
            finder
            for finder in sys.meta_path
            if not isinstance(finder, main._RecorderPatchFinder)
        ],
    )

    main._install_recorder_uint16_patch()
    recorder = importlib.import_module(main._RECORDER_MODULE)
    value = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint16)

    sliced = recorder._slice_to_envs(value, [1])

    assert recorder._openpi_eval_uint16_patch
    assert sliced.dtype == torch.uint16
    assert sliced.device == value.device
    assert sliced.tolist() == [[3, 4]]
