from __future__ import annotations

import importlib.abc
import importlib.machinery
import runpy
import sys
from pathlib import Path
from types import ModuleType

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RECORDER_MODULE = "robolab.core.logging.recorder_manager"


def _robolab_root(repo_root: Path = _REPO_ROOT) -> Path:
    return repo_root / "third_party" / "robolab"


def _robolab_runner(repo_root: Path = _REPO_ROOT) -> Path:
    return _robolab_root(repo_root) / "policies" / "pi0_family" / "run.py"


def _slice_uint16_tensor(value: torch.Tensor, env_ids):
    if isinstance(env_ids, torch.Tensor):
        env_ids = env_ids.detach().cpu().numpy()
    else:
        env_ids = list(env_ids)
    return torch.from_numpy(value.detach().cpu().numpy()[env_ids]).to(
        device=value.device
    )


def _patch_recorder_module(module: ModuleType) -> None:
    if getattr(module, "_openpi_eval_uint16_patch", False):
        return

    def _slice_to_envs(value, env_ids):
        if isinstance(value, dict):
            return {key: _slice_to_envs(item, env_ids) for key, item in value.items()}
        if isinstance(value, torch.Tensor):
            if value.dtype == torch.uint16:
                return _slice_uint16_tensor(value, env_ids)
            return value[env_ids]
        return value

    module._slice_to_envs = _slice_to_envs
    module._openpi_eval_uint16_patch = True


class _RecorderPatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader) -> None:
        self._wrapped = wrapped

    def create_module(self, spec):  # noqa: ANN001
        if hasattr(self._wrapped, "create_module"):
            return self._wrapped.create_module(spec)
        return None

    def exec_module(self, module: ModuleType) -> None:
        self._wrapped.exec_module(module)
        _patch_recorder_module(module)


class _RecorderPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path, target=None):  # noqa: ANN001
        if fullname != _RECORDER_MODULE:
            return None

        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        if isinstance(spec.loader, _RecorderPatchLoader):
            return spec

        spec.loader = _RecorderPatchLoader(spec.loader)
        return spec


def _install_recorder_uint16_patch() -> None:
    module = sys.modules.get(_RECORDER_MODULE)
    if module is not None:
        _patch_recorder_module(module)
        return

    if not any(isinstance(finder, _RecorderPatchFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _RecorderPatchFinder())


def main() -> None:
    robolab_root = _robolab_root()
    runner = _robolab_runner()
    if not runner.exists():
        raise SystemExit(
            "RoboLab submodule is missing. Run: "
            "git submodule update --init --recursive third_party/robolab"
        )

    sys.path.insert(0, str(robolab_root))
    _install_recorder_uint16_patch()
    runpy.run_path(str(runner), run_name="__main__")


if __name__ == "__main__":
    main()
