import importlib
import json
import sys
from pathlib import Path

import torch

import main


def test_robolab_runner_path_resolves() -> None:
    repo_root = Path(__file__).resolve().parents[3]

    root = main._robolab_root(repo_root)

    assert root == repo_root / "third_party" / "robolab"


def test_client_kwargs_forwards_public_server_args() -> None:
    args = main.Args(
        host="127.0.0.1",
        port=9001,
        policy="pi0_fast",
        open_loop_horizon=10,
    )

    assert main._client_kwargs(args) == {
        "remote_host": "127.0.0.1",
        "remote_port": 9001,
        "open_loop_horizon": 10,
        "policy_variant": "pi0_fast",
    }


def test_default_output_root() -> None:
    args = main.Args(policy="pi05", task=["BananaInBowlTask"])

    output_dir = Path(main._resolve_output_folder_name(args))

    assert output_dir.is_absolute()
    assert output_dir == Path(main.__file__).resolve().parent / "output" / "pi05"


def test_run_robolab_rejects_output_dir_with_other_policy_results(tmp_path) -> None:
    repo_root = tmp_path
    (repo_root / "third_party" / "robolab").mkdir(parents=True)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "episode_results.jsonl").write_text(
        json.dumps({"env_name": "BananaInBowlTask", "policy": "pi05", "success": True})
        + "\n"
    )

    try:
        main.run_robolab(
            main.Args(
                policy="pi0_fast", task=["BananaInBowlTask"], output_dir=str(output_dir)
            ),
            repo_root=repo_root,
        )
    except ValueError as exc:
        assert "contains RoboLab results from policy" in str(exc)
        assert "pi05" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_client_kwargs_uses_remote_uri_and_adaptive_sampling() -> None:
    args = main.Args(
        task=["BananaInBowlTask"],
        remote_uri="wss://example.test/policy",
        num_episodes_adaptive=25,
        ci_pp_width=0.2,
    )

    assert main._client_kwargs(args)["remote_uri"] == "wss://example.test/policy"
    assert main._total_episodes(args) == 25


def test_validate_args_requires_task_or_tag() -> None:
    args = main.Args(task=[], tag=[])

    try:
        main._validate_args(args)
    except ValueError as exc:
        assert "task or --tag" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_run_robolab_restores_sys_state(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path
    robolab_root = repo_root / "third_party" / "robolab"
    robolab_root.mkdir(parents=True)

    captured: dict[str, object] = {}

    class FakeApp:
        def close(self) -> None:
            captured["closed"] = True

    def fake_launch(args):  # noqa: ANN001
        captured["launch_args"] = args
        return FakeApp()

    def fake_run_task_set(args):  # noqa: ANN001
        captured["run_args"] = args
        captured["sys_path"] = list(sys.path)

    original_sys_path = list(sys.path)
    monkeypatch.setattr(main, "_launch_simulation_app", fake_launch)
    monkeypatch.setattr(main, "_configure_robolab_runtime", lambda args: None)
    monkeypatch.setattr(main, "_register_task_envs", lambda args: None)
    monkeypatch.setattr(main, "_run_task_set", fake_run_task_set)

    args = main.Args(task=["BananaInBowlTask"])
    main.run_robolab(args, repo_root=repo_root)

    assert captured["launch_args"] is args
    assert captured["run_args"] is args
    assert captured["closed"] is True
    assert str(robolab_root) in captured["sys_path"]
    assert sys.path == original_sys_path


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
        "def _slice_to_envs(value, env_ids):\n    return 'unpatched'\n"
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
