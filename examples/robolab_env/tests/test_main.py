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


def test_build_runner_argv_forwards_public_args(tmp_path) -> None:
    runner = tmp_path / "run.py"
    args = main.Args(
        host="127.0.0.1",
        port=9001,
        policy="pi0_fast",
        task=["BananaInBowlTask", "OneBottleInSquarePailTask"],
        task_dirs=["benchmark", "custom_tasks"],
        num_envs=4,
        num_runs=2,
        open_loop_horizon=10,
        output_dir="relative-output",
        video_mode="none",
        enable_subtask=True,
        device="cuda:1",
    )

    argv = main._build_runner_argv(args, runner)

    expected_pairs = {
        "--policy": "pi0_fast",
        "--remote-host": "127.0.0.1",
        "--remote-port": "9001",
        "--num-envs": "4",
        "--num-runs": "2",
        "--open-loop-horizon": "10",
        "--video-mode": "none",
        "--device": "cuda:1",
    }
    for flag, value in expected_pairs.items():
        assert flag in argv
        assert argv[argv.index(flag) + 1] == value

    assert argv[0] == str(runner)
    assert argv.count("--task") == 1
    assert argv[argv.index("--task") + 1 : argv.index("--task") + 3] == [
        "BananaInBowlTask",
        "OneBottleInSquarePailTask",
    ]
    assert argv.count("--task-dirs") == 1
    assert argv[argv.index("--task-dirs") + 1 : argv.index("--task-dirs") + 3] == [
        "benchmark",
        "custom_tasks",
    ]
    assert "--output-folder-name" in argv
    assert Path(argv[argv.index("--output-folder-name") + 1]).is_absolute()
    assert "--headless" in argv
    assert "--enable-subtask" in argv


def test_build_runner_argv_uses_remote_uri_and_adaptive_sampling(tmp_path) -> None:
    runner = tmp_path / "run.py"
    args = main.Args(
        task=["BananaInBowlTask"],
        remote_uri="wss://example.test/policy",
        num_episodes_adaptive=25,
        ci_pp_width=0.2,
    )

    argv = main._build_runner_argv(args, runner)

    assert "--remote-uri" in argv
    assert argv[argv.index("--remote-uri") + 1] == "wss://example.test/policy"
    assert "--num-episodes-adaptive" in argv
    assert argv[argv.index("--num-episodes-adaptive") + 1] == "25"
    assert argv[argv.index("--ci-pp-width") + 1] == "0.2"


def test_build_runner_argv_requires_task_or_tag(tmp_path) -> None:
    runner = tmp_path / "run.py"
    args = main.Args(task=[], tag=[])

    try:
        main._build_runner_argv(args, runner)
    except ValueError as exc:
        assert "task or --tag" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_run_robolab_restores_sys_state(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path
    runner = repo_root / "third_party" / "robolab" / "policies" / "pi0_family" / "run.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("print('fake runner')\n")

    captured = {}

    def fake_run_path(path, run_name):  # noqa: ANN001
        captured["path"] = path
        captured["run_name"] = run_name
        captured["argv"] = list(sys.argv)
        captured["sys_path"] = list(sys.path)

    original_argv = list(sys.argv)
    original_sys_path = list(sys.path)
    monkeypatch.setattr(main.runpy, "run_path", fake_run_path)

    main.run_robolab(main.Args(task=["BananaInBowlTask"]), repo_root=repo_root)

    assert captured["path"] == str(runner)
    assert captured["run_name"] == "__main__"
    assert captured["argv"][0] == str(runner)
    assert str(repo_root / "third_party" / "robolab") in captured["sys_path"]
    assert sys.argv == original_argv
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
