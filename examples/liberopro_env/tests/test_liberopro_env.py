"""Tests for the liberopro_env example.

These tests run inside `examples/liberopro_env/.venv` (the dedicated Python 3.8
LIBERO-Pro venv), so they can `import main` and `import setup_liberopro_config`
directly without any subprocess shenanigans. `examples/liberopro_env/tests/conftest.py`
adds the example dir to sys.path and bootstraps `~/.liberopro/config.yaml` if
missing (so `import main` doesn't hit libero's interactive prompt).

Run them from this directory:

    uv sync                  # one-time, installs the LIBERO-Pro venv
    MUJOCO_GL=osmesa uv run pytest   # CPU-only software renderer (works in CI)
    # or:
    MUJOCO_GL=egl uv run pytest      # faster, requires a GPU

The TestLiberoProEnv tests construct a real MuJoCo OffScreenRenderEnv. They need
a working OpenGL backend (osmesa = software, egl = GPU, glx = X display) but
**do not strictly need a GPU**. The TestSetupLiberoConfig tests are pure
Python; they don't touch MuJoCo at all.
"""

from __future__ import annotations

import os
import pathlib
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

import setup_liberopro_config

GENERATED_SUITE_FAMILIES = {
    "Position": [
        "libero_goal_swap",
        "libero_spatial_swap",
        "libero_10_swap",
        "libero_object_swap",
    ],
    "Object": [
        "libero_goal_object",
        "libero_spatial_object",
        "libero_10_object",
        "libero_object_object",
    ],
    "Semantic language": [
        "libero_goal_lan",
        "libero_spatial_lan",
        "libero_10_lan",
        "libero_object_lan",
    ],
    "Task": [
        "libero_goal_task",
        "libero_spatial_task",
        "libero_10_task",
        "libero_object_task",
    ],
}


def _fake_task(folder: str, bddl_file: str = "example.bddl") -> SimpleNamespace:
    return SimpleNamespace(
        problem_folder=folder,
        bddl_file=bddl_file,
        init_states_file=bddl_file.replace(".bddl", ".pruned_init"),
    )


def _generated_data_available(main, suite_name: str = "libero_goal_task") -> bool:
    suite = main.get_task_suite(suite_name)
    task = suite.get_task(0)
    try:
        main.resolve_bddl_file(task)
        main.resolve_init_states_file(task)
    except FileNotFoundError:
        return False
    return True


@contextmanager
def _stub_policy_server() -> Iterator[tuple[int, list[dict]]]:
    from openpi_client import msgpack_numpy
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.server import serve

    requests = []
    packer = msgpack_numpy.Packer()

    def handler(websocket) -> None:
        websocket.send(packer.pack({"server": "liberopro-test"}))
        while True:
            try:
                request = msgpack_numpy.unpackb(websocket.recv())
            except ConnectionClosed:
                return
            requests.append(request)
            websocket.send(
                packer.pack({"actions": np.zeros((10, 7), dtype=np.float32)})
            )

    with serve(handler, "127.0.0.1", 0) as server:
        port = server.socket.getsockname()[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield port, requests
        finally:
            server.shutdown()
            thread.join(timeout=5)


# ---------------------------------------------------------- pure integration tests


class TestLiberoProRegistryAndDataResolution:
    """Pure tests that make sure LIBERO-Pro resolves the generated suite files.

    The most important regression guard here is that `libero_goal_task` must
    not silently fall back to `libero_goal` if the generated task perturbation
    files are absent. Running base LIBERO while claiming LIBERO-Pro would be a
    very bad result, so the resolver should fail loudly.
    """

    def test_all_generated_suite_families_are_registered(self) -> None:
        import main

        registered = set(main.benchmark.get_benchmark_dict())
        expected = {
            suite
            for suite_names in GENERATED_SUITE_FAMILIES.values()
            for suite in suite_names
        }
        assert expected <= registered

    def test_default_max_steps_cover_all_generated_suite_families(self) -> None:
        import main

        for suite_names in GENERATED_SUITE_FAMILIES.values():
            for suite_name in suite_names:
                assert main.get_max_steps(suite_name, None) > 0

    def test_missing_generated_file_message_is_actionable(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import main

        monkeypatch.setattr(
            main,
            "get_libero_path",
            lambda key: str(tmp_path / key),
        )

        with pytest.raises(FileNotFoundError) as exc_info:
            main.resolve_bddl_file(_fake_task("libero_goal_task"))

        message = str(exc_info.value)
        assert "Missing LIBERO-Pro BDDL file" in message
        assert main.LIBEROPRO_DATASET_REPO in message
        assert "hf download" in message
        assert "bddl_files" in message
        assert "init_files" in message

    def test_resolver_does_not_fall_back_to_base_libero_files(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import main

        bddl_root = tmp_path / "bddl_files"
        base_file = bddl_root / "libero_goal" / "open_drawer.bddl"
        base_file.parent.mkdir(parents=True)
        base_file.write_text("base libero file")
        monkeypatch.setattr(
            main,
            "get_libero_path",
            lambda key: str(tmp_path / key),
        )

        task = _fake_task("libero_goal_task", "open_drawer.bddl")
        with pytest.raises(FileNotFoundError):
            main.resolve_bddl_file(task)

        generated_file = bddl_root / "libero_goal_task" / "open_drawer.bddl"
        generated_file.parent.mkdir(parents=True)
        generated_file.write_text("generated liberopro file")
        assert main.resolve_bddl_file(task) == generated_file

    def test_generated_dataset_files_are_complete_when_installed(self) -> None:
        import main

        if not _generated_data_available(main):
            pytest.skip(
                "Generated LIBERO-Pro BDDL/init-state dataset is not installed."
            )

        for family, suite_names in GENERATED_SUITE_FAMILIES.items():
            for suite_name in suite_names:
                suite = main.get_task_suite(suite_name)
                assert suite.n_tasks == 10, (family, suite_name)
                for task_id in range(suite.n_tasks):
                    task = suite.get_task(task_id)
                    bddl_file = main.resolve_bddl_file(task)
                    init_file = main.resolve_init_states_file(task)
                    assert bddl_file.parent.name == suite_name
                    assert init_file.parent.name == suite_name
                    assert bddl_file.name == task.bddl_file
                    assert init_file.name == task.init_states_file


# ---------------------------------------------------------- env smoke tests


@pytest.mark.skipif(
    os.environ.get("RUN_LIBEROPRO_ENV_TESTS") != "1",
    reason="Set RUN_LIBEROPRO_ENV_TESTS=1 to run MuJoCo LIBERO-Pro smoke tests.",
)
class TestLiberoProEnv:
    """End-to-end smoke tests against the real LIBERO-Pro env. Need a working
    MuJoCo OpenGL backend (set via `MUJOCO_GL=osmesa|egl|glx`); osmesa
    works on a CPU-only CI runner with libosmesa6 installed."""

    def _require_generated_data(self, main, suite_name: str = "libero_goal_task"):
        suite = main.get_task_suite(suite_name)
        task = suite.get_task(0)
        try:
            main.resolve_bddl_file(task)
            main.resolve_init_states_file(task)
        except FileNotFoundError as exc:
            pytest.skip(str(exc))
        return suite, task

    def test_make_env_reset_and_step(self) -> None:
        """The env can be created, reset to a known initial state, and stepped once."""
        import main

        suite, task = self._require_generated_data(main)
        env = main.make_env(task, main.LIBERO_ENV_RESOLUTION, seed=7)
        try:
            env.reset()
            obs = env.set_init_state(main.load_init_states(task)[0])

            # Initial obs shape contract.
            assert obs["agentview_image"].shape == (256, 256, 3)
            assert obs["agentview_image"].dtype == np.uint8
            assert obs["robot0_eye_in_hand_image"].shape == (256, 256, 3)
            assert obs["robot0_eye_in_hand_image"].dtype == np.uint8

            state = main.build_state(obs)
            assert state.shape == (8,)
            assert state.dtype == np.float32

            # One env step with the dummy no-op action.
            obs2, reward, done, info = env.step(main.LIBERO_DUMMY_ACTION)
            assert isinstance(reward, float)
            assert isinstance(done, bool)
            assert isinstance(info, dict)
            assert "agentview_image" in obs2
        finally:
            env.close()

    def test_seed_selects_different_initial_state(self, tmp_path: pathlib.Path) -> None:
        """``--seed`` offsets into ``initial_states`` so different seeds start
        rollouts from different canonical LIBERO states.

        Regression test for the bug where ``--seed`` only affected env physics
        RNG and ``initial_states[episode]`` always hit the first ``num_episodes``
        canonical slots regardless of seed.
        """

        class StubPolicy:
            def infer(self, element: dict) -> dict:
                return {"actions": np.zeros((10, 7), dtype=np.float32)}

        def first_obs_for_seed(seed: int) -> np.ndarray:
            import main

            self._require_generated_data(main)

            args = main.Args(
                num_episodes=1,
                max_steps=1,  # render the initial frame before any env.step.
                num_steps_wait=0,
                replan_steps=1,
                fps=2,
                seed=seed,
            )
            main.eval_task(
                task_suite_name="libero_goal_task",
                task_id=0,
                policy=StubPolicy(),
                args=args,
                output_dir=str(tmp_path / f"seed_{seed}"),
            )
            # Read the first frame of the rendered video.
            import imageio.v3 as iio

            video = sorted((tmp_path / f"seed_{seed}").rglob("*.mp4"))[0]
            frames = iio.imread(video)  # (T, H, W, 3) uint8
            return frames[0]

        # Different seeds select different canonical initial states.
        frame_seed_0 = first_obs_for_seed(seed=0)
        frame_seed_5 = first_obs_for_seed(seed=5)
        # Large pixel delta confirms the env rendered a different initial scene.
        diff = np.abs(
            frame_seed_0.astype(np.int32) - frame_seed_5.astype(np.int32)
        ).mean()
        assert diff > 1.0, (
            f"expected seed=0 vs seed=5 to render different initial frames "
            f"(mean absolute pixel delta > 1.0), got {diff:.3f}"
        )

        # Same seed must reproduce the same initial frame exactly.
        frame_seed_0_repeat = first_obs_for_seed(seed=0)
        np.testing.assert_array_equal(frame_seed_0, frame_seed_0_repeat)

    def test_eval_task_runs_with_stub_policy_and_writes_video(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The full eval loop runs to completion with a stub policy and writes a video."""

        class StubPolicy:
            """Returns zero actions; satisfies the WebsocketClientPolicy duck type."""

            def infer(self, element: dict) -> dict:
                # eval_task hands us a single-example obs dict (no batch dim).
                assert element["observation/image"].shape == (224, 224, 3)
                assert element["observation/wrist_image"].shape == (224, 224, 3)
                assert element["observation/state"].shape == (8,)
                assert isinstance(element["prompt"], str) and element["prompt"]
                return {"actions": np.zeros((10, 7), dtype=np.float32)}

        import main

        self._require_generated_data(main)

        args = main.Args(
            num_episodes=1,
            max_steps=2,
            num_steps_wait=1,
            replan_steps=1,
            fps=2,
            seed=7,
        )
        result = main.eval_task(
            task_suite_name="libero_goal_task",
            task_id=0,
            policy=StubPolicy(),
            args=args,
            output_dir=str(tmp_path),
        )

        # Result dict shape contract.
        assert set(result.keys()) == {
            "success_rate",
            "num_episodes",
            "task_id",
            "task_name",
            "task_description",
        }
        assert result["success_rate"] in {0.0, 1.0}
        assert result["num_episodes"] == 1.0

        # eval_task writes one .mp4 per episode under
        # <output_dir>/<task_id>-<task_name>/episode_NNN.mp4
        videos = sorted(tmp_path.rglob("*.mp4"))
        assert len(videos) == 1, f"expected 1 video, got {len(videos)}: {videos}"
        assert videos[0].stat().st_size > 0

    def test_main_runs_against_websocket_policy_server(
        self, tmp_path: pathlib.Path
    ) -> None:
        """End-to-end smoke through the same websocket path used by real evals."""
        import main

        self._require_generated_data(main)

        with _stub_policy_server() as (port, requests):
            main.main(
                main.Args(
                    host="127.0.0.1",
                    port=port,
                    task_suite_name="libero_goal_task",
                    task_id=0,
                    num_episodes=1,
                    max_steps=1,
                    num_steps_wait=0,
                    replan_steps=1,
                    fps=2,
                    seed=7,
                    output_dir=str(tmp_path),
                )
            )

        assert requests, "policy server should receive at least one inference request"
        request = requests[0]
        assert request["observation/image"].shape == (224, 224, 3)
        assert request["observation/wrist_image"].shape == (224, 224, 3)
        assert request["observation/state"].shape == (8,)
        assert request["observation/image"].dtype == np.uint8
        assert request["observation/wrist_image"].dtype == np.uint8
        assert request["observation/state"].dtype == np.float32
        assert isinstance(request["prompt"], str) and request["prompt"]

        videos = sorted(tmp_path.rglob("*.mp4"))
        assert len(videos) == 1, f"expected 1 video, got {len(videos)}: {videos}"
        assert videos[0].stat().st_size > 0


# ---------------------------------------------------------- setup script tests


class TestSetupLiberoProConfig:
    """Pure-Python tests for setup_liberopro_config.py; no GPU, no libero needed."""

    def test_build_config_text_contains_required_keys(self) -> None:
        """The YAML body has all four LIBERO config keys with non-empty paths."""
        text = setup_liberopro_config.build_config_text()
        for key in (
            "benchmark_root:",
            "bddl_files:",
            "init_states:",
            "datasets:",
            "assets:",
        ):
            assert key in text, f"missing {key!r} in build_config_text() output"
        # Each line should be `key: /some/absolute/path`.
        non_empty_lines = [line for line in text.splitlines() if line.strip()]
        assert len(non_empty_lines) == 5
        for line in non_empty_lines:
            key, _, path = line.partition(":")
            assert key.strip(), f"empty key in line {line!r}"
            assert path.strip().startswith("/"), f"non-absolute path in line {line!r}"

    def test_build_config_text_paths_resolve_under_repo(self) -> None:
        """All paths point inside this repo's third_party/liberopro tree."""
        text = setup_liberopro_config.build_config_text()
        repo_root = pathlib.Path(__file__).resolve().parents[3]
        third_party_liberopro = repo_root / "third_party" / "liberopro"
        for line in text.splitlines():
            if not line.strip():
                continue
            _, _, path = line.partition(":")
            path_str = path.strip()
            assert str(third_party_liberopro) in path_str, (
                f"line {line!r} doesn't point under third_party/liberopro "
                f"({third_party_liberopro})"
            )

    def test_setup_liberopro_config_writes_yaml_to_sandboxed_home(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end: setup_liberopro_config() creates ~/.liberopro/config.yaml under
        a fake HOME so the test never touches the real user home directory.
        """
        # Monkeypatch pathlib.Path.home() so the script writes under tmp_path.
        monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)

        config_path = setup_liberopro_config.setup_liberopro_config()
        assert config_path == tmp_path / ".liberopro" / "config.yaml"
        assert config_path.exists()

        text = config_path.read_text()
        assert "benchmark_root:" in text
        assert "bddl_files:" in text
        assert "assets:" in text
