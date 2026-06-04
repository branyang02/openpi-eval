"""Tests for the libero_env example.

These tests run inside `examples/libero_env/.venv` (the dedicated Python 3.8
LIBERO venv), so they can `import main` and `import setup_libero_config`
directly without any subprocess shenanigans. `examples/libero_env/tests/conftest.py`
adds the example dir to sys.path and bootstraps `~/.libero/config.yaml` if
missing (so `import main` doesn't hit libero's interactive prompt).

Run them from this directory:

    uv sync                  # one-time, installs the libero venv
    MUJOCO_GL=osmesa uv run pytest   # CPU-only software renderer (works in CI)
    # or:
    MUJOCO_GL=egl uv run pytest      # faster, requires a GPU

The TestLiberoEnv tests construct a real MuJoCo OffScreenRenderEnv. They need
a working OpenGL backend (osmesa = software, egl = GPU, glx = X display) but
**do not strictly need a GPU**. The TestSetupLiberoConfig tests are pure
Python — they don't touch MuJoCo at all.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

import main
import setup_libero_config

# ---------------------------------------------------------- env smoke tests


class TestLiberoEnv:
    """End-to-end smoke tests against the real LIBERO env. Need a working
    MuJoCo OpenGL backend (set via `MUJOCO_GL=osmesa|egl|glx`) — osmesa
    works on a CPU-only CI runner with libosmesa6 installed."""

    def test_make_env_reset_and_step(self) -> None:
        """The env can be created, reset to a known initial state, and stepped once."""
        suite = main.get_task_suite("libero_spatial")
        task = suite.get_task(0)
        env = main.make_env(task, main.LIBERO_ENV_RESOLUTION, seed=7)
        try:
            env.reset()
            obs = env.set_init_state(suite.get_task_init_states(0)[0])

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
            args = main.Args(
                num_episodes=1,
                max_steps=1,  # render the initial frame before any env.step.
                num_steps_wait=0,
                replan_steps=1,
                fps=2,
                seed=seed,
            )
            main.eval_task(
                task_suite_name="libero_spatial",
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

        # Different seeds → different canonical initial state → visually distinct frame 0.
        frame_seed_0 = first_obs_for_seed(seed=0)
        frame_seed_5 = first_obs_for_seed(seed=5)
        # Large pixel delta confirms the env rendered a different initial scene.
        diff = np.abs(
            frame_seed_0.astype(np.int32) - frame_seed_5.astype(np.int32)
        ).mean()
        assert diff > 1.0, (
            f"expected seed=0 vs seed=5 to render different initial frames "
            f"(mean |Δpixel| > 1.0), got {diff:.3f}"
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

        args = main.Args(
            num_episodes=1,
            max_steps=2,
            num_steps_wait=1,
            replan_steps=1,
            fps=2,
            seed=7,
        )
        result = main.eval_task(
            task_suite_name="libero_spatial",
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


# ---------------------------------------------------------- setup script tests


class TestSetupLiberoConfig:
    """Pure-Python tests for setup_libero_config.py — no GPU, no libero needed."""

    def test_build_config_text_contains_required_keys(self) -> None:
        """The YAML body has all four LIBERO config keys with non-empty paths."""
        text = setup_libero_config.build_config_text()
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
        """All paths point inside this repo's third_party/libero tree."""
        text = setup_libero_config.build_config_text()
        repo_root = pathlib.Path(__file__).resolve().parents[3]
        third_party_libero = repo_root / "third_party" / "libero"
        for line in text.splitlines():
            if not line.strip():
                continue
            _, _, path = line.partition(":")
            path_str = path.strip()
            assert str(third_party_libero) in path_str, (
                f"line {line!r} doesn't point under third_party/libero "
                f"({third_party_libero})"
            )

    def test_setup_libero_config_writes_yaml_to_sandboxed_home(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end: setup_libero_config() creates ~/.libero/config.yaml under
        a fake HOME so the test never touches the real user home directory.
        """
        # Monkeypatch pathlib.Path.home() so the script writes under tmp_path.
        monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)

        config_path = setup_libero_config.setup_libero_config()
        assert config_path == tmp_path / ".libero" / "config.yaml"
        assert config_path.exists()

        text = config_path.read_text()
        assert "benchmark_root:" in text
        assert "bddl_files:" in text
        assert "assets:" in text
