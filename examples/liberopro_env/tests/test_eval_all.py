"""Tests for eval_all.py's parallel LIBERO-Pro subprocess orchestration.

Covers the pure-Python units of eval_all.py without spinning up a policy
server or a real LIBERO-Pro env:

- ``SUCCESS_RATE_RE``: regex parsing of main.py's final ``success_rate=...`` log line
- ``_build_command``: argv list construction for various ``Args`` configurations
- ``_run_one_task``: subprocess launch + log capture + success_rate parsing
  via a minimal fake ``main.py`` stub written to a tmp dir

None of these tests need a running policy server, MuJoCo, or the libero
benchmark. They can run on a CPU-only machine in well under 10 seconds.
Like ``test_liberopro_env.py`` they run inside ``examples/liberopro_env/.venv``.

    cd examples/liberopro_env
    uv run pytest tests/test_eval_all.py -v
"""

from __future__ import annotations

import math
import pathlib
import textwrap

import eval_all

# --------------------------------------------------------- SUCCESS_RATE_RE


class TestSuccessRateRegex:
    """``SUCCESS_RATE_RE`` must match the log line main.py emits at the end of
    ``eval_task``: ``[<suite>/<task_name>/task_NN] success_rate=X.XX (k/N)``.
    ``_run_one_task`` uses ``findall(...)[-1]`` so the *last* match is the one
    that becomes the per-task result.
    """

    def test_parses_full_success(self) -> None:
        matches = eval_all.SUCCESS_RATE_RE.findall(
            "[libero_goal_task/open_drawer/task_00] success_rate=1.00 (1/1)"
        )
        assert matches == ["1.00"]

    def test_parses_partial_success(self) -> None:
        matches = eval_all.SUCCESS_RATE_RE.findall(
            "[libero_10/KITCHEN_SCENE4/task_03] success_rate=0.67 (2/3)"
        )
        assert matches == ["0.67"]

    def test_parses_zero(self) -> None:
        assert eval_all.SUCCESS_RATE_RE.findall("success_rate=0.00 (0/1)") == ["0.00"]

    def test_no_match_on_unrelated_line(self) -> None:
        assert eval_all.SUCCESS_RATE_RE.findall("some other output, nothing here") == []

    def test_last_match_wins(self) -> None:
        """Multiple success_rate lines in the log: the last one is what
        ``_run_one_task`` uses as the per-task result."""
        log = textwrap.dedent(
            """\
            INFO:__main__:[task_00] success_rate=0.00 (0/1)
            INFO:__main__:some intermediate noise
            INFO:__main__:[task_00] success_rate=1.00 (1/1)
            """
        )
        matches = eval_all.SUCCESS_RATE_RE.findall(log)
        assert matches == ["0.00", "1.00"]
        assert float(matches[-1]) == 1.0


# --------------------------------------------------------- _build_command


def _default_args(**overrides) -> eval_all.Args:
    """Instantiate ``Args`` with only the given overrides. All other fields
    fall back to their dataclass defaults, so adding new fields to ``Args``
    does not break the tests."""
    return eval_all.Args(**overrides)


class TestBuildCommand:
    """``_build_command`` is the contract between eval_all and main.py.
    These tests lock in the argv layout so a refactor can't silently break
    subprocess invocation."""

    # Every test in this class passes a concrete output_dir. After the output_dir
    # rework, ``_build_command`` requires a non-None output_dir: main() always
    # resolves its own ``args.output_dir`` (or the default) to an absolute path
    # before forwarding, so the None-output_dir case is unreachable at runtime
    # and no longer a valid input to _build_command.
    _TEST_OUTPUT_DIR = "/tmp/eval_all_test_output"

    def test_includes_main_py_and_python_executable(self) -> None:
        args = _default_args()
        cmd = eval_all._build_command(args, task_id=0, output_dir=self._TEST_OUTPUT_DIR)
        assert cmd[0].endswith("python") or cmd[0].endswith(
            "python3"
        ), f"first element should be the Python executable, got {cmd[0]!r}"
        assert cmd[1] == "main.py"

    def test_required_main_flags_are_forwarded(self) -> None:
        args = _default_args(
            task_suite_name="libero_goal_task",
            num_episodes=3,
            num_steps_wait=5,
            replan_steps=7,
            resize_size=128,
            fps=15,
            seed=42,
            host="1.2.3.4",
            port=9999,
        )
        cmd = eval_all._build_command(args, task_id=4, output_dir=self._TEST_OUTPUT_DIR)
        # Every flag-value pair should appear somewhere in the argv list.
        expected_pairs = {
            "--host": "1.2.3.4",
            "--port": "9999",
            "--task_suite_name": "libero_goal_task",
            "--task_id": "4",
            "--num_episodes": "3",
            "--num_steps_wait": "5",
            "--replan_steps": "7",
            "--resize_size": "128",
            "--fps": "15",
            "--seed": "42",
        }
        for flag, value in expected_pairs.items():
            assert flag in cmd, f"missing {flag} in {cmd}"
            assert (
                cmd[cmd.index(flag) + 1] == value
            ), f"{flag} followed by {cmd[cmd.index(flag) + 1]!r}, expected {value!r}"

    def test_task_id_is_the_only_thing_that_changes_across_calls(self) -> None:
        args = _default_args()
        cmd_a = eval_all._build_command(
            args, task_id=0, output_dir=self._TEST_OUTPUT_DIR
        )
        cmd_b = eval_all._build_command(
            args, task_id=7, output_dir=self._TEST_OUTPUT_DIR
        )
        # Both should differ only at the --task_id value position.
        assert len(cmd_a) == len(cmd_b)
        diffs = [i for i, (a, b) in enumerate(zip(cmd_a, cmd_b)) if a != b]
        assert len(diffs) == 1, f"expected exactly one index to differ, got {diffs}"
        diff_idx = diffs[0]
        assert cmd_a[diff_idx - 1] == "--task_id"
        assert cmd_a[diff_idx] == "0"
        assert cmd_b[diff_idx] == "7"

    def test_max_steps_forwarded_when_set(self) -> None:
        cmd = eval_all._build_command(
            _default_args(max_steps=150), 0, self._TEST_OUTPUT_DIR
        )
        assert "--max_steps" in cmd
        assert cmd[cmd.index("--max_steps") + 1] == "150"

    def test_max_steps_absent_when_none(self) -> None:
        cmd = eval_all._build_command(
            _default_args(max_steps=None), 0, self._TEST_OUTPUT_DIR
        )
        assert "--max_steps" not in cmd

    def test_output_dir_always_forwarded(self) -> None:
        """After the output_dir rework, ``_build_command`` unconditionally
        forwards ``--output_dir`` to main.py. This is the invariant that
        prevents main.py from falling back to its own
        ``output/{suite}-task{id:02d}/`` default and scattering videos into a
        sibling directory."""
        cmd = eval_all._build_command(
            _default_args(), task_id=0, output_dir="/tmp/eval_all_test"
        )
        assert "--output_dir" in cmd
        assert cmd[cmd.index("--output_dir") + 1] == "/tmp/eval_all_test"
        # The flag must appear exactly once.
        assert cmd.count("--output_dir") == 1

    def test_output_dir_forwarded_verbatim_without_suite_name_nesting(self) -> None:
        """A user passing ``--output_dir /foo`` to eval_all wants exactly
        ``/foo`` forwarded to main.py, not ``/foo/{suite}-task{id:02d}`` or
        ``/foo/{suite}``. This locks in that main.py's own per-task video dir
        lands directly under the provided path."""
        cmd = eval_all._build_command(
            _default_args(task_suite_name="libero_10_object"),
            task_id=0,
            output_dir="/foo/bar",
        )
        # The value immediately after --output_dir should be exactly what we passed.
        assert cmd[cmd.index("--output_dir") + 1] == "/foo/bar"

    def test_render_cameras_default_uses_space_separated_form(self) -> None:
        """Default is ``['agentview', 'eye_in_hand']`` and tyro's ``List[str]``
        form is ``--render_cameras cam1 cam2`` (single flag, positional values).
        The repeated-flag form silently drops all but the last value."""
        cmd = eval_all._build_command(_default_args(), 0, self._TEST_OUTPUT_DIR)
        assert "--render_cameras" in cmd
        flag_idx = cmd.index("--render_cameras")
        assert cmd[flag_idx + 1] == "agentview"
        assert cmd[flag_idx + 2] == "eye_in_hand"
        # The flag must appear exactly once: no repeated --render_cameras.
        assert cmd.count("--render_cameras") == 1

    def test_render_cameras_custom_single_value(self) -> None:
        cmd = eval_all._build_command(
            _default_args(render_cameras=["agentview"]), 0, self._TEST_OUTPUT_DIR
        )
        flag_idx = cmd.index("--render_cameras")
        assert cmd[flag_idx + 1] == "agentview"
        # The next element (if any) should be the start of another flag,
        # not a stray camera name.
        if flag_idx + 2 < len(cmd):
            assert cmd[flag_idx + 2].startswith("--")

    def test_render_cameras_empty_list_omits_flag(self) -> None:
        """Passing an empty list should omit ``--render_cameras`` entirely so
        main.py falls back to its own default. This guards against the
        degenerate case of ``eval_all.Args(render_cameras=[])``."""
        cmd = eval_all._build_command(
            _default_args(render_cameras=[]), 0, self._TEST_OUTPUT_DIR
        )
        assert "--render_cameras" not in cmd

    def test_output_dir_render_cameras_max_steps_compose_without_interleaving(
        self,
    ) -> None:
        """Optional flags together must not interleave each other's
        values. This is the scenario that would break if render_cameras used
        ``cmd.extend([...])`` and a later optional flag landed mid-list."""
        args = _default_args(
            render_cameras=["agentview", "eye_in_hand"],
            max_steps=200,
        )
        cmd = eval_all._build_command(args, 0, output_dir="/tmp/x")
        # All three must be present (render_cameras, max_steps, output_dir).
        assert "--render_cameras" in cmd
        assert "--max_steps" in cmd
        assert "--output_dir" in cmd
        # --render_cameras values must be immediately followed by its camera
        # names, not another optional flag.
        rc = cmd.index("--render_cameras")
        assert cmd[rc + 1] == "agentview"
        assert cmd[rc + 2] == "eye_in_hand"


# --------------------------------------------------------- _run_one_task

# Minimal fake main.py stubs. Each writes the canned success_rate line via
# logging.info (which goes to stderr and is merged into stdout by
# _run_one_task's subprocess.run(stderr=STDOUT) call), then exits with the
# configured code. No libero imports, no MuJoCo; just an argv-parsing shim
# so main.py's flags (which eval_all forwards) don't crash tyro.

_FAKE_MAIN_SUCCESS = """
    import sys, logging
    # Swallow all CLI args; we don't need to validate them here.
    logging.basicConfig(level=logging.INFO)
    logging.info('[fake_suite/fake_task/task_00] success_rate=1.00 (1/1)')
    sys.exit(0)
"""

_FAKE_MAIN_FAILURE = """
    import sys, logging
    logging.basicConfig(level=logging.INFO)
    logging.info('[fake_suite/fake_task/task_03] success_rate=0.00 (0/1)')
    sys.exit(0)
"""

_FAKE_MAIN_SILENT = """
    import sys
    # Exit cleanly but never emit a success_rate line.
    sys.exit(0)
"""

_FAKE_MAIN_CRASH = """
    import sys
    sys.stderr.write('Traceback (most recent call last):\\n')
    sys.stderr.write('RuntimeError: boom\\n')
    sys.exit(1)
"""


def _make_fake_env(
    tmp_path: pathlib.Path, script_body: str
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """Write a minimal fake main.py in a tmp subdirectory and return
    ``(cwd, log_dir, output_dir)``. The fake main.py ignores its CLI args
    (including ``--output_dir``) and just exits with a canned success_rate
    line, so the output_dir is created but nothing lands under it; it only
    exists to satisfy ``_build_command``'s unconditional ``--output_dir``
    forwarding after the output_dir rework."""
    fake_dir = tmp_path / "fake_env"
    fake_dir.mkdir()
    (fake_dir / "main.py").write_text(textwrap.dedent(script_body).lstrip())
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    output_dir = tmp_path / "run_output"
    output_dir.mkdir()
    return fake_dir, log_dir, output_dir


class TestRunOneTask:
    """Integration tests for ``_run_one_task``: writes a fake main.py to a tmp
    dir, runs eval_all._run_one_task with that dir as cwd, and verifies
    the parsed result dict plus the on-disk log file.

    The fake main.py is Python-only: no libero, no MuJoCo, no policy server.
    """

    def test_parses_success_rate_from_subprocess_stdout(
        self, tmp_path: pathlib.Path
    ) -> None:
        fake_dir, log_dir, output_dir = _make_fake_env(tmp_path, _FAKE_MAIN_SUCCESS)
        result = eval_all._run_one_task(
            _default_args(task_suite_name="fake_suite"),
            task_id=0,
            log_dir=str(log_dir),
            cwd=str(fake_dir),
            output_dir=str(output_dir),
        )
        assert result["task_id"] == 0
        assert result["success_rate"] == 1.0
        assert result["returncode"] == 0
        # The log file exists and contains the success line.
        log_path = pathlib.Path(result["log_path"])
        assert log_path.exists()
        assert "success_rate=1.00" in log_path.read_text()

    def test_parses_failure_success_rate(self, tmp_path: pathlib.Path) -> None:
        fake_dir, log_dir, output_dir = _make_fake_env(tmp_path, _FAKE_MAIN_FAILURE)
        result = eval_all._run_one_task(
            _default_args(task_suite_name="fake_suite"),
            task_id=3,
            log_dir=str(log_dir),
            cwd=str(fake_dir),
            output_dir=str(output_dir),
        )
        assert result["task_id"] == 3
        assert result["success_rate"] == 0.0
        assert result["returncode"] == 0

    def test_silent_exit_returns_nan(self, tmp_path: pathlib.Path) -> None:
        """Subprocess exits cleanly but emits no success_rate line: nan."""
        fake_dir, log_dir, output_dir = _make_fake_env(tmp_path, _FAKE_MAIN_SILENT)
        result = eval_all._run_one_task(
            _default_args(task_suite_name="fake_suite"),
            task_id=0,
            log_dir=str(log_dir),
            cwd=str(fake_dir),
            output_dir=str(output_dir),
        )
        assert math.isnan(result["success_rate"])
        assert result["returncode"] == 0

    def test_crash_returns_nan_and_nonzero_returncode(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Subprocess crashes with a traceback: nan success_rate and non-zero rc."""
        fake_dir, log_dir, output_dir = _make_fake_env(tmp_path, _FAKE_MAIN_CRASH)
        result = eval_all._run_one_task(
            _default_args(task_suite_name="fake_suite"),
            task_id=0,
            log_dir=str(log_dir),
            cwd=str(fake_dir),
            output_dir=str(output_dir),
        )
        assert math.isnan(result["success_rate"])
        assert result["returncode"] != 0
        # The traceback should be captured in the log file.
        log_path = pathlib.Path(result["log_path"])
        assert "Traceback" in log_path.read_text()

    def test_per_task_log_filename_uses_task_id(self, tmp_path: pathlib.Path) -> None:
        """Two tasks in the same log_dir must produce distinct log files."""
        fake_dir, log_dir, output_dir = _make_fake_env(tmp_path, _FAKE_MAIN_SUCCESS)
        r0 = eval_all._run_one_task(
            _default_args(),
            task_id=0,
            log_dir=str(log_dir),
            cwd=str(fake_dir),
            output_dir=str(output_dir),
        )
        r5 = eval_all._run_one_task(
            _default_args(),
            task_id=5,
            log_dir=str(log_dir),
            cwd=str(fake_dir),
            output_dir=str(output_dir),
        )
        assert r0["log_path"] != r5["log_path"]
        assert "task_00" in r0["log_path"]
        assert "task_05" in r5["log_path"]


# --------------------------------------------------------- Args dataclass


class TestArgsDefaults:
    """Sanity-check the Args defaults so a refactor doesn't silently change them."""

    def test_default_values(self) -> None:
        args = eval_all.Args()
        assert args.host == "0.0.0.0"
        assert args.port == 8000
        assert args.task_suite_name == "libero_goal_task"
        assert args.num_episodes == 15
        assert args.num_steps_wait == 10
        assert args.replan_steps == 5
        assert args.resize_size == 224
        assert args.fps == 10
        assert args.seed == 7
        assert args.num_workers == 10
        assert args.output_dir is None
        assert args.max_steps is None
        assert args.render_cameras == ["agentview", "eye_in_hand"]
