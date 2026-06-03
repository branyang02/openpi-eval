"""Pytest session bootstrap for examples/robocasa_env tests.

Adds `examples/robocasa_env` to sys.path so test files can `import main`
and `import eval_all` directly. pytest does not add the parent of the
tests/ folder to sys.path automatically.
"""

from __future__ import annotations

import pathlib
import sys

_ROBOCASA_ENV_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_ROBOCASA_ENV_DIR) not in sys.path:
    sys.path.insert(0, str(_ROBOCASA_ENV_DIR))
