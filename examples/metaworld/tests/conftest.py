"""Pytest bootstrap for examples/metaworld tests.

Makes ``examples/metaworld`` importable so test files can ``import main``
directly. pytest does not add the parent of the tests/ folder to sys.path
automatically, so mirror the libero_env/robocasa_env test setup.
"""

from __future__ import annotations

import pathlib
import sys

_METAWORLD_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_METAWORLD_DIR) not in sys.path:
    sys.path.insert(0, str(_METAWORLD_DIR))
