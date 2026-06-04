"""Pytest session bootstrap for examples/liberopro_env tests.

Two side effects, both gated:

1. **Make `examples/liberopro_env` importable** so test files can `import main`
   and `import setup_liberopro_config` directly. pytest does not add the parent
   of the tests/ folder to sys.path automatically.

2. **Bootstrap `~/.liberopro/config.yaml` if missing.** The `libero.libero` package
   has a top-level `input()` prompt at module load time when its config file
   doesn't exist (see `third_party/liberopro/libero/libero/__init__.py`). Under
   pytest there is no TTY, so the prompt raises `EOFError` and `import main`
   fails during pytest collection. We avoid that by writing the default config
   *only when it doesn't already exist*. We never overwrite an existing config
   on a developer machine.
"""

from __future__ import annotations

import os
import pathlib
import sys

_LIBEROPRO_ENV_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_LIBEROPRO_ENV_DIR) not in sys.path:
    sys.path.insert(0, str(_LIBEROPRO_ENV_DIR))

os.environ.setdefault("LIBERO_CONFIG_PATH", str(pathlib.Path.home() / ".liberopro"))

# Conditional bootstrap: only write the config if it isn't there yet.
_CONFIG_PATH = pathlib.Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"
if not _CONFIG_PATH.exists():
    import setup_liberopro_config

    setup_liberopro_config.setup_liberopro_config()
