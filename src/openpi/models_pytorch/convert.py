"""Auto-conversion of JAX checkpoints to PyTorch format with caching.

Checks if conversion is needed by hashing the JAX checkpoint metadata + config name.
Skips conversion if an up-to-date model.safetensors already exists.
"""

import hashlib
import importlib
import logging
from pathlib import Path
import sys

from openpi.training import config as _config

logger = logging.getLogger(__name__)


def ensure_pytorch_checkpoint(checkpoint_dir: str, config_name: str) -> None:
    """Convert JAX checkpoint to PyTorch if needed. No-op if already converted and up-to-date."""
    checkpoint_path = Path(checkpoint_dir)
    safetensors_path = checkpoint_path / "model.safetensors"
    hash_path = checkpoint_path / ".pytorch_conversion_hash"

    current_hash = _compute_checkpoint_hash(checkpoint_path, config_name)

    # Check if conversion is up-to-date
    if safetensors_path.exists() and hash_path.exists():
        stored_hash = hash_path.read_text().strip()
        if stored_hash == current_hash:
            logger.info("PyTorch checkpoint is up-to-date, skipping conversion.")
            return

    # Run conversion
    logger.info("Converting JAX checkpoint to PyTorch format...")
    convert_module = _import_conversion_module()
    model_config = _config.get_config(config_name).model
    convert_module.convert_pi0_checkpoint(str(checkpoint_path), "bfloat16", str(checkpoint_path), model_config)

    # Save hash
    hash_path.write_text(current_hash)
    logger.info("JAX -> PyTorch conversion complete. Saved model.safetensors to %s", checkpoint_path)


def _compute_checkpoint_hash(checkpoint_dir: Path, config_name: str) -> str:
    """Hash the JAX checkpoint metadata + config name to detect changes."""
    h = hashlib.sha256()
    metadata_path = checkpoint_dir / "params" / "_METADATA"
    if metadata_path.exists():
        h.update(metadata_path.read_bytes())
    else:
        # Fall back to _CHECKPOINT_METADATA if params/_METADATA doesn't exist
        fallback = checkpoint_dir / "_CHECKPOINT_METADATA"
        if fallback.exists():
            h.update(fallback.read_bytes())
    h.update(config_name.encode())
    return h.hexdigest()


def _import_conversion_module():
    """Import the conversion module from examples/convert_jax_model_to_pytorch.py."""
    examples_dir = Path(__file__).parent.parent.parent.parent / "examples"
    module_path = examples_dir / "convert_jax_model_to_pytorch.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Conversion script not found at {module_path}")

    # Add examples dir to sys.path temporarily and import
    sys_path_entry = str(examples_dir)
    if sys_path_entry not in sys.path:
        sys.path.insert(0, sys_path_entry)

    spec = importlib.util.spec_from_file_location("convert_jax_model_to_pytorch", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
