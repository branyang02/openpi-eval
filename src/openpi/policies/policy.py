from collections.abc import Sequence
import time
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


def _iter_leaves(tree):
    if isinstance(tree, dict):
        for value in tree.values():
            yield from _iter_leaves(value)
    else:
        yield tree


def _batch_size(tree: dict) -> int:
    for leaf in _iter_leaves(tree):
        shape = getattr(leaf, "shape", None)
        if shape:
            return int(shape[0])
    raise ValueError("Cannot infer batch size from observation without array leaves.")


def _slice_batch(tree, index: int):
    if isinstance(tree, dict):
        return {key: _slice_batch(value, index) for key, value in tree.items()}
    return tree[index]


def _stack_singles(singles: Sequence[dict]) -> dict:
    if not singles:
        raise ValueError("Cannot batch an empty sequence.")

    first = singles[0]
    output = {}
    for key, value in first.items():
        if isinstance(value, dict):
            output[key] = _stack_singles([single[key] for single in singles])
        else:
            output[key] = np.stack([np.asarray(single[key]) for single in singles], axis=0)
    return output


def collate_transformed_singles(singles: list[dict]) -> dict:
    """Stack single-example transformed inputs back into one model batch."""
    out = {}
    flat_keys = ["state", "tokenized_prompt", "tokenized_prompt_mask"]
    flat_keys.extend(k for k in ["token_ar_mask", "token_loss_mask"] if k in singles[0])

    for key in flat_keys:
        out[key] = jnp.stack([jnp.asarray(example[key]) for example in singles], axis=0)

    for key in ["image", "image_mask"]:
        out[key] = {
            image_key: jnp.stack([jnp.asarray(example[key][image_key]) for example in singles], axis=0)
            for image_key in singles[0][key]
        }

    return out


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        model_type: _model.ModelType,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            model_type: Which model architecture is being served.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._model_type = model_type
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    def infer_batched(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)

        eval_batch_size = _batch_size(inputs)
        singles = [_slice_batch(inputs, i) for i in range(eval_batch_size)]
        singles = [self._input_transform(example) for example in singles]
        inputs = collate_transformed_singles(singles)

        if self._is_pytorch_model:
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device), inputs)
            sample_rng_or_pytorch_device = self._pytorch_device
        else:
            inputs = jax.tree.map(lambda x: jnp.asarray(x), inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)

        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)
            if noise.ndim == 2:
                noise = noise[None, ...]
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time

        if self._is_pytorch_model:
            outputs = jax.tree.map(
                lambda x: np.asarray(x.detach().cpu()) if isinstance(x, torch.Tensor) else np.asarray(x),
                outputs,
            )
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x), outputs)

        if self._model_type == _model.ModelType.PI0_FAST:
            per_sample_outputs = [
                self._output_transform({"state": outputs["state"][i], "actions": outputs["actions"][i]})
                for i in range(eval_batch_size)
            ]
            outputs = {
                key: np.stack([item[key] for item in per_sample_outputs], axis=0) for key in per_sample_outputs[0]
            }
        else:
            outputs = self._output_transform(outputs)

        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def infer_many(self, obs: Sequence[dict], *, noise: np.ndarray | None = None) -> list[dict]:
        if not obs:
            return []

        outputs = self.infer_batched(_stack_singles(obs), noise=noise)
        policy_timing = outputs.pop("policy_timing", None)

        results = []
        for i in range(len(obs)):
            result = _slice_batch(outputs, i)
            if policy_timing is not None:
                result["policy_timing"] = policy_timing
            results.append(result)
        return results

    def warmup_many(self, obs: dict, batch_sizes: Sequence[int]) -> None:
        if not batch_sizes:
            return

        rng = None if self._is_pytorch_model else self._rng
        try:
            for batch_size in batch_sizes:
                if batch_size <= 0:
                    continue
                if batch_size == 1:
                    self.infer(obs)
                else:
                    self.infer_many([obs] * batch_size)
        finally:
            if rng is not None:
                self._rng = rng

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        if "observation/state" in obs and obs["observation/state"].ndim == 2:
            return self.infer_batched(obs, noise=noise)

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata
