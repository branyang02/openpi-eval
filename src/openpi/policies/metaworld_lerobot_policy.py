import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class MetaworldInputs(transforms.DataTransformFn):
    """
    Adapted from libero_policy.LiberoInputs
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # NOTE: action and state padding is handled in transforms.PadStatesAndActions

        # TODO(branyang02): Should we add third camera angle as well?
        base_image = _parse_image(data["observation/image"])  # Main Camera
        wrist_image = _parse_image(data["observation/wrist_image"])  # Wrist Camera

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class MetaworldOutputs(transforms.DataTransformFn):
    """
    Adapted from libero_policy.LiberoOutputs
    """

    def __call__(self, data: dict) -> dict:
        # For Metaworld, we only return the first 4 actions (since the rest is padding).
        return {"actions": np.asarray(data["actions"][..., :4])}
