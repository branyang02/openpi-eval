import numpy as np

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.policies import droid_policy
from openpi.shared import normalize as _normalize


def test_droid_inputs_pi05_shape_and_keys() -> None:
    example = droid_policy.make_droid_example()

    transformed = droid_policy.DroidInputs(model_type=_model.ModelType.PI05)(example)

    assert transformed["state"].shape == (8,)
    assert set(transformed["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert transformed["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert transformed["image_mask"]["right_wrist_0_rgb"] == np.False_
    assert transformed["prompt"] == example["prompt"]


def test_droid_inputs_fast_image_keys() -> None:
    example = droid_policy.make_droid_example()

    transformed = droid_policy.DroidInputs(model_type=_model.ModelType.PI0_FAST)(example)

    assert set(transformed["image"]) == {"base_0_rgb", "base_1_rgb", "wrist_0_rgb"}
    assert all(transformed["image_mask"].values())


def test_droid_jointpos_outputs_decode_absolute_arm_targets() -> None:
    state = np.arange(8, dtype=np.float32)
    actions = np.zeros((2, 8), dtype=np.float32)
    data = {"state": state, "actions": actions}

    data = _transforms.AbsoluteActions(_transforms.make_bool_mask(7, -1))(data)
    outputs = droid_policy.DroidOutputs()(data)

    assert outputs["actions"].shape == (2, 8)
    np.testing.assert_allclose(outputs["actions"][:, :7], np.broadcast_to(state[:7], (2, 7)))
    np.testing.assert_allclose(outputs["actions"][:, 7], 0.0)


def test_droid_pi05_serving_transforms_decode_padded_actions() -> None:
    example = droid_policy.make_droid_example()
    state = np.linspace(0.1, 0.8, 8, dtype=np.float32)
    action_mean = np.linspace(1.0, 4.1, 32, dtype=np.float32)
    norm_stats = {
        "state": _normalize.NormStats(
            mean=np.linspace(-0.5, 0.5, 32, dtype=np.float32),
            std=np.full(32, 2.0, dtype=np.float32),
        ),
        "actions": _normalize.NormStats(
            mean=action_mean,
            std=np.full(32, 3.0, dtype=np.float32),
        ),
    }
    example["observation/joint_position"] = state[:7]
    example["observation/gripper_position"] = state[7:]

    inputs = _transforms.compose(
        [
            droid_policy.DroidInputs(model_type=_model.ModelType.PI05),
            _transforms.Normalize(norm_stats),
            _transforms.PadStatesAndActions(32),
        ]
    )(example)
    outputs = _transforms.compose(
        [
            _transforms.Unnormalize(norm_stats),
            _transforms.AbsoluteActions(_transforms.make_bool_mask(7, -1)),
            droid_policy.DroidOutputs(),
        ]
    )({"state": inputs["state"], "actions": np.zeros((2, 32), dtype=np.float32)})

    expected_actions = np.broadcast_to(action_mean[:8], (2, 8)).copy()
    expected_actions[:, :7] += state[:7]
    np.testing.assert_allclose(outputs["actions"], expected_actions, rtol=1e-5, atol=1e-5)
