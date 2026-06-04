import numpy as np

from openpi.policies import libero_policy


def test_libero_outputs_slice_action_dim_for_unbatched_actions() -> None:
    actions = np.arange(10 * 32).reshape(10, 32)

    output = libero_policy.LiberoOutputs()({"actions": actions})

    assert output["actions"].shape == (10, 7)
    np.testing.assert_array_equal(output["actions"], actions[..., :7])


def test_libero_outputs_slice_action_dim_for_batched_actions() -> None:
    actions = np.arange(2 * 10 * 32).reshape(2, 10, 32)

    output = libero_policy.LiberoOutputs()({"actions": actions})

    assert output["actions"].shape == (2, 10, 7)
    np.testing.assert_array_equal(output["actions"], actions[..., :7])
