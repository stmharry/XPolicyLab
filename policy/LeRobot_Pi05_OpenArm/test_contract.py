import numpy as np

from XPolicyLab.policy.LeRobot_Pi05_OpenArm.model import (
    gripper_degrees_to_m,
    gripper_m_to_degrees,
    pack_openarm_state,
    unpack_openarm_action,
)


def test_gripper_endpoints():
    np.testing.assert_allclose(gripper_m_to_degrees([0.0, 0.044]), [0.0, -65.0])
    np.testing.assert_allclose(gripper_degrees_to_m([0.0, -65.0]), [0.0, 0.044])


def test_right_first_state_and_action_contract():
    observation = {
        "state": {
            "left_arm_joint_state": np.deg2rad(np.arange(1, 8)),
            "left_ee_joint_state": [0.044],
            "right_arm_joint_state": np.deg2rad(np.arange(11, 18)),
            "right_ee_joint_state": [0.0],
        }
    }
    packed = pack_openarm_state(observation)
    np.testing.assert_allclose(packed[:8], [11, 12, 13, 14, 15, 16, 17, 0], atol=1e-5)
    np.testing.assert_allclose(packed[8:], [1, 2, 3, 4, 5, 6, 7, -65], atol=1e-5)
    unpacked = unpack_openarm_action(packed)
    np.testing.assert_allclose(
        unpacked["right_arm_joint_state"], np.deg2rad(np.arange(11, 18)), atol=1e-7
    )
    np.testing.assert_allclose(
        unpacked["left_arm_joint_state"], np.deg2rad(np.arange(1, 8)), atol=1e-7
    )


def test_joint_limits_are_clipped_before_simulator_commands():
    action = np.zeros(16, dtype=np.float32)
    action[:7] = 999
    action[8:15] = -999
    unpacked = unpack_openarm_action(action)
    np.testing.assert_allclose(
        unpacked["right_arm_joint_state"], np.deg2rad([75, 90, 85, 135, 85, 40, 80])
    )
    np.testing.assert_allclose(
        unpacked["left_arm_joint_state"], np.deg2rad([-75, -90, -85, 0, -85, -40, -80])
    )


def test_rejects_wrong_or_non_finite_actions():
    for action in (np.zeros(15), np.full(16, np.nan)):
        try:
            unpack_openarm_action(action)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid action was accepted")
