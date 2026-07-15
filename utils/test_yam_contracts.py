from __future__ import annotations

import unittest

import numpy as np

from XPolicyLab.utils import bimanual_yam_contract as yam, yam_molmoact2_frame as molmoact2_frame


class BimanualYamContractTest(unittest.TestCase):
    def test_canonical_contract_and_dataset_frame_are_independent(self):
        yam.validate_environment("bimanual_yam")
        yam.validate_robot_contract({"arm_dim": [6, 6], "ee_dim": [1, 1]})
        with self.assertRaisesRegex(ValueError, "env_cfg_type"):
            yam.validate_environment("arx_x5")

        values = np.arange(14, dtype=np.float32)
        transformed = molmoact2_frame.simulator_to_dataset(values)
        np.testing.assert_array_equal(
            transformed,
            values * np.array([1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1]),
        )
        np.testing.assert_array_equal(molmoact2_frame.dataset_to_simulator(transformed), values)
        np.testing.assert_array_equal(values, np.arange(14, dtype=np.float32))

    def test_action_validation_selects_and_clips(self):
        actions = np.zeros((30, 14), dtype=np.float32)
        actions[:, 6] = 1.02
        actions[:, 13] = -0.02
        selected = yam.validate_action_chunk(actions, predicted_horizon=30, executed_horizon=25)
        self.assertEqual(selected.shape, (25, 14))
        np.testing.assert_array_equal(selected[:, 6], np.ones(25))
        np.testing.assert_array_equal(selected[:, 13], np.zeros(25))

    def test_camera_validation_accepts_classic_and_moonlake_source_shapes(self):
        for shape in yam.CAMERA_SHAPES:
            images = {key: np.zeros(shape, dtype=np.uint8) for key in yam.CAMERA_KEYS}
            yam.validate_camera_payload(images)

        images = {key: np.zeros((3, 400, 640), dtype=np.uint8) for key in yam.CAMERA_KEYS}
        with self.assertRaisesRegex(ValueError, "one of"):
            yam.validate_camera_payload(images)


if __name__ == "__main__":
    unittest.main()
