from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

from XPolicyLab.policy.MolmoACT2 import contract


class MolmoYamContractTest(unittest.TestCase):
    def test_profile_alias_is_pinned_and_other_names_are_unchanged(self):
        original = {"ckpt_name": "local_run", "actions_per_chunk": 7}
        self.assertEqual(contract.apply_checkpoint_profile(original), original)

        with mock.patch.dict(os.environ, {"ROBODOJO_STORAGE_ROOT": "/runtime"}):
            cfg = contract.apply_checkpoint_profile({"ckpt_name": contract.PROFILE_NAME})
        self.assertEqual(cfg["checkpoint_backend"], "original_hf")
        self.assertEqual(cfg["actions_per_chunk"], 25)
        self.assertEqual(cfg["predicted_horizon"], 30)
        self.assertEqual(cfg["num_steps"], 10)
        self.assertEqual(cfg["dtype"], "float32")
        self.assertTrue(cfg["enable_inference_cuda_graph"])
        self.assertEqual(cfg["warmup_runs"], 3)
        self.assertEqual(
            cfg["pretrained_path"],
            f"/runtime/model_weights/MolmoACT2/{contract.PROFILE_NAME}/{contract.HF_REVISION}",
        )

    def test_robot_state_camera_and_action_contract(self):
        contract.validate_robot_contract({"arm_dim": [6, 6], "ee_dim": [1, 1]})
        state = np.zeros(14, dtype=np.float32)
        state[[6, 13]] = [1.0, 0.0]
        np.testing.assert_array_equal(contract.validate_state(state), state)

        images = {key: np.zeros(contract.CAMERA_SHAPE, dtype=np.uint8) for key in contract.CAMERA_KEYS}
        contract.validate_camera_payload(images)
        with self.assertRaisesRegex(ValueError, "camera order"):
            contract.validate_camera_payload(dict(reversed(tuple(images.items()))))

        actions = np.zeros((30, 14), dtype=np.float32)
        actions[:, 6] = 1.02
        actions[:, 13] = -0.02
        selected = contract.validate_and_select_actions(actions)
        self.assertEqual(selected.shape, (25, 14))
        np.testing.assert_array_equal(selected[:, 6], np.ones(25))
        np.testing.assert_array_equal(selected[:, 13], np.zeros(25))

    def test_invalid_shapes_and_nonfinite_actions_fail(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            contract.validate_state(np.zeros(13))
        actions = np.zeros((30, 14), dtype=np.float32)
        actions[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            contract.validate_and_select_actions(actions)


if __name__ == "__main__":
    unittest.main()
