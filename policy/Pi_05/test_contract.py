from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from XPolicyLab.policy.Pi_05 import contract


class Pi05ArxContractTest(unittest.TestCase):
    def test_profile_alias_is_pinned_and_other_names_are_unchanged(self):
        original = {"ckpt_name": "local_run", "checkpoint_num": 123}
        self.assertEqual(contract.apply_checkpoint_profile(original), original)

        with mock.patch.dict(os.environ, {"ROBODOJO_STORAGE_ROOT": "/runtime"}):
            cfg = contract.apply_checkpoint_profile({"ckpt_name": contract.PROFILE_NAME})
        self.assertEqual(cfg["checkpoint_num"], 55000)
        self.assertEqual(cfg["train_config_name"], contract.TRAIN_CONFIG_NAME)
        self.assertEqual(cfg["actions_per_chunk"], 50)
        self.assertEqual(
            cfg["model_path"],
            f"/runtime/model_weights/Pi_05/{contract.PROFILE_NAME}/{contract.HF_REVISION}/checkpoints/55000",
        )

    def test_gripper_contract_round_trips_and_clips_predictions(self):
        values = np.arange(14, dtype=np.float32)
        values[[6, 13]] = [0.0, 1.0]
        physical = contract.robodojo_to_checkpoint(values)
        np.testing.assert_allclose(physical[[6, 13]], [-0.01, 0.044])

        actions = np.tile(physical, (50, 1))
        actions[0, 6] = -1.0
        actions[0, 13] = 1.0
        normalized = contract.checkpoint_to_robodojo(actions)
        self.assertEqual(normalized.shape, (50, 14))
        np.testing.assert_allclose(normalized[1, [6, 13]], [0.0, 1.0])
        np.testing.assert_allclose(normalized[0, [6, 13]], [0.0, 1.0])

    def test_profile_requires_exact_step_and_params(self):
        with tempfile.TemporaryDirectory() as temporary:
            storage = Path(temporary)
            with mock.patch.dict(os.environ, {"ROBODOJO_STORAGE_ROOT": str(storage)}):
                exact = contract.checkpoint_path()
                exact.mkdir(parents=True)
                with self.assertRaises(FileNotFoundError):
                    contract.validate_profile_checkpoint(exact)
                (exact / "params").mkdir()
                contract.validate_profile_checkpoint(exact)
                with self.assertRaisesRegex(ValueError, "requires checkpoint step"):
                    contract.validate_profile_checkpoint(exact.parent / "40000")

    def test_openpi_config_declares_absolute_50_action_contract(self):
        config_path = Path(__file__).parent / "openpi" / "src" / "openpi" / "training" / "config.py"
        source = config_path.read_text()
        profile_start = source.index('name="pi05_arx5_multitask_v1"')
        profile = source[profile_start : profile_start + 1800]
        self.assertIn("Pi0Config(pi05=True, action_horizon=50)", profile)
        self.assertIn("use_delta_joint_actions=False", profile)
        self.assertIn("adapt_to_pi=False", profile)
        self.assertLess(profile.index('"cam_high"'), profile.index('"cam_left_wrist"'))
        self.assertLess(profile.index('"cam_left_wrist"'), profile.index('"cam_right_wrist"'))


if __name__ == "__main__":
    unittest.main()
