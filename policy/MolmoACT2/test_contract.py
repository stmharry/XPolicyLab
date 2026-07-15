from __future__ import annotations

import os
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from XPolicyLab.policy.MolmoACT2 import contract
from XPolicyLab.policy.MolmoACT2.model import _OriginalHFPolicy


class MolmoYamContractTest(unittest.TestCase):
    def test_original_hf_policy_uses_and_resets_configured_generator(self):
        class FakeModel:
            def __init__(self):
                self.generators = []

            def predict_action(self, **kwargs):
                generator = kwargs.get("generator")
                self.generators.append(generator)
                actions = torch.randn((30, 14), generator=generator)
                actions[:, contract.GRIPPER_INDICES] = 0.5
                return SimpleNamespace(actions=actions)

        policy = _OriginalHFPolicy.__new__(_OriginalHFPolicy)
        policy.processor = object()
        policy.model = FakeModel()
        policy.norm_tag = contract.NORM_TAG
        policy.num_steps = contract.FLOW_STEPS
        policy.enable_depth_reasoning = False
        policy.enable_cuda_graph = False
        policy._bridge_yam_joint_5_sign = False
        policy._seed = 17
        policy._generator = torch.Generator(device="cpu").manual_seed(policy._seed)
        policy._candidate_generators = [policy._generator]
        policy._candidate_count = 1
        policy.last_candidate_scores = ()
        policy._lock = threading.Lock()
        payload = {
            "state": np.zeros(contract.STATE_DIM, dtype=np.float32),
            "images": {
                camera: np.zeros(contract.CAMERA_SHAPE, dtype=np.uint8)
                for camera in contract.CAMERA_KEYS
            },
            "prompt": "fold the shirt",
        }

        first = policy.predict(payload)
        second = policy.predict(payload)
        policy.reset()
        replayed = policy.predict(payload)

        self.assertTrue(all(generator is policy._generator for generator in policy.model.generators))
        self.assertFalse(np.array_equal(first, second))
        np.testing.assert_array_equal(first, replayed)

    def test_original_hf_policy_selects_most_active_deterministic_candidate(self):
        class FakeModel:
            def predict_action(self, **kwargs):
                generator = kwargs["generator"]
                candidate_index = generator.initial_seed() - 17
                actions = torch.zeros((30, 14))
                actions[:, 0] = 0.2 * candidate_index
                actions[:, contract.GRIPPER_INDICES] = 1.0 if candidate_index == 0 else 0.5
                return SimpleNamespace(actions=actions)

        policy = _OriginalHFPolicy.__new__(_OriginalHFPolicy)
        policy.processor = object()
        policy.model = FakeModel()
        policy.norm_tag = contract.NORM_TAG
        policy.num_steps = contract.FLOW_STEPS
        policy.enable_depth_reasoning = False
        policy.enable_cuda_graph = False
        policy._bridge_yam_joint_5_sign = False
        policy._seed = 17
        policy._candidate_count = 3
        policy._candidate_generators = [
            torch.Generator(device="cpu").manual_seed(policy._seed + candidate_index)
            for candidate_index in range(policy._candidate_count)
        ]
        policy._generator = policy._candidate_generators[0]
        policy.last_candidate_scores = ()
        policy._lock = threading.Lock()
        payload = {
            "state": np.zeros(contract.STATE_DIM, dtype=np.float32),
            "images": {
                camera: np.zeros(contract.CAMERA_SHAPE, dtype=np.uint8)
                for camera in contract.CAMERA_KEYS
            },
            "prompt": "put the object into the box",
        }

        selected = policy.predict(payload)

        np.testing.assert_allclose(selected[:, 0], np.full(30, 0.4))
        np.testing.assert_allclose(policy.last_candidate_scores, (0.0, 0.2, 0.4))
        policy.reset()
        self.assertEqual(policy.last_candidate_scores, ())
        self.assertEqual(
            [generator.initial_seed() for generator in policy._candidate_generators],
            [17, 18, 19],
        )

    def test_profile_alias_is_pinned_and_other_names_are_unchanged(self):
        original = {"ckpt_name": "local_run", "actions_per_chunk": 7}
        self.assertEqual(contract.apply_checkpoint_profile(original), original)

        with mock.patch.dict(os.environ, {"ROBODOJO_STORAGE_ROOT": "/runtime"}):
            cfg = contract.apply_checkpoint_profile({"ckpt_name": contract.PROFILE_NAME})
        self.assertEqual(cfg["checkpoint_backend"], "original_hf")
        self.assertEqual(cfg["actions_per_chunk"], 30)
        self.assertEqual(cfg["predicted_horizon"], 30)
        self.assertEqual(cfg["num_steps"], 10)
        self.assertEqual(cfg["candidate_count"], 16)
        self.assertEqual(cfg["dtype"], "float32")
        self.assertTrue(cfg["enable_inference_cuda_graph"])
        self.assertEqual(cfg["warmup_runs"], 3)
        self.assertEqual(cfg["embodiment_contract"], "bimanual_yam")
        self.assertEqual(cfg["dataset_frame"], "yam_molmoact2")
        self.assertEqual(
            cfg["pretrained_path"],
            f"/runtime/model_weights/MolmoACT2/{contract.PROFILE_NAME}/{contract.HF_REVISION}",
        )

    def test_robot_state_camera_and_action_contract(self):
        contract.validate_environment("bimanual_yam")
        with self.assertRaisesRegex(ValueError, "env_cfg_type"):
            contract.validate_environment("arx_x5")
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
        self.assertEqual(selected.shape, (30, 14))
        np.testing.assert_array_equal(selected[:, 6], np.ones(30))
        np.testing.assert_array_equal(selected[:, 13], np.zeros(30))

    def test_invalid_shapes_and_nonfinite_actions_fail(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            contract.validate_state(np.zeros(13))
        actions = np.zeros((30, 14), dtype=np.float32)
        actions[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            contract.validate_and_select_actions(actions)

    def test_public_yam_joint_sign_bridge_is_profile_isolated(self):
        public_cfg = contract.apply_checkpoint_profile({"ckpt_name": contract.PROFILE_NAME})
        self.assertTrue(contract.uses_public_yam_joint_sign_bridge(public_cfg))
        self.assertFalse(
            contract.uses_public_yam_joint_sign_bridge(
                {
                    "ckpt_name": "/models/local-original-hf",
                    "checkpoint_backend": "original_hf",
                }
            )
        )
        self.assertFalse(
            contract.uses_public_yam_joint_sign_bridge(
                {
                    "ckpt_name": "local-lerobot",
                    "checkpoint_backend": "lerobot",
                }
            )
        )

    def test_yam_joint_sign_transforms_are_pure_shape_preserving_involutions(self):
        state = np.arange(14, dtype=np.float32)
        original_state = state.copy()
        checkpoint_state = contract.simulator_state_to_checkpoint(state)
        self.assertEqual(checkpoint_state.shape, state.shape)
        self.assertEqual(checkpoint_state.dtype, state.dtype)
        np.testing.assert_array_equal(state, original_state)
        np.testing.assert_array_equal(
            checkpoint_state,
            original_state * np.array([1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1]),
        )
        np.testing.assert_array_equal(contract.simulator_state_to_checkpoint(checkpoint_state), state)

        actions = np.stack((state, state + 14), axis=0)
        original_actions = actions.copy()
        simulator_actions = contract.checkpoint_actions_to_simulator(actions)
        self.assertEqual(simulator_actions.shape, actions.shape)
        self.assertEqual(simulator_actions.dtype, actions.dtype)
        np.testing.assert_array_equal(actions, original_actions)
        np.testing.assert_array_equal(
            contract.checkpoint_actions_to_simulator(simulator_actions),
            actions,
        )
        np.testing.assert_array_equal(
            simulator_actions[:, contract.GRIPPER_INDICES],
            actions[:, contract.GRIPPER_INDICES],
        )

    def test_yam_joint_sign_transforms_reject_non_contract_shapes(self):
        with self.assertRaisesRegex(ValueError, "end in dimension 14"):
            contract.simulator_state_to_checkpoint(np.zeros(13, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "end in dimension 14"):
            contract.checkpoint_actions_to_simulator(np.zeros((30, 13), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
