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

    def test_checkpoint_preparation_separates_provenance_from_integrity_gates(self):
        script_path = Path(__file__).parent / "prepare_checkpoint.sh"
        source = script_path.read_text()

        self.assertIn(
            'PARAMS_TAR_SHA256="7ee69681991cdc5e04b4759d3bf93bca5dac6bc98639ec7b00202d2f82fe5b2f"',
            source,
        )
        self.assertIn('tar cf - -C "${DESTINATION}/checkpoints/${CHECKPOINT_STEP}" params', source)
        self.assertNotIn("VERIFY_MODEL_CARD_TAR", source)
        self.assertIn("hashes local uid/gid/mode/mtime metadata", source)

        comparison = source[
            source.index('if [[ "${PARAMS_SHA256}" == "${PARAMS_TAR_SHA256}" ]]') : source.index("printf '%s  %s\\n'")
        ]
        self.assertNotIn("exit", comparison)

        self.assertIn("sha256sum --check --strict", source)
        self.assertIn('"${HF_BIN}" cache verify', source)
        self.assertIn('--revision "${REVISION}"', source)
        self.assertNotIn("--fail-on-extra-files", source)

    def test_installer_targets_the_policy_venv_even_from_an_active_root_venv(self):
        source = (Path(__file__).parent / "install.sh").read_text()

        self.assertIn('POLICY_PYTHON="${OPENPI_ROOT}/.venv/bin/python"', source)
        self.assertEqual(source.count('--python "${POLICY_PYTHON}"'), 2)
        self.assertIn('"${POLICY_PYTHON}" -c "import XPolicyLab', source)


class Pi05YamContractTest(unittest.TestCase):
    def test_profile_alias_pins_snapshot_norm_asset_and_horizon(self):
        with mock.patch.dict(os.environ, {"ROBODOJO_STORAGE_ROOT": "/runtime"}):
            cfg = contract.apply_checkpoint_profile(
                {
                    "ckpt_name": contract.YAM_PROFILE_NAME,
                    "env_cfg_type": "bimanual_yam",
                    "action_type": "joint",
                    "checkpoint_num": 59999,
                }
            )

        expected_root = f"/runtime/model_weights/Pi_05/{contract.YAM_PROFILE_NAME}/{contract.YAM_HF_REVISION}"
        self.assertEqual(cfg["checkpoint_profile"], contract.YAM_PROFILE_NAME)
        self.assertEqual(cfg["model_path"], expected_root)
        self.assertEqual(
            cfg["norm_stats_path"],
            f"{expected_root}/assets/{contract.YAM_NORM_ASSET_ID}",
        )
        self.assertEqual(cfg["train_config_name"], "yam_pi05")
        self.assertEqual(cfg["predicted_horizon"], 16)
        self.assertEqual(cfg["actions_per_chunk"], 16)
        self.assertNotIn("checkpoint_num", cfg)

    def test_profile_requires_canonical_yam_joint_runtime(self):
        cfg = contract.apply_checkpoint_profile(
            {
                "ckpt_name": contract.YAM_PROFILE_NAME,
                "env_cfg_type": "bimanual_yam",
                "action_type": "joint",
            }
        )
        robot_info = {"arm_dim": [6, 6], "ee_dim": [1, 1]}
        contract.validate_profile_runtime(cfg, robot_info)

        with self.assertRaisesRegex(ValueError, "bimanual_yam"):
            contract.validate_profile_runtime({**cfg, "env_cfg_type": "arx_x5"}, robot_info)
        with self.assertRaisesRegex(ValueError, "action_type='joint'"):
            contract.validate_profile_runtime({**cfg, "action_type": "ee"}, robot_info)

    def test_profile_requires_exact_snapshot_params_and_norm_stats(self):
        with tempfile.TemporaryDirectory() as temporary:
            storage = Path(temporary)
            with mock.patch.dict(os.environ, {"ROBODOJO_STORAGE_ROOT": str(storage)}):
                exact = contract.checkpoint_path(contract.YAM_PROFILE_NAME)
                (exact / "params").mkdir(parents=True)
                with self.assertRaisesRegex(FileNotFoundError, "normalization stats"):
                    contract.validate_profile_checkpoint(exact, contract.YAM_PROFILE_NAME)

                norm_dir = exact / "assets" / contract.YAM_NORM_ASSET_ID
                norm_dir.mkdir(parents=True)
                (norm_dir / "norm_stats.json").touch()
                contract.validate_profile_checkpoint(exact, contract.YAM_PROFILE_NAME)

                with self.assertRaisesRegex(ValueError, "pinned snapshot root"):
                    contract.validate_profile_checkpoint(exact / "checkpoints" / "1", contract.YAM_PROFILE_NAME)

    def test_state_and_action_use_shared_molmoact2_frame(self):
        cfg = {"checkpoint_profile": contract.YAM_PROFILE_NAME}
        state = np.zeros(14, dtype=np.float32)
        state[[4, 11]] = [0.25, -0.75]
        state[[6, 13]] = [0.0, 1.0]
        checkpoint_state = contract.robodojo_state_to_checkpoint(cfg, state)
        np.testing.assert_allclose(checkpoint_state[[4, 11]], [-0.25, 0.75])

        checkpoint_actions = np.tile(checkpoint_state, (16, 1))
        checkpoint_actions[0, [6, 13]] = [-0.01, 1.01]
        actions = contract.checkpoint_actions_to_robodojo(cfg, checkpoint_actions)
        self.assertEqual(actions.shape, (16, 14))
        np.testing.assert_allclose(actions[1, [4, 11]], [0.25, -0.75])
        np.testing.assert_allclose(actions[0, [6, 13]], [0.0, 1.0])

        with self.assertRaisesRegex(ValueError, r"\(16, 14\)"):
            contract.checkpoint_actions_to_robodojo(cfg, checkpoint_actions[:-1])

    def test_camera_contract_is_only_enforced_for_yam_profile(self):
        images = {
            key: np.zeros((3, 360, 640), dtype=np.uint8) for key in ("cam_high", "cam_left_wrist", "cam_right_wrist")
        }
        contract.validate_profile_camera_payload(
            {"checkpoint_profile": contract.YAM_PROFILE_NAME},
            images,
        )
        with self.assertRaisesRegex(ValueError, "camera order"):
            contract.validate_profile_camera_payload(
                {"checkpoint_profile": contract.YAM_PROFILE_NAME},
                dict(reversed(tuple(images.items()))),
            )
        contract.validate_profile_camera_payload({}, {})

    def test_openpi_config_reconstructs_released_yam_training_contract(self):
        config_path = Path(__file__).parent / "openpi" / "src" / "openpi" / "training" / "config.py"
        source = config_path.read_text()
        profile_start = source.index('name="yam_pi05"')
        profile = source[profile_start : profile_start + 2400]

        self.assertIn("Pi0Config(pi05=True, action_dim=32, action_horizon=16)", profile)
        self.assertIn('repo_id="yam-bimanual-merged"', profile)
        self.assertIn('asset_id="yam-bimanual-merged"', profile)
        self.assertIn("use_delta_joint_actions=False", profile)
        self.assertIn("adapt_to_pi=False", profile)
        self.assertIn("prompt_from_task=True", profile)
        self.assertLess(profile.index('"observation.images.top"'), profile.index('"observation.images.left"'))
        self.assertLess(profile.index('"observation.images.left"'), profile.index('"observation.images.right"'))

    def test_checkpoint_preparation_selects_and_fully_verifies_yam_snapshot(self):
        script_path = Path(__file__).parent / "prepare_checkpoint.sh"
        source = script_path.read_text()

        self.assertIn('YAM_PROFILE="pi05_yam_molmoact2"', source)
        self.assertIn('REPO_ID="robocurve/pi05-yam-molmoact2"', source)
        self.assertIn(
            'REVISION="df991e11e8f6540098338c56342b1143fac5b952"',
            source,
        )
        self.assertIn(
            'NORM_SHA256="16daf28cec63d4829f01d7858bfed079ad18e183ce826a268f66c6669f323863"',
            source,
        )
        self.assertIn(
            'PARAMS_METADATA_SHA256="303a4e354814928e1d29b75e310f2c1ac7e7e29b62f48395b631045ca1cffc73"',
            source,
        )
        self.assertIn("--fail-on-missing-files", source)
        self.assertIn("DOWNLOAD_INCLUDES=()", source)


if __name__ == "__main__":
    unittest.main()
