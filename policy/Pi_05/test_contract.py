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
        self.assertNotIn("executed_horizon", cfg)
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
        self.assertEqual(cfg["executed_horizon"], 8)
        self.assertEqual(cfg["actions_per_chunk"], 8)
        self.assertEqual(cfg["control_hz"], 30)
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

        for key, value in (
            ("predicted_horizon", 15),
            ("executed_horizon", 16),
            ("actions_per_chunk", 16),
            ("control_hz", 20),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                contract.validate_profile_runtime({**cfg, key: value}, robot_info)

    def test_pickup_profile_pins_bimanual_lerobot_checkpoint_and_native_timing(self):
        with mock.patch.dict(os.environ, {"ROBODOJO_STORAGE_ROOT": "/runtime"}):
            cfg = contract.apply_checkpoint_profile(
                {
                    "ckpt_name": contract.YAM_PICKUP_PROFILE_NAME,
                    "env_cfg_type": "bimanual_yam",
                    "action_type": "joint",
                    "checkpoint_num": 59999,
                }
            )

        expected_root = (
            f"/runtime/model_weights/Pi_05/{contract.YAM_PICKUP_PROFILE_NAME}/"
            f"{contract.YAM_PICKUP_HF_REVISION}"
        )
        self.assertEqual(cfg["checkpoint_profile"], contract.YAM_PICKUP_PROFILE_NAME)
        self.assertEqual(cfg["model_path"], expected_root)
        self.assertEqual(cfg["policy_backend"], "lerobot_pi05")
        self.assertEqual(cfg["predicted_horizon"], 50)
        self.assertEqual(cfg["executed_horizon"], 8)
        self.assertEqual(cfg["actions_per_chunk"], 8)
        self.assertEqual(cfg["control_hz"], 30)
        self.assertNotIn("checkpoint_num", cfg)
        self.assertEqual(
            contract.YAM_PICKUP_TASK_PROMPT,
            "Pick and place the object",
        )

        contract.validate_profile_runtime(cfg, {"arm_dim": [6, 6], "ee_dim": [1, 1]})
        with self.assertRaisesRegex(ValueError, "executed_horizon"):
            contract.validate_profile_runtime({**cfg, "executed_horizon": 16}, {"arm_dim": [6, 6], "ee_dim": [1, 1]})

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
        self.assertEqual(actions.shape, (8, 14))
        np.testing.assert_allclose(actions[1, [4, 11]], [0.25, -0.75])
        np.testing.assert_allclose(actions[0, [6, 13]], [0.0, 1.0])

        with self.assertRaisesRegex(ValueError, r"\(16, 14\)"):
            contract.checkpoint_actions_to_robodojo(cfg, checkpoint_actions[:-1])

    def test_pickup_profile_uses_native_yam_frame_and_temporal_cadence(self):
        cfg = contract.apply_checkpoint_profile(
            {
                "ckpt_name": contract.YAM_PICKUP_PROFILE_NAME,
                "env_cfg_type": "bimanual_yam",
                "action_type": "joint",
            }
        )
        state = np.zeros(14, dtype=np.float32)
        state[[4, 11]] = [0.25, -0.75]
        state[[6, 13]] = [0.0, 1.0]
        checkpoint_state = contract.robodojo_state_to_checkpoint(cfg, state)
        np.testing.assert_allclose(checkpoint_state, state)

        checkpoint_actions = np.tile(checkpoint_state, (50, 1))
        checkpoint_actions[:, 0] = np.arange(50)
        checkpoint_actions[:, 6] = 1.0
        checkpoint_actions[:, 13] = 0.8 + np.arange(50) / 500
        actions = contract.checkpoint_actions_to_robodojo(cfg, checkpoint_actions)
        self.assertEqual(actions.shape, (8, 14))
        expected = checkpoint_actions[:8].copy()
        np.testing.assert_allclose(actions, expected)

        post_grasp = contract.checkpoint_actions_to_robodojo(cfg, checkpoint_actions, pickup_grasped=True)
        self.assertEqual(post_grasp.shape, (50, 14))
        np.testing.assert_allclose(post_grasp, checkpoint_actions)

        checkpoint_actions[:, 13] = np.linspace(1.0, 0.0, 50)
        grasp = contract.checkpoint_actions_to_robodojo(cfg, checkpoint_actions)
        self.assertEqual(grasp.shape, (50, 14))
        np.testing.assert_allclose(grasp[:, :7], checkpoint_actions[:, :7])
        np.testing.assert_allclose(grasp[:, 13], checkpoint_actions[:, 13])
        np.testing.assert_allclose(grasp[0, 7:13], checkpoint_actions[0, 7:13])
        np.testing.assert_allclose(
            grasp[20:, 7:13],
            checkpoint_actions[20:, 7:13] + contract.YAM_PICKUP_GRASP_JOINT_CALIBRATION,
        )
        np.testing.assert_allclose(
            grasp[10, 7:13],
            checkpoint_actions[10, 7:13] + contract.YAM_PICKUP_GRASP_JOINT_CALIBRATION / 2,
        )

        with self.assertRaisesRegex(ValueError, r"\(50, 14\)"):
            contract.checkpoint_actions_to_robodojo(cfg, checkpoint_actions[:-1])

    def test_pickup_grasp_calibration_only_changes_checkpoint_selected_arm(self):
        actions = np.zeros((50, 14), dtype=np.float32)

        right = contract.calibrate_yam_pickup_grasp(actions, (False, True))
        np.testing.assert_allclose(right[:, :7], actions[:, :7])
        np.testing.assert_allclose(right[:, 13], actions[:, 13])
        np.testing.assert_allclose(right[0, 7:13], 0.0)
        expected = np.tile(contract.YAM_PICKUP_GRASP_JOINT_CALIBRATION, (30, 1))
        np.testing.assert_allclose(right[20:, 7:13], expected)

        left = contract.calibrate_yam_pickup_grasp(actions, (True, False))
        np.testing.assert_allclose(left[:, 7:], actions[:, 7:])
        np.testing.assert_allclose(left[:, 6], actions[:, 6])
        np.testing.assert_allclose(left[20:, :6], expected)

        with self.assertRaisesRegex(ValueError, "closing-arm state"):
            contract.calibrate_yam_pickup_grasp(actions, (True,))

    def test_pickup_profile_requires_complete_lerobot_snapshot(self):
        required = (
            "config.json",
            "model.safetensors",
            "policy_preprocessor.json",
            "policy_preprocessor_step_3_normalizer_processor.safetensors",
            "policy_postprocessor.json",
            "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        )
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.dict(os.environ, {"ROBODOJO_STORAGE_ROOT": temporary}):
                exact = contract.checkpoint_path(contract.YAM_PICKUP_PROFILE_NAME)
                exact.mkdir(parents=True)
                with self.assertRaisesRegex(FileNotFoundError, "LeRobot PI0.5"):
                    contract.validate_profile_checkpoint(exact, contract.YAM_PICKUP_PROFILE_NAME)
                for filename in required:
                    (exact / filename).touch()
                contract.validate_profile_checkpoint(exact, contract.YAM_PICKUP_PROFILE_NAME)

    def test_pickup_profile_holds_only_an_unambiguous_learned_close(self):
        actions = np.ones((4, 14), dtype=np.float32)
        actions[:, 6] = [0.5, 0.19, 0.8, 1.0]
        actions[:, 13] = [0.5, 0.2, 0.18, 0.9]

        held, hold_targets = contract.hold_closed_yam_pickup_grippers(actions, [np.nan, np.nan])

        np.testing.assert_allclose(held, actions)
        np.testing.assert_allclose(hold_targets, [0.0, 0.0])

        held, hold_targets = contract.hold_closed_yam_pickup_grippers(actions, hold_targets)

        np.testing.assert_allclose(held[:, 6], 0.0)
        np.testing.assert_allclose(held[:, 13], 0.0)
        np.testing.assert_allclose(hold_targets, [0.0, 0.0])

        with self.assertRaisesRegex(ValueError, "gripper state"):
            contract.hold_closed_yam_pickup_grippers(actions, [np.nan])

    def test_timing_preflight_matches_robodojo_profiles(self):
        from XPolicyLab.policy.Pi_05.preflight_timing import validate_timing_chain

        root = Path(__file__).resolve().parents[3]
        self.assertEqual(
            validate_timing_chain(root, contract.YAM_PROFILE_NAME),
            {
                "predicted_horizon": 16,
                "executed_horizon": 8,
                "control_hz": 30,
                "physics_hz": 240,
                "ticks_per_action": 8,
                "camera_count": 3,
            },
        )
        self.assertEqual(
            validate_timing_chain(root, contract.YAM_PICKUP_PROFILE_NAME),
            {
                "predicted_horizon": 50,
                "executed_horizon": 8,
                "control_hz": 30,
                "physics_hz": 240,
                "ticks_per_action": 8,
                "camera_count": 3,
            },
        )

    def test_camera_contract_is_only_enforced_for_yam_profile(self):
        for height in (360, 480):
            images = {
                key: np.zeros((3, height, 640), dtype=np.uint8)
                for key in ("cam_high", "cam_left_wrist", "cam_right_wrist")
            }
            contract.validate_profile_camera_payload(
                {"checkpoint_profile": contract.YAM_PROFILE_NAME},
                images,
            )
        images = {
            key: np.zeros((3, 360, 640), dtype=np.uint8)
            for key in ("cam_high", "cam_left_wrist", "cam_right_wrist")
        }
        with self.assertRaisesRegex(ValueError, "camera order"):
            contract.validate_profile_camera_payload(
                {"checkpoint_profile": contract.YAM_PROFILE_NAME},
                dict(reversed(tuple(images.items()))),
            )
        contract.validate_profile_camera_payload(
            {"checkpoint_profile": contract.YAM_PICKUP_PROFILE_NAME},
            images,
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

    def test_checkpoint_preparation_pins_pickup_model_without_training_checkpoints(self):
        source = (Path(__file__).parent / "prepare_checkpoint.sh").read_text()

        self.assertIn('YAM_PICKUP_PROFILE="pi05_yam_abc_pickplace"', source)
        self.assertIn('REPO_ID="pztang/yam-abc-pickplace-safe-pi05-8gpu-m1"', source)
        self.assertIn('REVISION="44cc2cd8d7edf9be332bc3cfa7475484897c61e9"', source)
        self.assertIn(
            'MODEL_SHA256="0c697969f4cefbfe781b83389212b40493ce5ed51dc5c31f15a1d2b31233eebc"',
            source,
        )
        self.assertIn('--include "model.safetensors"', source)
        self.assertNotIn('--include "checkpoints/**"', source)
        # Only the complete MolmoAct2 snapshot requires the full remote
        # manifest. The pickup profile deliberately selects root pretrained
        # files and must not reject omitted training checkpoints.
        self.assertEqual(source.count("--fail-on-missing-files"), 1)

    def test_lerobot_runtime_pins_official_openpi_transformers_branch(self):
        project = (Path(__file__).parent / "openpi" / "pyproject.toml").read_text()
        lock = (Path(__file__).parent / "openpi" / "uv.lock").read_text()

        revision = "dcddb970176382c0fcf4521b0c0e6fc15894dfe0"
        self.assertIn(f'rev = "{revision}"', project)
        self.assertIn(f"transformers.git?rev={revision}", lock)

    def test_lerobot_resize_preserves_training_aspect_ratio(self):
        from XPolicyLab.policy.Pi_05.lerobot_backend import _resize_chw

        training_source = np.full((3, 360, 640), 255, dtype=np.uint8)
        moonlake = np.full((3, 480, 640), 255, dtype=np.uint8)
        moonlake[:, :60] = 127
        moonlake[:, 420:] = 127

        expected = _resize_chw(training_source, height=240, width=360)
        resized = _resize_chw(moonlake, height=240, width=360)

        self.assertEqual(resized.shape, (240, 360, 3))
        np.testing.assert_array_equal(resized, expected)
        self.assertFalse(resized[:18].any())
        self.assertTrue((resized[18:221] == 255).all())
        self.assertFalse(resized[221:].any())

    def test_lerobot_remaps_only_legacy_vision_tower_keys(self):
        from XPolicyLab.policy.Pi_05.lerobot_backend import _remap_checkpoint_key

        legacy = "model.paligemma_with_expert.paligemma.model.vision_tower.encoder.layers.0.weight"
        current = (
            "model.paligemma_with_expert.paligemma.model."
            "vision_tower.vision_model.encoder.layers.0.weight"
        )
        self.assertEqual(_remap_checkpoint_key(legacy), current)
        self.assertEqual(_remap_checkpoint_key(current), current)
        self.assertEqual(_remap_checkpoint_key("model.action_out_proj.weight"), "model.action_out_proj.weight")

    def test_lerobot_accepts_omitted_disabled_action_processors(self):
        from XPolicyLab.policy.Pi_05.lerobot_backend import _validate_saved_processors

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "policy_preprocessor.json").write_text('{"steps": []}')
            (root / "policy_postprocessor.json").write_text('{"steps": []}')
            _validate_saved_processors(root)

            (root / "policy_preprocessor.json").write_text(
                '{"steps": [{"registry_name": "delta_actions_processor", "config": {"enabled": true}}]}'
            )
            with self.assertRaisesRegex(ValueError, "disabled delta_actions_processor"):
                _validate_saved_processors(root)


if __name__ == "__main__":
    unittest.main()
