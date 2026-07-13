from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import yaml

from XPolicyLab.utils.process_data import get_action_dim, get_robot_action_dim_info


class RobotMetadataFallbackTest(unittest.TestCase):
    def test_shell_profile_registry_includes_bimanual_yam(self):
        self.assertEqual(
            get_robot_action_dim_info("bimanual_yam"),
            {"arm_dim": [6, 6], "ee_dim": [1, 1]},
        )
        self.assertEqual(get_action_dim("bimanual_yam"), 14)

    def test_robodojo_environment_to_robot_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env_dir = root / "configs" / "environment"
            robot_dir = root / "configs" / "robot"
            env_dir.mkdir(parents=True)
            robot_dir.mkdir(parents=True)
            (env_dir / "custom.yml").write_text(yaml.safe_dump({"config": {"robot": "dual_custom"}}))
            (robot_dir / "_robot_info.json").write_text(
                json.dumps({"dual_custom": {"arm_dim": [4, 4], "ee_dim": [2, 2]}})
            )
            with mock.patch.dict(os.environ, {"ROBODOJO_ROOT": str(root)}):
                self.assertEqual(get_action_dim("custom"), 12)


if __name__ == "__main__":
    unittest.main()
