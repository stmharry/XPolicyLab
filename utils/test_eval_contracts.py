from pathlib import Path

import yaml

from XPolicyLab.policy.MolmoACT2 import contract as molmo
from XPolicyLab.policy.Pi_05 import contract as pi05


ROOT = Path(__file__).resolve().parents[1]
TRACKED = {
    "MolmoACT2": {"molmoact2_bimanual_yam"},
    "Pi_05": {
        "pi05_arx5_multitask_v1",
        "pi05_yam_molmoact2",
        "pi05_yam_abc_pickplace",
    },
    "LeRobot_Pi05_OpenArm": {"folding_final"},
    "SmolVLA": {"smolvla-aloha-bimanual"},
}


def _manifest(policy: str) -> dict:
    path = ROOT / "policy" / policy / "eval_contracts.yml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert set(payload["profiles"]) == TRACKED[policy]
    return payload["profiles"]


def test_tracked_policy_descriptors_are_complete():
    for policy in TRACKED:
        for profile in _manifest(policy).values():
            interface = profile["interface"]
            execution = profile["execution"]
            assert interface["state"]["dimension"] == interface["action"]["dimension"]
            assert interface["action"]["rate_hz"] > 0
            assert execution["strategy"] in {"full_chunk", "fixed_prefix", "adaptive"}
            assert 1 <= execution["nominal_execution_horizon"] <= execution["maximum_execution_horizon"]
            assert execution["maximum_execution_horizon"] <= execution["prediction_horizon"]


def test_pi05_descriptor_matches_checkpoint_execution_constants():
    profiles = _manifest("Pi_05")
    expected = {
        pi05.ARX_PROFILE_NAME: (pi05.ARX_ACTION_HORIZON, pi05.ARX_ACTION_HORIZON, pi05.ARX_ACTION_HORIZON),
        pi05.YAM_PROFILE_NAME: (pi05.YAM_ACTION_HORIZON, pi05.YAM_EXECUTED_HORIZON, pi05.YAM_EXECUTED_HORIZON),
        pi05.YAM_PICKUP_PROFILE_NAME: (
            pi05.YAM_PICKUP_ACTION_HORIZON,
            pi05.YAM_PICKUP_EXECUTED_HORIZON,
            pi05.YAM_PICKUP_POST_GRASP_EXECUTED_HORIZON,
        ),
    }
    for name, horizons in expected.items():
        execution = profiles[name]["execution"]
        assert (
            execution["prediction_horizon"],
            execution["nominal_execution_horizon"],
            execution["maximum_execution_horizon"],
        ) == horizons


def test_molmo_descriptor_matches_checkpoint_execution_constants():
    execution = _manifest("MolmoACT2")[molmo.PROFILE_NAME]["execution"]
    assert execution["prediction_horizon"] == molmo.PREDICTED_HORIZON
    assert execution["nominal_execution_horizon"] == molmo.EXECUTED_HORIZON
    assert execution["maximum_execution_horizon"] == molmo.EXECUTED_HORIZON
