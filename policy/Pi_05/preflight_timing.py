"""Cross-check the public pi05 YAM timing contract against RoboDojo configs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import yaml

from XPolicyLab.policy.Pi_05 import contract


def _load_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing timing configuration: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"timing configuration must be a mapping: {path}")
    return data


def validate_timing_chain(root: Path) -> dict[str, int]:
    root = root.resolve()
    environment = _load_mapping(root / "configs/environment/bimanual_yam.yml")
    profile = contract.apply_checkpoint_profile(
        {
            "ckpt_name": contract.YAM_PROFILE_NAME,
            "env_cfg_type": "bimanual_yam",
            "action_type": "joint",
        }
    )
    contract.validate_profile_runtime(
        profile,
        {"arm_dim": [6, 6], "ee_dim": [1, 1]},
    )

    config = environment.get("config") or {}
    sim_name = config.get("sim")
    camera_name = config.get("camera")
    if not isinstance(sim_name, str) or not isinstance(camera_name, str):
        raise ValueError("bimanual_yam must select named sim and camera profiles")

    sim = _load_mapping(root / "configs/sim" / f"{sim_name}.yml")
    observation = environment.get("observation") or {}

    control_hz = int(profile["control_hz"])
    collect_hz = observation.get("collect_freq")
    if isinstance(collect_hz, bool) or collect_hz != control_hz:
        raise ValueError(f"bimanual_yam observation rate must be {control_hz} Hz; got {collect_hz!r}")

    dt = sim.get("dt")
    if isinstance(dt, bool) or not isinstance(dt, int | float) or not math.isfinite(dt) or dt <= 0:
        raise ValueError(f"simulation dt must be finite and positive; got {dt!r}")
    physics_hz_float = 1.0 / float(dt)
    physics_hz = round(physics_hz_float)
    if not math.isclose(physics_hz_float, 240.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"bimanual_yam physics rate must be 240 Hz; got {physics_hz_float}")

    ticks_float = physics_hz_float / control_hz
    ticks_per_action = round(ticks_float)
    if not math.isclose(ticks_float, ticks_per_action, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"physics/control ratio must be integral; got {ticks_float}")
    if ticks_per_action != 8:
        raise ValueError(f"bimanual_yam must execute eight physics ticks per action; got {ticks_per_action}")
    if sim.get("render_interval") != ticks_per_action:
        raise ValueError(
            f"render_interval must equal ticks per action ({ticks_per_action}); got {sim.get('render_interval')!r}"
        )

    expected_cameras = ("cam_head", "cam_left_wrist", "cam_right_wrist")
    setup_resolutions = {
        "bimanual_yam_molmoact2": [640, 360],
        "bimanual_yam_moonlake_office": [640, 480],
    }
    for setup_name, expected_resolution in setup_resolutions.items():
        setup = _load_mapping(root / "configs/environment" / f"{setup_name}.yml")
        setup_camera_name = (setup.get("config") or {}).get("camera")
        if not isinstance(setup_camera_name, str):
            raise ValueError(f"{setup_name} must select a named camera profile")
        camera = _load_mapping(root / "configs/camera" / f"{setup_camera_name}.yml")
        cameras = (camera.get("camera_rig") or {}).get("cameras") or {}
        if tuple(cameras) != expected_cameras:
            raise ValueError(f"{setup_name} cameras must be {expected_cameras}; got {tuple(cameras)}")
        for key in expected_cameras:
            sensor = cameras[key].get("sensor") or {}
            if sensor.get("fps") != control_hz:
                raise ValueError(f"{setup_name}/{key} must run at {control_hz} Hz; got {sensor.get('fps')!r}")
            if sensor.get("stream_resolution") != expected_resolution:
                raise ValueError(
                    f"{setup_name}/{key} must stream at {expected_resolution}; "
                    f"got {sensor.get('stream_resolution')!r}"
                )

    return {
        "predicted_horizon": int(profile["predicted_horizon"]),
        "executed_horizon": int(profile["executed_horizon"]),
        "control_hz": control_hz,
        "physics_hz": physics_hz,
        "ticks_per_action": ticks_per_action,
        "camera_count": len(expected_cameras),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    timing = validate_timing_chain(args.root)
    print(
        f"prediction={timing['predicted_horizon']} execution={timing['executed_horizon']} "
        f"control={timing['control_hz']}Hz physics={timing['physics_hz']}Hz "
        f"ticks_per_action={timing['ticks_per_action']} "
        f"cameras={timing['camera_count']}x640x{{360,480}}@{timing['control_hz']}Hz"
    )


if __name__ == "__main__":
    main()
