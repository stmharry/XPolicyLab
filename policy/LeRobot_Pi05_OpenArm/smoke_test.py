#!/usr/bin/env python3
"""Load folding_final and run one native-shape OpenARM inference chunk."""

import argparse
from pathlib import Path

import numpy as np

from XPolicyLab.policy.LeRobot_Pi05_OpenArm.model import Model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).parent / "checkpoints/folding_final",
    )
    args = parser.parse_args()
    model = Model(
        {
            "checkpoint_path": str(args.checkpoint.resolve()),
            "prompt": "Fold the T-shirt properly.",
            "chunk_size": 30,
        }
    )
    state = {
        "left_arm_joint_state": np.zeros(7, dtype=np.float32),
        "left_ee_joint_state": np.asarray([0.022], dtype=np.float32),
        "right_arm_joint_state": np.zeros(7, dtype=np.float32),
        "right_ee_joint_state": np.asarray([0.022], dtype=np.float32),
    }
    observation = {
        "state": state,
        "vision": {
            "cam_left_wrist": np.zeros((720, 1280, 3), dtype=np.uint8),
            "cam_right_wrist": np.zeros((720, 1280, 3), dtype=np.uint8),
            "cam_head": np.full((480, 640, 3), 240, dtype=np.uint8),
        },
    }
    actions = model._action_chunk(observation)
    if len(actions) != 30:
        raise RuntimeError(f"expected 30 actions, got {len(actions)}")
    flat = []
    for action in actions:
        flat.append(
            np.concatenate(
                (
                    action["right_arm_joint_state"],
                    action["right_ee_joint_state"],
                    action["left_arm_joint_state"],
                    action["left_ee_joint_state"],
                )
            )
        )
    chunk = np.asarray(flat, dtype=np.float32)
    if chunk.shape != (30, 16) or not np.isfinite(chunk).all():
        raise RuntimeError(f"invalid action chunk: shape={chunk.shape}, finite={np.isfinite(chunk).all()}")
    print(f"PASS checkpoint={args.checkpoint} action_shape={chunk.shape} finite=true")


if __name__ == "__main__":
    main()
