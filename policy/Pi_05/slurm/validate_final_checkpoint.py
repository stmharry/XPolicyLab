from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from openpi import transforms
from openpi.models import model as model_lib
from openpi.policies import policy_config
from openpi.training import config as train_config

CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def _inference_repack() -> transforms.Group:
    return transforms.Group(
        inputs=[
            transforms.RepackTransform(
                {
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "prompt": "prompt",
                }
            )
        ]
    )


def _latest_checkpoint(
    checkpoint_root: Path, num_train_steps: int, *, expected_checkpoint_step: int | None = None
) -> Path:
    # OpenPI enumerates optimizer updates from zero, so a 30,000-update run
    # writes its terminal checkpoint at step 29,999.
    expected_step = num_train_steps - 1 if expected_checkpoint_step is None else expected_checkpoint_step
    steps = sorted(int(path.name) for path in checkpoint_root.iterdir() if path.is_dir() and path.name.isdigit())
    if not steps or steps[-1] != expected_step:
        raise ValueError(f"Expected final checkpoint {expected_step}, found {steps} in {checkpoint_root}.")
    checkpoint = checkpoint_root / str(expected_step)
    for item in ("params", "assets"):
        if not (checkpoint / item).exists():
            raise FileNotFoundError(f"Final checkpoint is missing {item}: {checkpoint / item}")
    return checkpoint


def _infer_model_actions(policy, observation: dict[str, object]) -> np.ndarray:
    """Run the loaded model before output transforms to validate its padded action contract."""
    inputs = policy._input_transform(dict(observation))
    inputs = jax.tree.map(lambda value: jnp.asarray(value)[np.newaxis, ...], inputs)
    model_observation = model_lib.Observation.from_dict(inputs)
    policy._rng, sample_rng = jax.random.split(policy._rng)
    actions = policy._sample_actions(sample_rng, model_observation, **policy._sample_kwargs)
    return np.asarray(actions[0])


def validate(args: argparse.Namespace) -> dict[str, object]:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415
    except ModuleNotFoundError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    checkpoint = _latest_checkpoint(
        args.checkpoint_root.resolve(),
        args.num_train_steps,
        expected_checkpoint_step=args.expected_checkpoint_step,
    )
    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.dataset_root.resolve(), video_backend="pyav")
    sample = dataset[args.sample_index]

    observation: dict[str, object] = {key: np.asarray(sample[key]) for key in (*CAMERA_KEYS, "observation.state")}
    observation["prompt"] = sample["task"]

    camera_shapes = {key: list(np.asarray(observation[key]).shape) for key in CAMERA_KEYS}
    if set(map(tuple, camera_shapes.values())) != {(3, 480, 640)}:
        raise ValueError(f"Unexpected camera shapes: {camera_shapes}")
    state = np.asarray(observation["observation.state"])
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError(f"Invalid robot state: shape={state.shape}, finite={np.isfinite(state).all()}.")

    config = dataclasses.replace(
        train_config.get_config(args.config),
        assets_base_dir=str(args.assets_base_dir.resolve()),
    )
    policy = policy_config.create_trained_policy(
        config,
        checkpoint,
        repack_transforms=_inference_repack(),
    )
    model_actions = _infer_model_actions(policy, observation)
    if model_actions.shape != (50, args.model_action_dim) or not np.isfinite(model_actions).all():
        raise ValueError(
            f"Invalid model actions: shape={model_actions.shape}, finite={np.isfinite(model_actions).all()}."
        )
    policy.reset()
    result = policy.infer(observation)
    actions = np.asarray(result["actions"])
    if actions.shape != (50, 14) or not np.isfinite(actions).all():
        raise ValueError(f"Invalid inferred actions: shape={actions.shape}, finite={np.isfinite(actions).all()}.")

    payload: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(checkpoint.name),
        "num_train_steps": args.num_train_steps,
        "sample_index": args.sample_index,
        "task": sample["task"],
        "camera_shapes": camera_shapes,
        "state_shape": list(state.shape),
        "model_action_shape": list(model_actions.shape),
        "model_actions_finite": bool(np.isfinite(model_actions).all()),
        "action_shape": list(actions.shape),
        "physical_action_shape": list(actions.shape),
        "physical_actions_finite": bool(np.isfinite(actions).all()),
        "actions_finite": bool(np.isfinite(actions).all()),
        "inference_ms": float(result["policy_timing"]["infer_ms"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_base_aloha_full_real_piper_seed_0")
    parser.add_argument("--repo-id", default="RoboDojo-real_piper_6task-bimanual_piper-joint")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--assets-base-dir", type=Path, required=True)
    parser.add_argument("--num-train-steps", type=int, default=30_000)
    parser.add_argument("--expected-checkpoint-step", type=int)
    parser.add_argument("--model-action-dim", type=int, default=32)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(validate(parse_args()), sort_keys=True))
