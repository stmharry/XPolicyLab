import os
from pathlib import Path

import numpy as np

from XPolicyLab.utils.load_file import load_json, load_yaml

_XPOLICYLAB_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_ROBOT_INFO = Path(__file__).resolve().parent / "robot" / "_robot_info.json"


def _find_robodojo_root() -> Path | None:
    """Find the enclosing RoboDojo checkout without assuming a submodule layout.

    ``ROBODOJO_ROOT`` is authoritative when set.  Walking the source path also
    supports XPolicyLab linked worktrees, whose Git metadata cannot report the
    superproject working tree.
    """

    configured = os.environ.get("ROBODOJO_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if not (root / "configs" / "environment").is_dir():
            raise FileNotFoundError(f"ROBODOJO_ROOT does not contain configs/environment: {root}")
        return root

    for candidate in (_XPOLICYLAB_ROOT, *_XPOLICYLAB_ROOT.parents):
        if (candidate / "configs" / "environment").is_dir() and (
            candidate / "configs" / "robot" / "_robot_info.json"
        ).is_file():
            return candidate
    return None


def _official_env_paths(env_cfg_type: str) -> tuple[Path, Path]:
    env_root = _XPOLICYLAB_ROOT / "env_cfg"
    return env_root / f"{env_cfg_type}.yml", env_root / "robot" / "_robot_info.json"


def _load_robot_action_dim_info(env_cfg_type: str) -> dict:
    """Resolve robot dimensions through upstream, RoboDojo, then profile data.

    The first two branches retain the official environment-to-robot mapping.
    The final direct profile lookup is the shell-facing compatibility registry
    used when an adapter is developed outside a complete RoboDojo checkout.
    """

    official_env, official_robots = _official_env_paths(env_cfg_type)
    if official_env.is_file() and official_robots.is_file():
        env_cfg = load_yaml(str(official_env))
        robot_name = env_cfg["config"]["robot"]
        return load_json(str(official_robots))[robot_name]

    robodojo_root = _find_robodojo_root()
    if robodojo_root is not None:
        env_path = robodojo_root / "configs" / "environment" / f"{env_cfg_type}.yml"
        robot_info_path = robodojo_root / "configs" / "robot" / "_robot_info.json"
        if env_path.is_file() and robot_info_path.is_file():
            env_cfg = load_yaml(str(env_path))
            robot_name = env_cfg["config"]["robot"]
            return load_json(str(robot_info_path))[robot_name]

    if _LOCAL_ROBOT_INFO.is_file():
        profiles = load_json(str(_LOCAL_ROBOT_INFO))
        if env_cfg_type in profiles:
            return profiles[env_cfg_type]

    checked = [str(official_env)]
    if robodojo_root is not None:
        checked.append(str(robodojo_root / "configs" / "environment" / f"{env_cfg_type}.yml"))
    checked.append(f"{_LOCAL_ROBOT_INFO}::{env_cfg_type}")
    raise FileNotFoundError(
        f"Could not resolve robot metadata for environment {env_cfg_type!r}. Checked: {', '.join(checked)}"
    )


def _validate_config(action_type: str, robot_action_dim_info: dict, source_type: str):
    """
    Validate configuration and return normalized values.

    Args:
        action_type: 'joint' or 'ee'.
        robot_action_dim_info: Dict with keys:
            - 'arm_dim': list[int]
            - 'ee_dim': list[int]
        source_type: 'obs' or 'dataset'.

    Returns:
        arm_dims, ee_dims, num_arms
    """
    if action_type not in {"joint", "ee"}:
        raise ValueError(f"Unsupported action_type: {action_type!r}. Supported values are 'joint' and 'ee'.")

    if source_type not in {"obs", "dataset"}:
        raise ValueError(f"Unsupported source_type: {source_type!r}. Supported values are 'obs' and 'dataset'.")

    if "arm_dim" not in robot_action_dim_info or "ee_dim" not in robot_action_dim_info:
        raise KeyError("robot_action_dim_info must contain both 'arm_dim' and 'ee_dim'.")

    arm_dims = robot_action_dim_info["arm_dim"]
    ee_dims = robot_action_dim_info["ee_dim"]

    if not isinstance(arm_dims, (list, tuple)) or not isinstance(ee_dims, (list, tuple)):
        raise TypeError("'arm_dim' and 'ee_dim' must be list or tuple.")

    if len(arm_dims) != len(ee_dims):
        raise ValueError(f"'arm_dim' and 'ee_dim' must have the same length, got {len(arm_dims)} and {len(ee_dims)}.")

    if len(arm_dims) not in {1, 2}:
        raise ValueError(f"Only single-arm or dual-arm robots are supported, got {len(arm_dims)} arms.")

    if any(d <= 0 for d in arm_dims) or any(d <= 0 for d in ee_dims):
        raise ValueError("All dimensions in 'arm_dim' and 'ee_dim' must be positive.")

    return list(arm_dims), list(ee_dims), len(arm_dims)


def _get_state_keys(action_type: str, num_arms: int, source_type: str):
    """
    Return arm keys and ee keys for the current state schema.

    source_type='obs' uses singular keys:
        single-arm:
            action_type='joint' -> ['joint_state'], ['ee_joint_state']
            action_type='ee'    -> ['ee_pose'], ['ee_joint_state']
        dual-arm:
            action_type='joint' -> ['left_arm_joint_state', 'right_arm_joint_state'],
                                   ['left_ee_joint_state', 'right_ee_joint_state']
            action_type='ee'    -> ['left_ee_pose', 'right_ee_pose'],
                                   ['left_ee_joint_state', 'right_ee_joint_state']

    source_type='dataset' uses plural keys:
        single-arm:
            action_type='joint' -> ['joint_states'], ['ee_joint_states']
            action_type='ee'    -> ['ee_poses'], ['ee_joint_states']
        dual-arm:
            action_type='joint' -> ['left_arm_joint_states', 'right_arm_joint_states'],
                                   ['left_ee_joint_states', 'right_ee_joint_states']
            action_type='ee'    -> ['left_ee_poses', 'right_ee_poses'],
                                   ['left_ee_joint_states', 'right_ee_joint_states']
    """
    suffix = "" if source_type == "obs" else "s"

    if num_arms == 1:
        arm_keys = [f"joint_state{suffix}"] if action_type == "joint" else [f"ee_pose{suffix}"]
        ee_keys = [f"ee_joint_state{suffix}"]
    else:
        if action_type == "joint":
            arm_keys = [
                f"left_arm_joint_state{suffix}",
                f"right_arm_joint_state{suffix}",
            ]
        else:
            arm_keys = [
                f"left_ee_pose{suffix}",
                f"right_ee_pose{suffix}",
            ]

        ee_keys = [
            f"left_ee_joint_state{suffix}",
            f"right_ee_joint_state{suffix}",
        ]

    return arm_keys, ee_keys


def _ensure_valid_state_array(name: str, value, expected_last_dim: int) -> np.ndarray:
    """Convert value to np.ndarray and validate only the last dimension."""
    arr = np.asarray(value)

    if arr.shape[-1] != expected_last_dim:
        raise ValueError(f"State field '{name}' last dim mismatch: expected {expected_last_dim}, got {arr.shape[-1]}.")

    return arr


def pack_robot_state(
    obs: dict,
    action_type: str,
    robot_action_dim_info: dict,
    source_type: str = "obs",
    state_type: str = "state",
) -> np.ndarray:
    """
    Pack robot state from obs['state'] into one vector along the last dimension.

    Packing order:
        single-arm:
            [arm_0, ee_0]
        dual-arm:
            [arm_0, ee_0, arm_1, ee_1]
    """
    if state_type not in obs:
        raise KeyError(f"Input obs must contain a '{state_type}' field.")

    state_dict = obs[state_type]

    arm_dims, ee_dims, num_arms = _validate_config(action_type, robot_action_dim_info, source_type)
    arm_keys, ee_keys = _get_state_keys(action_type, num_arms, source_type)

    parts = []
    expected_prefix_shape = None

    for i, (arm_key, ee_key, arm_dim, ee_dim) in enumerate(zip(arm_keys, ee_keys, arm_dims, ee_dims)):
        if arm_key not in state_dict:
            raise KeyError(f"Missing key '{arm_key}' in obs['state'] for arm {i}.")
        if ee_key not in state_dict:
            raise KeyError(f"Missing key '{ee_key}' in obs['state'] for arm {i}.")

        arm_value = _ensure_valid_state_array(arm_key, state_dict[arm_key], arm_dim)
        ee_value = _ensure_valid_state_array(ee_key, state_dict[ee_key], ee_dim)

        if arm_value.shape[:-1] != ee_value.shape[:-1]:
            raise ValueError(
                f"'{arm_key}' and '{ee_key}' must share the same prefix shape, "
                f"got {arm_value.shape[:-1]} and {ee_value.shape[:-1]}."
            )

        if expected_prefix_shape is None:
            expected_prefix_shape = arm_value.shape[:-1]
        elif arm_value.shape[:-1] != expected_prefix_shape:
            raise ValueError(
                "All state fields must share the same prefix shape. "
                f"Expected {expected_prefix_shape}, got {arm_value.shape[:-1]} for '{arm_key}'."
            )

        parts.append(np.concatenate([arm_value, ee_value], axis=-1))

    return np.concatenate(parts, axis=-1)


def unpack_robot_state(
    packed_state,
    action_type: str,
    robot_action_dim_info: dict,
    source_type: str = "obs",
):
    """
    Unpack packed robot state.

    Rules:
        - source_type='obs':
            * ndim must be <= 2
            * if ndim == 1: return dict
            * if ndim == 2: return list[dict]
        - source_type='dataset':
            * return dict of arrays directly

    Unpacking order:
        single-arm:
            [arm_0, ee_0]
        dual-arm:
            [arm_0, ee_0, arm_1, ee_1]
    """
    arm_dims, ee_dims, num_arms = _validate_config(action_type, robot_action_dim_info, source_type)
    arm_keys, ee_keys = _get_state_keys(action_type, num_arms, source_type)

    packed = np.asarray(packed_state)
    expected_dim = sum(arm_dims) + sum(ee_dims)

    if packed.shape[-1] != expected_dim:
        raise ValueError(f"packed_state last dim mismatch: expected {expected_dim}, got {packed.shape[-1]}.")

    if source_type == "obs":
        assert packed.ndim <= 2, f"When source_type='obs', packed_state.ndim must be <= 2, got {packed.ndim}."

        def _unpack_single_action(single_action: np.ndarray) -> dict:
            result = {}
            offset = 0

            for arm_key, ee_key, arm_dim, ee_dim in zip(arm_keys, ee_keys, arm_dims, ee_dims):
                result[arm_key] = single_action[offset : offset + arm_dim]
                offset += arm_dim

                result[ee_key] = single_action[offset : offset + ee_dim]
                offset += ee_dim

            return result

        if packed.ndim == 1:
            return _unpack_single_action(packed)

        return [_unpack_single_action(single_action) for single_action in packed]

    result = {}
    offset = 0

    for arm_key, ee_key, arm_dim, ee_dim in zip(arm_keys, ee_keys, arm_dims, ee_dims):
        result[arm_key] = packed[..., offset : offset + arm_dim]
        offset += arm_dim

        result[ee_key] = packed[..., offset : offset + ee_dim]
        offset += ee_dim

    return result


def get_robot_action_dim_info(env_cfg_type):
    return _load_robot_action_dim_info(str(env_cfg_type))


def get_batch_size(env_cfg_type):
    official_env, _ = _official_env_paths(str(env_cfg_type))
    if official_env.is_file():
        env_cfg = load_yaml(str(official_env))
        sim_cfg = env_cfg["config"]["sim"]
        sim_info = load_yaml(str(_XPOLICYLAB_ROOT / "env_cfg" / "sim" / f"{sim_cfg}.yml"))
        return sim_info["scene"]["num_envs"]

    robodojo_root = _find_robodojo_root()
    if robodojo_root is None:
        raise FileNotFoundError(f"Could not locate RoboDojo for environment {env_cfg_type!r}")
    env_cfg = load_yaml(str(robodojo_root / "configs" / "environment" / f"{env_cfg_type}.yml"))
    sim_cfg = env_cfg["config"]["sim"]
    sim_info = load_yaml(str(robodojo_root / "configs" / "sim" / f"{sim_cfg}.yml"))
    return sim_info["scene"]["num_envs"]


def get_action_dim(env_cfg_type):
    robot_action_dim_info = get_robot_action_dim_info(env_cfg_type)
    return sum(robot_action_dim_info["arm_dim"]) + sum(robot_action_dim_info["ee_dim"])


def decode_image_bit(image_bits):
    import cv2

    def _decode(single_image_bit):
        return cv2.imdecode(np.frombuffer(single_image_bit, np.uint8), cv2.IMREAD_COLOR)

    if isinstance(image_bits, np.ndarray) and image_bits.ndim == 1:
        return _decode(image_bits)

    if isinstance(image_bits, (list, tuple, np.ndarray)):
        images = [_decode(x) for x in image_bits]
        return np.array(images)
    else:
        return _decode(image_bits)
