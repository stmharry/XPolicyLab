"""Profile-driven LeRobot PI0.5 inference for pinned embodiments."""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
import logging
from pathlib import Path
from typing import Any

import cv2
from draccus.parsers.decoding import decode
import numpy as np
from openpi.shared import download as _download
from safetensors.torch import load_file
import torch

from XPolicyLab.policy.Pi_05.contract import (
    PIPER_PROFILE_NAME,
    PIPER_ROBOT_TYPE,
    PIPER_TASK_PROMPT,
    YAM_PICKUP_PROFILE_NAME,
    YAM_PICKUP_TASK_PROMPT,
)

_TOKENIZER_URI = "gs://big_vision/paligemma_tokenizer.model"
_LEGACY_VISION_TOWER_PREFIX = "model.paligemma_with_expert.paligemma.model.vision_tower."
_YAM_PICKUP_SOURCE_ASPECT = 16 / 9
_YAM_ACTION_NAMES = (
    "left_shoulder_pan.pos",
    "left_shoulder_lift.pos",
    "left_elbow_flex.pos",
    "left_wrist_flex.pos",
    "left_wrist_tilt.pos",
    "left_wrist_roll.pos",
    "left_gripper.pos",
    "right_shoulder_pan.pos",
    "right_shoulder_lift.pos",
    "right_elbow_flex.pos",
    "right_wrist_flex.pos",
    "right_wrist_tilt.pos",
    "right_wrist_roll.pos",
    "right_gripper.pos",
)


@dataclass(frozen=True)
class LeRobotProfile:
    name: str
    camera_mapping: dict[str, str]
    prompt: str
    robot_type: str
    action_feature_names: tuple[str, ...]
    camera_shape: tuple[int, int, int] | None
    source_aspect: float | None


_PROFILES = {
    YAM_PICKUP_PROFILE_NAME: LeRobotProfile(
        name=YAM_PICKUP_PROFILE_NAME,
        camera_mapping={
            "cam_high": "observation.images.head",
            "cam_left_wrist": "observation.images.wrist_left",
            "cam_right_wrist": "observation.images.wrist_right",
        },
        prompt=YAM_PICKUP_TASK_PROMPT,
        robot_type="yam_follower_bimanual",
        action_feature_names=_YAM_ACTION_NAMES,
        camera_shape=None,
        source_aspect=_YAM_PICKUP_SOURCE_ASPECT,
    ),
    PIPER_PROFILE_NAME: LeRobotProfile(
        name=PIPER_PROFILE_NAME,
        camera_mapping={
            "cam_high": "observation.images.cam_front",
            "cam_left_wrist": "observation.images.cam_left_wrist",
            "cam_right_wrist": "observation.images.cam_right_wrist",
        },
        prompt=PIPER_TASK_PROMPT,
        robot_type=PIPER_ROBOT_TYPE,
        action_feature_names=("motors",),
        camera_shape=(3, 224, 224),
        source_aspect=None,
    ),
}


def _profile(profile_name: str) -> LeRobotProfile:
    try:
        return _PROFILES[profile_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported LeRobot PI0.5 profile: {profile_name}") from exc


def _load_config(checkpoint_root: Path, profile: LeRobotProfile):
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    raw = json.loads((checkpoint_root / "config.json").read_text(encoding="utf-8"))
    if raw.get("use_relative_actions") is not False:
        raise ValueError(f"Pinned {profile.name} checkpoint must use absolute actions.")
    if tuple(raw.get("action_feature_names", ())) != profile.action_feature_names:
        raise ValueError(f"Pinned {profile.name} checkpoint action channels do not match its profile.")
    if raw.get("chunk_size") != 50 or raw.get("n_action_steps") != 50:
        raise ValueError(f"Pinned {profile.name} checkpoint must predict 50 absolute actions.")
    if (raw.get("input_features") or {}).get("observation.state", {}).get("shape") != [14]:
        raise ValueError(f"Pinned {profile.name} checkpoint must accept 14D state.")
    if (raw.get("output_features") or {}).get("action", {}).get("shape") != [14]:
        raise ValueError(f"Pinned {profile.name} checkpoint must emit 14D action.")
    for camera_key in profile.camera_mapping.values():
        shape = (raw.get("input_features") or {}).get(camera_key, {}).get("shape")
        if not isinstance(shape, list) or len(shape) != 3 or shape[0] != 3:
            raise ValueError(f"Pinned {profile.name} checkpoint has invalid camera feature {camera_key}: {shape!r}.")
        if profile.camera_shape is not None and tuple(shape) != profile.camera_shape:
            raise ValueError(
                f"Pinned {profile.name} checkpoint camera {camera_key} must have shape "
                f"{profile.camera_shape}; got {shape!r}."
            )

    valid_fields = {field.name for field in fields(PI05Config)}
    compatible = {key: value for key, value in raw.items() if key in valid_fields}
    config = decode(PI05Config, compatible)
    config.device = "cuda"
    return config


def _load_dataset_stats(path: Path) -> dict[str, dict[str, torch.Tensor]]:
    nested: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in load_file(path).items():
        feature, statistic = key.rsplit(".", 1)
        nested.setdefault(feature, {})[statistic] = value
    return nested


def _normalizer_path(checkpoint_root: Path) -> Path:
    matches = tuple(checkpoint_root.glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one LeRobot preprocessor normalizer; found {len(matches)}.")
    return matches[0]


def _remap_checkpoint_key(key: str) -> str:
    """Map the legacy SigLIP tower layout to the locked Transformers layout."""

    if key.startswith(_LEGACY_VISION_TOWER_PREFIX):
        suffix = key.removeprefix(_LEGACY_VISION_TOWER_PREFIX)
        if not suffix.startswith("vision_model."):
            return f"{_LEGACY_VISION_TOWER_PREFIX}vision_model.{suffix}"
    return key


def _load_policy(policy_class, config, checkpoint_root: Path):
    policy = policy_class(config)
    state_dict = load_file(checkpoint_root / "model.safetensors")
    remapped = {_remap_checkpoint_key(key): value for key, value in state_dict.items()}
    if len(remapped) != len(state_dict):
        raise ValueError("LeRobot PI0.5 vision-key remapping produced duplicate parameters.")
    missing, unexpected = policy.load_state_dict(remapped, strict=False)
    if missing or unexpected:
        raise ValueError(
            "LeRobot PI0.5 checkpoint does not match the locked runtime after vision-key remapping: "
            f"missing={len(missing)}, unexpected={len(unexpected)}."
        )
    logging.getLogger(__name__).info("Loaded LeRobot PI0.5 checkpoint with exact key coverage.")
    return policy.eval()


def _validate_saved_processors(checkpoint_root: Path) -> None:
    expected_disabled_steps = {
        "policy_preprocessor.json": "delta_actions_processor",
        "policy_postprocessor.json": "absolute_actions_processor",
    }
    for filename, registry_name in expected_disabled_steps.items():
        payload = json.loads((checkpoint_root / filename).read_text(encoding="utf-8"))
        matching = [step for step in payload.get("steps", ()) if step.get("registry_name") == registry_name]
        # Newer LeRobot serializers omit disabled action-conversion steps;
        # older releases retained them with enabled=false.  Both represent the
        # absolute-action contract pinned in config.json.
        if matching and (len(matching) != 1 or matching[0].get("config", {}).get("enabled") is not False):
            raise ValueError(f"{filename} must declare disabled {registry_name}.")


def _make_processors(config, checkpoint_root: Path):
    from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
    from lerobot.processor import (
        AddBatchDimensionProcessorStep,
        DeviceProcessorStep,
        NormalizerProcessorStep,
        PolicyAction,
        PolicyProcessorPipeline,
        RenameObservationsProcessorStep,
        TokenizerProcessorStep,
        UnnormalizerProcessorStep,
    )
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
    from transformers import GemmaTokenizer

    _validate_saved_processors(checkpoint_root)
    stats = _load_dataset_stats(_normalizer_path(checkpoint_root))
    tokenizer_path = _download.maybe_download(_TOKENIZER_URI, gs={"token": "anon"})
    tokenizer = GemmaTokenizer(vocab_file=str(tokenizer_path))

    preprocessor = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=[
            RenameObservationsProcessorStep(rename_map={}),
            AddBatchDimensionProcessorStep(),
            NormalizerProcessorStep(
                features={**config.input_features, **config.output_features},
                norm_map=config.normalization_mapping,
                stats=stats,
            ),
            Pi05PrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim),
            TokenizerProcessorStep(
                tokenizer=tokenizer,
                max_length=config.tokenizer_max_length,
                padding_side="right",
                padding="max_length",
                truncation=True,
            ),
            DeviceProcessorStep(device=config.device),
        ],
        name="policy_preprocessor",
    )
    postprocessor = PolicyProcessorPipeline[PolicyAction, PolicyAction](
        steps=[
            UnnormalizerProcessorStep(
                features=config.output_features,
                norm_map=config.normalization_mapping,
                stats=stats,
            ),
            DeviceProcessorStep(device="cpu"),
        ],
        name="policy_postprocessor",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor


def _resize_chw(
    image: Any,
    *,
    height: int,
    width: int,
    source_aspect: float | None = _YAM_PICKUP_SOURCE_ASPECT,
) -> np.ndarray:
    chw = np.asarray(image, dtype=np.uint8)
    if chw.ndim != 3 or chw.shape[0] != 3:
        raise ValueError(f"LeRobot image must be uint8 CHW, got {chw.shape}.")
    hwc = np.transpose(chw, (1, 2, 0))
    source_height, source_width = hwc.shape[:2]

    if source_aspect is None:
        if (source_height, source_width) == (height, width):
            return hwc.copy()
        return cv2.resize(hwc, (width, height), interpolation=cv2.INTER_AREA)

    # Generic pickup demonstrations used 640x360 YAM streams before dataset
    # standardization. Form that source view from Moonlake's 640x480 cameras;
    # otherwise direct 4:3 letterboxing produces unseen side bars and shrinks
    # every scene feature relative to training.
    actual_aspect = source_width / source_height
    if actual_aspect < source_aspect:
        crop_height = min(source_height, round(source_width / source_aspect))
        top = (source_height - crop_height) // 2
        hwc = hwc[top : top + crop_height]
    elif actual_aspect > source_aspect:
        crop_width = min(source_width, round(source_height * source_aspect))
        left = (source_width - crop_width) // 2
        hwc = hwc[:, left : left + crop_width]

    source_height, source_width = hwc.shape[:2]
    scale = min(width / source_width, height / source_height)
    # The dataset's FFmpeg standardizer rounds half pixels up (360x202.5 to
    # 360x203), unlike Python's banker rounding.
    resized_width = min(width, int(source_width * scale + 0.5))
    resized_height = min(height, int(source_height * scale + 0.5))
    resized = cv2.resize(hwc, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    # Reproduce the dataset's 640x360-to-360x240 standardization without
    # distorting object geometry.
    result = np.zeros((height, width, 3), dtype=np.uint8)
    top = (height - resized_height) // 2
    left = (width - resized_width) // 2
    result[top : top + resized_height, left : left + resized_width] = resized
    return result


class LeRobotPi05Policy:
    """Expose a pinned LeRobot PI0.5 checkpoint through OpenPI's infer API."""

    def __init__(self, checkpoint_root: Path, *, seed: int, profile_name: str = YAM_PICKUP_PROFILE_NAME):
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        self.profile = _profile(profile_name)
        self.config = _load_config(checkpoint_root, self.profile)
        self.policy = _load_policy(PI05Policy, self.config, checkpoint_root)
        self.preprocessor, self.postprocessor = _make_processors(self.config, checkpoint_root)
        self.seed = int(seed)
        self._generator = torch.Generator(device=self.config.device)
        self.reset()

    def reset(self) -> None:
        self.policy.reset()
        self._generator.manual_seed(self.seed)

    def infer(self, observation: dict[str, Any], **_: Any) -> dict[str, np.ndarray]:
        state = np.asarray(observation["state"], dtype=np.float32)
        if state.shape != (14,):
            raise ValueError(f"LeRobot state must have shape (14,), got {state.shape}.")

        batch: dict[str, Any] = {
            "observation.state": torch.from_numpy(state.copy()),
            # Use the checkpoint's exact generic pick-and-place vocabulary;
            # unlike the fruit-bag task, it does not imply a long placement
            # sweep through receptacle clutter after approaching the object.
            "task": self.profile.prompt,
            "robot_type": self.profile.robot_type,
        }
        for source_key, checkpoint_key in self.profile.camera_mapping.items():
            feature = self.config.input_features[checkpoint_key]
            _, height, width = feature.shape
            resized = _resize_chw(
                observation["images"][source_key],
                height=height,
                width=width,
                source_aspect=self.profile.source_aspect,
            )
            batch[checkpoint_key] = torch.from_numpy(resized).permute(2, 0, 1).float().div_(255.0)

        prepared = self.preprocessor(batch)
        noise = torch.randn(
            (1, self.config.chunk_size, self.config.max_action_dim),
            generator=self._generator,
            device=self.config.device,
        )
        with torch.inference_mode():
            actions = self.policy.predict_action_chunk(prepared, noise=noise)
        actions = self.postprocessor(actions).detach().cpu().numpy()[0].astype(np.float32, copy=False)
        return {"actions": actions}
