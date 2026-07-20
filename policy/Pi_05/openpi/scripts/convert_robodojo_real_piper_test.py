from __future__ import annotations

from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

from . import convert_robodojo_real_piper as converter


def _encoded_bgr(color: tuple[int, int, int]) -> bytes:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:] = color
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _write_episode(path: Path, *, frequency: int = 30, instruction: bytes = b"Stack the bowls.") -> None:
    frames = 3
    with h5py.File(path, "w") as target:
        target.create_dataset("data_format_version", data=b"v1.0")
        target.create_dataset("instruction", data=instruction)
        target.create_dataset("additional_info/frequency", data=frequency)
        state_group = target.create_group("state")
        action_group = target.create_group("action")
        for side, offset in (("left", 0.0), ("right", 10.0)):
            arm = np.arange(frames * 6, dtype=np.float32).reshape(frames, 6) + offset
            gripper = (np.arange(frames, dtype=np.float32) + offset)[:, None]
            state_group.create_dataset(f"{side}_arm_joint_states", data=arm)
            state_group.create_dataset(f"{side}_ee_joint_states", data=gripper)
            action_group.create_dataset(
                f"{side}_arm_joint_states", data=np.concatenate([arm[1:], arm[-1:]], axis=0)
            )
            action_group.create_dataset(
                f"{side}_ee_joint_states", data=np.concatenate([gripper[1:], gripper[-1:]], axis=0)
            )
        for camera, color in zip(converter.CAMERA_ALIASES, ((1, 2, 250), (4, 240, 6), (230, 8, 9)), strict=True):
            group = target.create_group(f"vision/{camera}")
            payload = _encoded_bgr(color)
            group.create_dataset("colors", data=np.asarray([payload] * frames, dtype=f"S{len(payload)}"))
            group.create_dataset("shape", data=np.asarray([480, 640, 3]))


def test_inspect_episode_preserves_scalar_instruction_and_action_alignment(tmp_path: Path):
    path = tmp_path / "episode_0000000.hdf5"
    _write_episode(path)

    episode = converter.inspect_episode(path, "stack_bowls")

    assert episode.instruction == "Stack the bowls."
    assert episode.fps == 30
    assert episode.state.shape == (3, 14)
    assert episode.action.shape == (3, 14)
    np.testing.assert_allclose(episode.action[:-1], episode.state[1:])
    np.testing.assert_allclose(episode.action[-1], episode.state[-1])
    assert episode.state[0, 6] == 0.0
    assert episode.state[0, 13] == 10.0


def test_iter_frames_decodes_rgb_hwc_and_keeps_camera_order(tmp_path: Path):
    path = tmp_path / "episode_0000000.hdf5"
    _write_episode(path)
    episode = converter.inspect_episode(path, "stack_bowls")

    frame = next(converter.iter_frames(episode))

    assert frame["task"] == "Stack the bowls."
    assert [key for key in frame if key.startswith("observation.images")] == [
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ]
    head = frame["observation.images.cam_high"]
    assert head.shape == (480, 640, 3)
    assert head.dtype == np.uint8
    assert int(head[0, 0, 0]) > 240
    assert int(head[0, 0, 2]) < 10


@pytest.mark.parametrize("frequency", [25, 50])
def test_inspect_episode_rejects_non_30_hz(tmp_path: Path, frequency: int):
    path = tmp_path / "episode_0000000.hdf5"
    _write_episode(path, frequency=frequency)

    with pytest.raises(ValueError, match="expected 30 Hz"):
        converter.inspect_episode(path, "stack_bowls")


def test_inspect_episode_fails_on_misaligned_action(tmp_path: Path):
    path = tmp_path / "episode_0000000.hdf5"
    _write_episode(path)
    with h5py.File(path, "r+") as target:
        target["action/left_arm_joint_states"][0, 0] += 1.0

    with pytest.raises(ValueError, match=r"action\[t\]"):
        converter.inspect_episode(path, "stack_bowls")


def test_discover_episode_files_uses_declared_task_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(converter, "EXPECTED_EPISODES_PER_TASK", 2)
    for task in converter.TASKS:
        data_dir = tmp_path / "data" / "RoboDojo_real" / task / "piper" / "data"
        data_dir.mkdir(parents=True)
        for index in (1, 0):
            (data_dir / f"episode_{index:07d}.hdf5").touch()

    discovered = converter.discover_episode_files(tmp_path)

    assert [task for task, _ in discovered] == [task for task in converter.TASKS for _ in range(2)]
    assert [path.name for _, path in discovered[:2]] == ["episode_0000000.hdf5", "episode_0000001.hdf5"]
