from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

from . import convert_robodojo_real_arx_x5 as converter


def _encoded_bgr(color: tuple[int, int, int]) -> np.ndarray:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:] = color
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded


def _padded_rows(payload: np.ndarray, frames: int) -> np.ndarray:
    rows = np.zeros((frames, payload.size + 17), dtype=np.uint8)
    rows[:, : payload.size] = payload
    return rows


def _write_episode(
    path: Path,
    *,
    frequency: int = 30,
    instruction: bytes | np.ndarray = b"Cover the blocks.",
) -> None:
    frames = 3
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as target:
        target.create_dataset("data_format_version", data=b"v1.0")
        target.create_dataset("instruction", data=instruction)
        target.create_dataset("additional_info/frequency", data=frequency)
        state_group = target.create_group("state")
        action_group = target.create_group("action")
        for side, offset in (("left", 0.0), ("right", 10.0)):
            arm = np.arange(frames * 6, dtype=np.float32).reshape(frames, 6) + offset
            gripper = np.linspace(0.0, 1.0, frames, dtype=np.float32)[:, None]
            state_group.create_dataset(f"{side}_arm_joint_states", data=arm)
            state_group.create_dataset(f"{side}_ee_joint_states", data=gripper)
            action_group.create_dataset(f"{side}_arm_joint_states", data=np.concatenate([arm[1:], arm[-1:]], axis=0))
            action_group.create_dataset(
                f"{side}_ee_joint_states", data=np.concatenate([gripper[1:], gripper[-1:]], axis=0)
            )
        colors = ((1, 2, 250), (4, 240, 6), (230, 8, 9))
        for camera, color in zip(converter.CAMERA_ALIASES, colors, strict=True):
            group = target.create_group(f"vision/{camera}")
            group.create_dataset("colors", data=_padded_rows(_encoded_bgr(color), frames))
            group.create_dataset("shape", data=np.asarray([480, 640, 3]))


def _write_corpus(root: Path, *, episodes_per_task: int = 1) -> None:
    for task in converter.TASKS:
        for index in range(episodes_per_task):
            _write_episode(root / "data" / "RoboDojo_real" / task / "arx_x5" / "data" / f"episode_{index:07d}.hdf5")


def test_inspect_episode_preserves_scalar_instruction_gripper_shape_and_alignment(tmp_path: Path):
    path = tmp_path / "episode_0000000.hdf5"
    _write_episode(path)

    episode = converter.inspect_episode(path, "cover_blocks")

    assert episode.instruction == "Cover the blocks."
    assert episode.fps == 30
    assert episode.state.shape == (3, 14)
    assert episode.action.shape == (3, 14)
    np.testing.assert_allclose(episode.action[:-1], episode.state[1:])
    np.testing.assert_allclose(episode.action[-1], episode.state[-1])
    np.testing.assert_allclose(episode.state[:, 6], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(episode.state[:, 13], [0.0, 0.5, 1.0])


def test_iter_frames_decodes_padded_bgr_as_rgb_hwc_and_keeps_camera_order(tmp_path: Path):
    path = tmp_path / "episode_0000000.hdf5"
    _write_episode(path)
    episode = converter.inspect_episode(path, "cover_blocks")

    frame = next(converter.iter_frames(episode))

    assert frame["task"] == "Cover the blocks."
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
        converter.inspect_episode(path, "cover_blocks")


def test_inspect_episode_rejects_non_scalar_instruction(tmp_path: Path):
    path = tmp_path / "episode_0000000.hdf5"
    _write_episode(path, instruction=np.asarray([b"one", b"two"]))

    with pytest.raises(ValueError, match="instruction must be scalar"):
        converter.inspect_episode(path, "cover_blocks")


def test_inspect_episode_fails_on_misaligned_action(tmp_path: Path):
    path = tmp_path / "episode_0000000.hdf5"
    _write_episode(path)
    with h5py.File(path, "r+") as target:
        target["action/left_arm_joint_states"][0, 0] += 1.0

    with pytest.raises(ValueError, match=r"action\[t\]"):
        converter.inspect_episode(path, "cover_blocks")


def test_inspect_episode_fails_on_unnormalized_gripper(tmp_path: Path):
    path = tmp_path / "episode_0000000.hdf5"
    _write_episode(path)
    with h5py.File(path, "r+") as target:
        target["state/right_ee_joint_states"][1, 0] = 2.0
        target["action/right_ee_joint_states"][0, 0] = 2.0

    with pytest.raises(ValueError, match=r"grippers must be normalized"):
        converter.inspect_episode(path, "cover_blocks")


def test_discover_episode_files_uses_declared_task_and_index_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(converter, "EXPECTED_EPISODES_PER_TASK", 2)
    for task in converter.TASKS:
        data_dir = tmp_path / "data" / "RoboDojo_real" / task / "arx_x5" / "data"
        data_dir.mkdir(parents=True)
        for index in (1, 0):
            (data_dir / f"episode_{index:07d}.hdf5").touch()

    discovered = converter.discover_episode_files(tmp_path)

    assert [task for task, _ in discovered] == [task for task in converter.TASKS for _ in range(2)]
    assert [path.name for _, path in discovered[:2]] == ["episode_0000000.hdf5", "episode_0000001.hdf5"]


def test_preflight_derives_and_locks_exact_frame_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(converter, "EXPECTED_EPISODES_PER_TASK", 1)
    _write_corpus(tmp_path)

    preflight = converter.run_preflight(tmp_path)

    assert preflight.total_frames == len(converter.TASKS) * 3
    manifest_path = tmp_path / converter.PREFLIGHT_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    assert manifest["frames"] == preflight.total_frames
    manifest["frames"] += 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="immutable gate"):
        converter.run_preflight(tmp_path)


def test_convert_preflights_before_overwriting_nonempty_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(converter, "EXPECTED_EPISODES_PER_TASK", 1)
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "output"
    _write_corpus(raw_root)
    broken = raw_root / "data" / "RoboDojo_real" / converter.TASKS[0] / "arx_x5" / "data" / "episode_0000000.hdf5"
    with h5py.File(broken, "r+") as target:
        target["action/left_arm_joint_states"][0, 0] += 1.0
    dataset_root = output_root / converter.OUTPUT_REPO_ID
    dataset_root.mkdir(parents=True)
    marker = dataset_root / "do-not-delete"
    marker.write_text("preserve")
    args = argparse.Namespace(
        raw_root=raw_root,
        output_root=output_root,
        repo_id=converter.OUTPUT_REPO_ID,
        validate_only=False,
        skip_download=True,
        overwrite=True,
        encoder_threads=1,
    )

    with pytest.raises(ValueError, match=r"action\[t\]"):
        converter.convert(args)
    assert marker.read_text() == "preserve"
