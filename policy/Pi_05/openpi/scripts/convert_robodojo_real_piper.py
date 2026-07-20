"""Convert the pinned RoboDojo real PiPER demonstrations to LeRobot v3."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
import dataclasses
import json
from pathlib import Path
import shutil
from typing import Any

import cv2
import h5py
import numpy as np
from tqdm import tqdm

SOURCE_REPO_ID = "RoboDojo-Benchmark/RoboDojo"
SOURCE_REVISION = "1a3c4c334aef294c31d7a0190d8d6dff68df78e0"
OUTPUT_REPO_ID = "RoboDojo-real_piper_6task-bimanual_piper-joint"
FPS = 30
EXPECTED_EPISODES_PER_TASK = 100
EXPECTED_TOTAL_FRAMES = 539_737
TASKS = (
    "fill_pen_holder",
    "put_objects_into_basket",
    "stack_and_cover_blocks",
    "stack_bowls",
    "stand_up_bottles",
    "insert_charger",
)
CAMERA_ALIASES = {
    "cam_head": "cam_high",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}
JOINT_KEYS = (
    "left_arm_joint_states",
    "left_ee_joint_states",
    "right_arm_joint_states",
    "right_ee_joint_states",
)
MOTOR_NAMES = (
    *(f"left_joint_{index}" for index in range(6)),
    "left_gripper",
    *(f"right_joint_{index}" for index in range(6)),
    "right_gripper",
)


@dataclasses.dataclass(frozen=True)
class Episode:
    path: Path
    task: str
    instruction: str
    fps: int
    state: np.ndarray
    action: np.ndarray

    @property
    def num_frames(self) -> int:
        return self.state.shape[0]


def _read_scalar_text(dataset: h5py.Dataset, *, name: str) -> str:
    value = dataset[()]
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError(f"{name} must be scalar, got shape {value.shape}.")
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode("utf-8")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty UTF-8 string.")
    return value.strip()


def _joint_matrix(group: h5py.Group) -> np.ndarray:
    columns = []
    for key in JOINT_KEYS:
        if key not in group:
            raise KeyError(f"Missing {group.name}/{key}.")
        values = np.asarray(group[key][:], dtype=np.float32)
        expected_width = 6 if "arm" in key else 1
        if values.ndim == 1 and expected_width == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[1] != expected_width:
            raise ValueError(
                f"{group.name}/{key} must have shape (T, {expected_width}), got {values.shape}."
            )
        columns.append(values)
    lengths = {values.shape[0] for values in columns}
    if len(lengths) != 1:
        raise ValueError(f"Joint arrays in {group.name} have inconsistent lengths: {sorted(lengths)}.")
    return np.concatenate(columns, axis=1)


def inspect_episode(path: Path, task: str) -> Episode:
    with h5py.File(path, "r") as source:
        version = _read_scalar_text(source["data_format_version"], name="data_format_version")
        if version != "v1.0":
            raise ValueError(f"{path}: expected data_format_version v1.0, got {version!r}.")
        instruction = _read_scalar_text(source["instruction"], name="instruction")
        frequency = int(source["additional_info/frequency"][()])
        if frequency != FPS:
            raise ValueError(f"{path}: expected {FPS} Hz, got {frequency} Hz.")

        state = _joint_matrix(source["state"])
        action = _joint_matrix(source["action"])
        if state.shape != action.shape or state.shape[1] != 14:
            raise ValueError(f"{path}: state/action shapes must both be (T, 14), got {state.shape}/{action.shape}.")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"{path}: state/action contains NaN or infinity.")
        if state.shape[0] < 2:
            raise ValueError(f"{path}: episode must contain at least two frames.")
        if not np.allclose(action[:-1], state[1:], rtol=0.0, atol=1e-6):
            raise ValueError(f"{path}: action[t] does not equal state[t+1].")
        if not np.allclose(action[-1], state[-1], rtol=0.0, atol=1e-6):
            raise ValueError(f"{path}: final action does not repeat the final state.")

        for source_name in CAMERA_ALIASES:
            colors = source[f"vision/{source_name}/colors"]
            shape = tuple(int(value) for value in source[f"vision/{source_name}/shape"][:])
            if len(colors) != state.shape[0]:
                raise ValueError(f"{path}: {source_name} has {len(colors)} frames, expected {state.shape[0]}.")
            if shape != (480, 640, 3):
                raise ValueError(f"{path}: {source_name} declares shape {shape}, expected (480, 640, 3).")

    return Episode(path=path, task=task, instruction=instruction, fps=frequency, state=state, action=action)


def decode_rgb(payload: bytes | np.bytes_ | np.ndarray) -> np.ndarray:
    if isinstance(payload, np.ndarray):
        if payload.shape != ():
            raise TypeError(f"Compressed image payload must be scalar, got shape {payload.shape}.")
        payload = payload.item()
    if not isinstance(payload, (bytes, np.bytes_)):
        raise TypeError(f"Unsupported compressed image payload type: {type(payload)!r}.")
    encoded = np.frombuffer(bytes(payload).rstrip(b"\0"), dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("OpenCV could not decode the compressed image.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
        raise ValueError(f"Decoded image must be uint8 HWC (480, 640, 3), got {rgb.dtype} {rgb.shape}.")
    return rgb


def iter_frames(episode: Episode) -> Iterator[dict[str, Any]]:
    with h5py.File(episode.path, "r") as source:
        cameras = {name: source[f"vision/{name}/colors"] for name in CAMERA_ALIASES}
        for index in range(episode.num_frames):
            frame: dict[str, Any] = {
                "observation.state": episode.state[index],
                "action": episode.action[index],
                "task": episode.instruction,
            }
            for source_name, output_name in CAMERA_ALIASES.items():
                frame[f"observation.images.{output_name}"] = decode_rgb(cameras[source_name][index])
            yield frame


def discover_episode_files(raw_root: Path, tasks: Sequence[str] = TASKS) -> list[tuple[str, Path]]:
    discovered = []
    for task in tasks:
        data_dir = raw_root / "data" / "RoboDojo_real" / task / "piper" / "data"
        files = sorted(data_dir.glob("episode_*.hdf5"))
        if len(files) != EXPECTED_EPISODES_PER_TASK:
            raise ValueError(
                f"{task}: expected {EXPECTED_EPISODES_PER_TASK} episodes in {data_dir}, found {len(files)}."
            )
        discovered.extend((task, path) for path in files)
    return discovered


def download_source(raw_root: Path) -> None:
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    allow_patterns = [f"data/RoboDojo_real/{task}/piper/data/*.hdf5" for task in TASKS]
    snapshot_download(
        repo_id=SOURCE_REPO_ID,
        repo_type="dataset",
        revision=SOURCE_REVISION,
        local_dir=raw_root,
        allow_patterns=allow_patterns,
    )


def create_dataset(repo_id: str, root: Path, *, encoder_threads: int | None):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415
    except ModuleNotFoundError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    features = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": list(MOTOR_NAMES)},
        "action": {"dtype": "float32", "shape": (14,), "names": list(MOTOR_NAMES)},
    }
    for camera_name in CAMERA_ALIASES.values():
        features[f"observation.images.{camera_name}"] = {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        }
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        robot_type="bimanual_piper",
        fps=FPS,
        features=features,
        use_videos=True,
        vcodec="h264",
        streaming_encoding=True,
        encoder_threads=encoder_threads,
    )


def validate_converted_dataset(repo_id: str, dataset_root: Path) -> dict[str, int]:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415
    except ModuleNotFoundError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root)
    if dataset.fps != FPS:
        raise ValueError(f"Converted dataset fps is {dataset.fps}, expected {FPS}.")
    if dataset.num_episodes != len(TASKS) * EXPECTED_EPISODES_PER_TASK:
        raise ValueError(f"Converted dataset has {dataset.num_episodes} episodes, expected 600.")
    if dataset.num_frames != EXPECTED_TOTAL_FRAMES:
        raise ValueError(f"Converted dataset has {dataset.num_frames} frames, expected {EXPECTED_TOTAL_FRAMES}.")
    sample = dataset[0]
    for key in CAMERA_ALIASES.values():
        image = np.asarray(sample[f"observation.images.{key}"])
        if image.shape != (3, 480, 640):
            raise ValueError(f"Reloaded {key} image has shape {image.shape}, expected CHW (3, 480, 640).")
    for key in ("observation.state", "action"):
        values = np.asarray(sample[key])
        if values.shape[-1] != 14 or not np.isfinite(values).all():
            raise ValueError(f"Reloaded {key} is invalid: shape={values.shape}.")
    return {"episodes": dataset.num_episodes, "frames": dataset.num_frames, "fps": dataset.fps}


def convert(args: argparse.Namespace) -> Path:
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    dataset_root = output_root / args.repo_id
    if not args.skip_download:
        download_source(raw_root)
    episode_files = discover_episode_files(raw_root)

    if dataset_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {dataset_root}. Pass --overwrite to replace it.")
        shutil.rmtree(dataset_root)
    dataset_root.parent.mkdir(parents=True, exist_ok=True)

    dataset = create_dataset(args.repo_id, dataset_root, encoder_threads=args.encoder_threads)
    task_counts = dict.fromkeys(TASKS, 0)
    total_frames = 0
    for task, path in tqdm(episode_files, desc="Converting PiPER episodes", unit="episode"):
        episode = inspect_episode(path, task)
        for frame in iter_frames(episode):
            dataset.add_frame(frame)
        dataset.save_episode()
        task_counts[task] += 1
        total_frames += episode.num_frames
    dataset.finalize()

    if total_frames != EXPECTED_TOTAL_FRAMES:
        raise ValueError(f"Source contained {total_frames} frames, expected {EXPECTED_TOTAL_FRAMES}.")
    validation = validate_converted_dataset(args.repo_id, dataset_root)
    manifest = {
        "source_repo_id": SOURCE_REPO_ID,
        "source_revision": SOURCE_REVISION,
        "repo_id": args.repo_id,
        "tasks": task_counts,
        **validation,
    }
    manifest_path = dataset_root / "robodojo_real_piper_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", default=OUTPUT_REPO_ID)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--encoder-threads", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    print(convert(parse_args()))
