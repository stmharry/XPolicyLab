"""Convert the pinned RoboDojo real ARX X5 demonstrations to LeRobot v3."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
import dataclasses
import json
from pathlib import Path
import shutil
import time
from typing import Any

import cv2
import h5py
import numpy as np
from tqdm import tqdm

try:
    from .convert_robodojo_real_piper import MOTOR_NAMES
    from .convert_robodojo_real_piper import Episode
    from .convert_robodojo_real_piper import _joint_matrix
    from .convert_robodojo_real_piper import _read_scalar_text
except ImportError:  # Direct script execution.
    from convert_robodojo_real_piper import MOTOR_NAMES
    from convert_robodojo_real_piper import Episode
    from convert_robodojo_real_piper import _joint_matrix
    from convert_robodojo_real_piper import _read_scalar_text


SOURCE_REPO_ID = "RoboDojo-Benchmark/RoboDojo"
SOURCE_REVISION = "1a3c4c334aef294c31d7a0190d8d6dff68df78e0"
OUTPUT_REPO_ID = "RoboDojo-real_arx_x5_6task-bimanual_arx_x5-joint"
FPS = 30
EXPECTED_EPISODES_PER_TASK = 100
TASKS = (
    "cover_blocks",
    "insert_tubes",
    "make_bread",
    "make_food",
    "pack_and_pour_fruit",
    "store_in_safe",
)
CAMERA_ALIASES = {
    "cam_head": "cam_high",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}
PREFLIGHT_MANIFEST = "robodojo_real_arx_x5_preflight.json"
DATASET_MANIFEST = "robodojo_real_arx_x5_manifest.json"


@dataclasses.dataclass(frozen=True)
class Preflight:
    episode_files: tuple[tuple[str, Path], ...]
    task_episodes: dict[str, int]
    task_frames: dict[str, int]
    total_frames: int

    def as_manifest(self) -> dict[str, Any]:
        return {
            "source_repo_id": SOURCE_REPO_ID,
            "source_revision": SOURCE_REVISION,
            "fps": FPS,
            "episodes": len(self.episode_files),
            "frames": self.total_frames,
            "task_episodes": self.task_episodes,
            "task_frames": self.task_frames,
        }


def decode_rgb(payload: bytes | np.bytes_ | np.ndarray) -> np.ndarray:
    """Decode scalar bytes or ARX's zero-padded one-dimensional uint8 rows."""
    if isinstance(payload, np.ndarray):
        if payload.shape == ():
            payload = payload.item()
        elif payload.ndim == 1 and payload.dtype == np.uint8:
            payload = np.ascontiguousarray(payload).tobytes()
        else:
            raise TypeError(
                f"Compressed image payload must be scalar bytes or a 1D uint8 row, got {payload.dtype} {payload.shape}."
            )
    if not isinstance(payload, bytes | np.bytes_):
        raise TypeError(f"Unsupported compressed image payload type: {type(payload)!r}.")
    encoded = np.frombuffer(bytes(payload).rstrip(b"\0"), dtype=np.uint8)
    if encoded.size == 0:
        raise ValueError("Compressed image payload is empty after removing padding.")
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("OpenCV could not decode the compressed image.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
        raise ValueError(f"Decoded image must be uint8 HWC (480, 640, 3), got {rgb.dtype} {rgb.shape}.")
    return rgb


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
        if state.shape[0] < 2:
            raise ValueError(f"{path}: episode must contain at least two frames.")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"{path}: state/action contains NaN or infinity.")
        for name, values in (("state", state), ("action", action)):
            grippers = values[:, (6, 13)]
            if np.any(grippers < -1e-6) or np.any(grippers > 1.0 + 1e-6):
                raise ValueError(f"{path}: {name} grippers must be normalized to [0, 1].")
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
            decode_rgb(colors[0])
            decode_rgb(colors[-1])

    return Episode(path=path, task=task, instruction=instruction, fps=frequency, state=state, action=action)


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
    discovered: list[tuple[str, Path]] = []
    expected_names = [f"episode_{index:07d}.hdf5" for index in range(EXPECTED_EPISODES_PER_TASK)]
    for task in tasks:
        data_dir = raw_root / "data" / "RoboDojo_real" / task / "arx_x5" / "data"
        files = sorted(data_dir.glob("episode_*.hdf5"))
        names = [path.name for path in files]
        if names != expected_names:
            missing = sorted(set(expected_names) - set(names))
            extra = sorted(set(names) - set(expected_names))
            raise ValueError(
                f"{task}: expected exactly episodes 0..{EXPECTED_EPISODES_PER_TASK - 1} in {data_dir}; "
                f"found {len(files)}, missing={missing[:5]}, extra={extra[:5]}."
            )
        discovered.extend((task, path) for path in files)
    return discovered


def run_preflight(raw_root: Path) -> Preflight:
    episode_files = tuple(discover_episode_files(raw_root))
    task_episodes = dict.fromkeys(TASKS, 0)
    task_frames = dict.fromkeys(TASKS, 0)
    total_frames = 0
    for task, path in tqdm(episode_files, desc="Preflighting ARX X5 episodes", unit="episode"):
        episode = inspect_episode(path, task)
        task_episodes[task] += 1
        task_frames[task] += episode.num_frames
        total_frames += episode.num_frames
    preflight = Preflight(episode_files, task_episodes, task_frames, total_frames)

    manifest_path = raw_root / PREFLIGHT_MANIFEST
    candidate = preflight.as_manifest()
    if manifest_path.exists():
        recorded = json.loads(manifest_path.read_text())
        if recorded != candidate:
            raise ValueError(
                f"Raw preflight no longer matches immutable gate {manifest_path}: "
                f"recorded frames={recorded.get('frames')}, current frames={total_frames}."
            )
    else:
        manifest_path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    return preflight


def download_source(raw_root: Path) -> None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    allow_patterns = [f"data/RoboDojo_real/{task}/arx_x5/data/*.hdf5" for task in TASKS]
    for attempt in range(6):
        try:
            snapshot_download(
                repo_id=SOURCE_REPO_ID,
                repo_type="dataset",
                revision=SOURCE_REVISION,
                local_dir=raw_root,
                allow_patterns=allow_patterns,
                max_workers=4,
            )
            return
        except HfHubHTTPError as exc:
            if exc.response is None or exc.response.status_code != 429 or attempt == 5:
                raise
            delay_s = min(60 * 2**attempt, 600)
            print(f"Hugging Face rate limit hit; retrying in {delay_s}s (attempt {attempt + 2}/6).")
            time.sleep(delay_s)


def create_dataset(repo_id: str, root: Path, *, encoder_threads: int | None):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

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
        robot_type="bimanual_arx_x5",
        fps=FPS,
        features=features,
        use_videos=True,
        vcodec="h264",
        streaming_encoding=True,
        encoder_threads=encoder_threads,
    )


def validate_converted_dataset(repo_id: str, dataset_root: Path, *, expected_frames: int) -> dict[str, int]:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root, video_backend="pyav")
    expected_episodes = len(TASKS) * EXPECTED_EPISODES_PER_TASK
    if dataset.fps != FPS:
        raise ValueError(f"Converted dataset fps is {dataset.fps}, expected {FPS}.")
    if dataset.num_episodes != expected_episodes:
        raise ValueError(f"Converted dataset has {dataset.num_episodes} episodes, expected {expected_episodes}.")
    if dataset.num_frames != expected_frames:
        raise ValueError(
            f"Converted dataset has {dataset.num_frames} frames, expected preflight gate {expected_frames}."
        )
    sample = dataset[0]
    for key in CAMERA_ALIASES.values():
        image = np.asarray(sample[f"observation.images.{key}"])
        if image.shape != (3, 480, 640):
            raise ValueError(f"Reloaded {key} image has shape {image.shape}, expected CHW (3, 480, 640).")
        if not np.isfinite(image).all():
            raise ValueError(f"Reloaded {key} image contains NaN or infinity.")
    for key in ("observation.state", "action"):
        values = np.asarray(sample[key])
        if values.shape[-1] != 14 or not np.isfinite(values).all():
            raise ValueError(f"Reloaded {key} is invalid: shape={values.shape}.")
    return {"episodes": dataset.num_episodes, "frames": dataset.num_frames, "fps": dataset.fps}


def convert(args: argparse.Namespace) -> Path:
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    dataset_root = output_root / args.repo_id
    if not args.validate_only and not args.skip_download:
        download_source(raw_root)
    preflight = run_preflight(raw_root)

    if args.validate_only:
        if not dataset_root.exists():
            raise FileNotFoundError(f"Converted dataset does not exist: {dataset_root}.")
        validation = validate_converted_dataset(args.repo_id, dataset_root, expected_frames=preflight.total_frames)
    else:
        if dataset_root.exists():
            if not args.overwrite:
                raise FileExistsError(f"Output already exists: {dataset_root}. Pass --overwrite to replace it.")
            shutil.rmtree(dataset_root)
        dataset_root.parent.mkdir(parents=True, exist_ok=True)
        dataset = create_dataset(args.repo_id, dataset_root, encoder_threads=args.encoder_threads)
        task_counts = dict.fromkeys(TASKS, 0)
        converted_frames = 0
        for task, path in tqdm(preflight.episode_files, desc="Converting ARX X5 episodes", unit="episode"):
            episode = inspect_episode(path, task)
            for frame in iter_frames(episode):
                dataset.add_frame(frame)
            dataset.save_episode()
            task_counts[task] += 1
            converted_frames += episode.num_frames
        dataset.finalize()
        if task_counts != preflight.task_episodes or converted_frames != preflight.total_frames:
            raise ValueError(
                "Converted data no longer matches the raw preflight gate: "
                f"episodes={task_counts}, frames={converted_frames}."
            )
        validation = validate_converted_dataset(args.repo_id, dataset_root, expected_frames=preflight.total_frames)

    manifest = {
        "source_repo_id": SOURCE_REPO_ID,
        "source_revision": SOURCE_REVISION,
        "repo_id": args.repo_id,
        "preflight_manifest": str(raw_root / PREFLIGHT_MANIFEST),
        "task_episodes": preflight.task_episodes,
        "task_frames": preflight.task_frames,
        **validation,
    }
    manifest_path = dataset_root / DATASET_MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", default=OUTPUT_REPO_ID)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--encoder-threads", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    print(convert(parse_args()))
