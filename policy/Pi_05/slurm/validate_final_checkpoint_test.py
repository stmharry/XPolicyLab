from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_final_checkpoint import _inference_repack, _latest_checkpoint  # noqa: E402


def test_inference_repack_does_not_require_ground_truth_action() -> None:
    sample = {
        "observation.images.cam_high": "high",
        "observation.images.cam_left_wrist": "left",
        "observation.images.cam_right_wrist": "right",
        "observation.state": "state",
        "prompt": "prompt",
    }

    repacked = _inference_repack().inputs[0](sample)

    assert repacked == {
        "images": {"cam_high": "high", "cam_left_wrist": "left", "cam_right_wrist": "right"},
        "state": "state",
        "prompt": "prompt",
    }


def test_latest_checkpoint_requires_exact_final_step(tmp_path: Path) -> None:
    for step in (5_000, 29_999):
        (tmp_path / str(step) / "params").mkdir(parents=True)
        (tmp_path / str(step) / "assets").mkdir()

    assert _latest_checkpoint(tmp_path, 30_000) == tmp_path / "29999"


def test_latest_checkpoint_rejects_incomplete_or_early_run(tmp_path: Path) -> None:
    (tmp_path / "29000" / "params").mkdir(parents=True)
    (tmp_path / "29000" / "assets").mkdir()

    with pytest.raises(ValueError, match="Expected final checkpoint 29999"):
        _latest_checkpoint(tmp_path, 30_000)

    (tmp_path / "29999" / "params").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="missing assets"):
        _latest_checkpoint(tmp_path, 30_000)
