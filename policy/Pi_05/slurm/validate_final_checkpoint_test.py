from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_final_checkpoint import _latest_checkpoint  # noqa: E402


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
