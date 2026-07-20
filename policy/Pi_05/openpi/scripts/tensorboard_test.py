from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import openpi.training.config as _config

from . import train


def test_camera_strip_is_nhwc_rgb():
    images = {
        "cam_high": np.zeros((6, 224, 224, 3), dtype=np.uint8),
        "cam_left_wrist": np.ones((6, 224, 224, 3), dtype=np.uint8),
        "cam_right_wrist": np.full((6, 224, 224, 3), 2, dtype=np.uint8),
    }

    strip = train.camera_strip(images)

    assert strip.shape == (5, 224, 672, 3)
    assert strip.dtype == np.uint8
    assert strip[0, 0, 0, 0] == 0
    assert strip[0, 0, 224, 0] == 1
    assert strip[0, 0, 448, 0] == 2


def test_tensorboard_contains_hparams_metrics_and_images(tmp_path: Path):
    config = dataclasses.replace(
        _config.get_config("pi05_base_aloha_full_real_piper_seed_0"),
        exp_name="tensorboard-test",
        checkpoint_dir_override=str(tmp_path / "checkpoint"),
        tensorboard_enabled=True,
        tensorboard_dir_override=str(tmp_path / "tensorboard"),
    )
    writer = train.init_tensorboard(config)
    assert writer is not None
    writer.add_scalar("loss", 1.0, 0)
    writer.add_scalar("grad_norm", 2.0, 0)
    writer.add_scalar("param_norm", 3.0, 0)
    writer.add_scalar("learning_rate", 2.5e-5, 0)
    writer.add_images("camera_views", np.zeros((5, 8, 24, 3), dtype=np.uint8), 0, dataformats="NHWC")
    writer.close()

    events = EventAccumulator(str(tmp_path / "tensorboard"))
    events.Reload()

    assert {"loss", "grad_norm", "param_norm", "learning_rate"} <= set(events.Tags()["scalars"])
    assert "camera_views" in events.Tags()["images"]
    assert "hparams/config/text_summary" in events.Tags()["tensors"]
