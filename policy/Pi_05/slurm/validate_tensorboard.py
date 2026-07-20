from __future__ import annotations

import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REQUIRED_SCALARS = {"loss", "grad_norm", "param_norm", "learning_rate"}


def validate(log_dir: Path) -> dict[str, list[str]]:
    events = EventAccumulator(str(log_dir), size_guidance={"scalars": 0, "images": 0, "tensors": 0})
    events.Reload()
    tags = events.Tags()
    missing_scalars = REQUIRED_SCALARS - set(tags["scalars"])
    if missing_scalars:
        raise ValueError(f"TensorBoard is missing scalar tags: {sorted(missing_scalars)}")
    if "camera_views" not in tags["images"]:
        raise ValueError("TensorBoard is missing the step-zero camera_views image strip.")
    if "hparams/config/text_summary" not in tags["tensors"]:
        raise ValueError("TensorBoard is missing the hparams/config text summary.")
    return {key: sorted(value) for key, value in tags.items() if isinstance(value, list)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    print(validate(args.log_dir))


if __name__ == "__main__":
    main()
