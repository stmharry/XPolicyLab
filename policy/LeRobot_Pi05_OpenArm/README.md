# LeRobot π0.5 OpenARM cloth folding

Native LeRobot v0.5.1 adapter for `lerobot-data-collection/folding_final`.
It preserves the checkpoint's right-first 16-dimensional OpenARM contract,
three-camera ordering, saved processors, and 30-step action chunks. Runtime
execution mirrors LeRobot's official folding command: RTC queue 30, horizon 20,
guidance 5.0, LINEAR prefix attention, relative-prefix re-anchoring, and 3×
action interpolation.

```bash
bash install.sh lerobot-pi05
bash download_checkpoint.sh lerobot-pi05
conda run -n lerobot-pi05 python smoke_test.py
```

Run from the RoboDojo parent checkout with `--env-cfg openarm_cloth_folding`,
`--action-type joint`, and the checkpoint label `folding_final`. This canonical
environment uses the available DYNA base camera while preserving the
checkpoint's observation and action contract.

Set `ROBODOJO_OPENARM_ZERO_ACTION=1` to run the same transport and simulator
path while holding the observed OpenARM state, without loading checkpoint
weights. This mode exists only for the required scene/camera smoke test.
