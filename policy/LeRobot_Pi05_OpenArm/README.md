# LeRobot π0.5 OpenARM cloth folding

Native LeRobot v0.5.1 adapter for `lerobot-data-collection/folding_final`.
It preserves the checkpoint's right-first 16-dimensional OpenARM contract,
three-camera ordering, saved processors, and 30-step action chunks.

```bash
bash install.sh lerobot-pi05
bash download_checkpoint.sh lerobot-pi05
conda run -n lerobot-pi05 python smoke_test.py
```

Run from the RoboDojo parent checkout with `--env-cfg openarm_cloth_folding`,
`--action-type joint`, and the checkpoint label `folding_final`.
