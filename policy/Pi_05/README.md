# Pi_05

**Contributor:** RoboDojo Team | **Paper:** Pi0.5 technical report | **arXiv:** TBD | **Original code:** https://github.com/Physical-Intelligence/openpi

`Pi_05` is the XPolicyLab/RoboDojo adapter for the corresponding policy. It keeps integration-facing scripts at this directory level and leaves the original or vendored implementation in the nested source tree when present.

<details>
<summary>File Structure</summary>

| Path | Purpose |
|---|---|
| `README.md` | Supplemental documentation or environment metadata. |
| `install.sh` | Installs the policy-side runtime and editable dependencies. |
| `process_data.sh` | Converts RoboDojo demonstration data into the policy-specific training format. |
| `train.sh` | Launches the XPolicyLab training wrapper for this policy. |
| `eval.sh` | Runs a same-machine policy server plus RoboDojo environment client evaluation. |
| `setup_eval_policy_server.sh` | Starts only the policy server for distributed/debug evaluation. |
| `setup_eval_env_client.sh` | Starts only the RoboDojo environment client and connects to a policy server. |
| `deploy.py` | Policy wrapper used by the XPolicyLab model server. |
| `model.py` | Model adapter loaded by `deploy.py` or the policy server. |
| `deploy.yml` | Runtime configuration and default checkpoint/model parameters. |
| `openpi/` | Vendored upstream code, policy-specific assets, or helper scripts. |

</details>

## Installation

What it does: installs or activates the policy-side runtime so the XPolicyLab server can import the adapter and upstream model code.

Parameters used by the command:

| Parameter | Description |
|---|---|
| `policy_uv_env` | `uv` to use `deploy.yml` `policy_uv_env_path`, or an explicit OpenPI project path. |

```bash
cd XPolicyLab/policy/Pi_05
# Example: install dependencies for the Pi_05 policy adapter.
bash install.sh
# `eval.sh` arg 9 is not a conda env. Pass `uv` or the OpenPI project path.
source openpi/.venv/bin/activate
```

From a RoboDojo checkout, use the standardized setup and validation surfaces:

```bash
make policy-setup
make preflight
make preflight DEEP=true
```

`prepare_eval_policy.sh` prepares either public alias when selected and installs
OpenPI plus XPolicyLab idempotently. `check_eval_policy.sh` is read-only. It
checks the selected physical GPU, the OpenPI uv lock and environment, required
imports, alias/environment/joint-action compatibility, Orbax `params`,
quantile-stat fields, and the pinned small-file hashes. It does not load model
weights. Local paths and non-public aliases retain legacy behavior and warn
when no pinned integrity profile exists.

### ARX X5 multitask public baseline

The checkpoint alias `pi05_arx5_multitask_v1` pins the public repository at
revision `880fa61406540d80b1c3b9824f12c19b903a233f` and requires checkpoint
step `55000`. It uses absolute 14D joint actions, three ARX cameras, and the
full 50-action horizon. Other local/path checkpoints retain their existing
resolution and transformation behavior.

```bash
cd XPolicyLab/policy/Pi_05
bash prepare_checkpoint.sh
bash install.sh
bash setup_eval_policy_server.sh \
  RoboDojo fold_clothes pi05_arx5_multitask_v1 arx_x5 joint 0 \
  0 uv 6000 0.0.0.0
```

The snapshot is stored under
`${ROBODOJO_STORAGE_ROOT:-<robodojo>/.robodojo}/model_weights/Pi_05/pi05_arx5_multitask_v1/880fa61406540d80b1c3b9824f12c19b903a233f`.
Preparation downloads only step 55000 and its assets, verifies every downloaded
Hub object and the normalization-stat checksum, and fails rather than selecting
another step. It always computes and compares the model card's declared plain-
tar digest
`7ee69681991cdc5e04b4759d3bf93bca5dac6bc98639ec7b00202d2f82fe5b2f`,
but a mismatch is informational: `tar cf -` hashes local uid/gid, modes,
mtimes, and traversal metadata in addition to file bytes, so the result is not
reproducible after `hf download`. Integrity acceptance is therefore enforced by
the pinned-revision, per-file Hub verification and the exact normalization hash;
neither gate is optional. RoboDojo grippers use `[0,1]`; the adapter maps them to checkpoint
units with `p=-0.01+0.054g` before normalization and reverses that mapping on
predicted actions.

### Bimanual YAM MolmoAct2 fine-tune

The checkpoint alias `pi05_yam_molmoact2` pins
[`robocurve/pi05-yam-molmoact2`](https://huggingface.co/robocurve/pi05-yam-molmoact2)
at revision `df991e11e8f6540098338c56342b1143fac5b952`. It requires the
`bimanual_yam` embodiment and joint actions, consumes the canonical top, left-
wrist, and right-wrist cameras, and predicts and executes all 16 absolute 14D
actions in each chunk. The adapter reconstructs the released `yam_pi05`
OpenPI inference config, loads the `yam-bimanual-merged` quantile statistics,
and shares the `yam_molmoact2` joint-frame bridge with MolmoACT2.

```bash
cd XPolicyLab/policy/Pi_05
bash prepare_checkpoint.sh pi05_yam_molmoact2
bash install.sh
bash setup_eval_policy_server.sh \
  RoboDojo general_pickup pi05_yam_molmoact2 bimanual_yam joint 0 \
  0 uv 6000 0.0.0.0
```

The complete pinned snapshot is stored under
`${ROBODOJO_STORAGE_ROOT:-<robodojo>/.robodojo}/model_weights/Pi_05/pi05_yam_molmoact2/df991e11e8f6540098338c56342b1143fac5b952`.
Preparation verifies the Hub manifest, requires every remote file, and checks
the released normalization and Orbax metadata hashes. The checkpoint profile
is independent of scene selection: `molmo_yam` is the recommended released
workspace for matched evaluation, while another compatible scene can be
selected without changing the embodiment or checkpoint contract.

## Demo Data Processing

What it does: prepares RoboDojo demonstration data for policy training. The output name should match the training run identity so `train.sh` can find it.

Parameters used by the command:

| Parameter | Description |
|---|---|
| `bench_name` | Benchmark or dataset family, usually `RoboDojo`. |
| `ckpt_name` | Data/run identifier. Use a different value for ablations, for example `stack_bowls_50ep`. |
| `env_cfg_type` | Robot/environment configuration, for example `arx_x5`. |
| `action_type` | Action representation, for example `joint`. |
| `expert_data_num` | Optional episode limit for data conversion only. It is not part of checkpoint naming. |
| `raw_task_dirs` | Optional source task directory or comma-separated task list under `data/<bench_name>/`; defaults to `ckpt_name`. |

```bash
cd XPolicyLab/policy/Pi_05
# Template: convert all available demonstrations for one run.
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type>

# Example: convert stack_bowls demos for arx_x5 joint control.
bash process_data.sh RoboDojo stack_bowls arx_x5 joint

# Example: write a differently named dataset while reading all stack_bowls demos.
bash process_data.sh RoboDojo stack_bowls_ablation arx_x5 joint stack_bowls

# Example: create a 50-episode ablation while reading from the original task data.
bash process_data.sh RoboDojo stack_bowls_50ep arx_x5 joint 50 stack_bowls
```

## Model Training

What it does: starts the policy-specific training recipe through the XPolicyLab wrapper and writes checkpoints under this adapter directory.

Parameters used by the command:

| Parameter | Description |
|---|---|
| `bench_name` | Benchmark or dataset family, usually `RoboDojo`. |
| `ckpt_name` | Training run identifier, for example `cotrain`. |
| `env_cfg_type` | Robot/environment configuration, for example `arx_x5`. |
| `action_type` | Action representation, for example `joint`. |
| `seed` | Random seed. |
| `gpu_id` | GPU id or comma-separated GPU ids for the policy trainer. `train.sh` sets `fsdp_devices=1` for one visible GPU and `2` for multi-GPU by default. |

```bash
cd XPolicyLab/policy/Pi_05
# Template: train a policy run on one GPU or a GPU list.
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# Example: train a cotrain run on GPU 0.
bash train.sh RoboDojo cotrain arx_x5 joint 0 0

# Example: train the same run on four GPUs if the upstream trainer supports it.
bash train.sh RoboDojo cotrain arx_x5 joint 0 0,1,2,3
```

The usual checkpoint directory is `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`. During evaluation, `ckpt_name` may be the short run name from training (auto-combined into that directory name), the full run-directory name, or a path to a checkpoint directory.

By default, training reads the LeRobot repo produced by `process_data.sh`: `<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>`. Override this with `OPENPI_LEROBOT_REPO_ID` when reusing an existing dataset.

## Deployment and Evaluation

What it does: serves the policy through XPolicyLab and connects it to a RoboDojo evaluation client. Use `eval.sh` for a same-machine smoke test, or split server/client scripts for debugging and multi-machine evaluation.

Parameters used by `eval.sh`:

| Parameter | Description |
|---|---|
| `bench_name` | Benchmark or dataset family, usually `RoboDojo`. |
| `task_name` | RoboDojo simulation task to evaluate, for example `stack_bowls`. |
| `ckpt_name` | Checkpoint/run directory name, usually under `checkpoints/`. |
| `env_cfg_type` | Robot/environment configuration, for example `arx_x5`. |
| `action_type` | Action representation, for example `joint`. |
| `seed` | Evaluation seed. |
| `policy_gpu_id` | GPU used by the policy server. |
| `env_gpu_id` | GPU used by the RoboDojo simulation client. |
| `policy_uv_env` | `uv` or an explicit OpenPI project path for the policy server. |
| `eval_env_conda_env` | Conda environment for RoboDojo simulation/client. |

```bash
cd XPolicyLab/policy/Pi_05
# Template: run same-machine policy server and RoboDojo environment client.
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <policy_gpu_id> <env_gpu_id> <policy_uv_env> <eval_env_conda_env>

# Example: evaluate a trained cotrain checkpoint on stack_bowls.
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 0 0 uv <eval_env_conda_env>
```

Parameters used by the split server/client flow:

| Parameter | Description |
|---|---|
| `bench_name` | Benchmark or dataset family, usually `RoboDojo`. |
| `task_name` | RoboDojo simulation task to evaluate, for example `stack_bowls`. |
| `ckpt_name` | Checkpoint/run directory name, usually under `checkpoints/`. |
| `env_cfg_type` | Robot/environment configuration, for example `arx_x5`. |
| `action_type` | Action representation, for example `joint`. |
| `seed` | Evaluation seed. |
| `policy_gpu_id` | GPU used by the policy server. |
| `env_gpu_id` | GPU used by the RoboDojo simulation client. |
| `policy_uv_env` | `uv` or an explicit OpenPI project path for the policy server. |
| `eval_env_conda_env` | Conda environment for RoboDojo simulation/client. |
| `policy_server_port` | Port exposed by the policy server, for example `5000`. |
| `policy_server_host` | Server bind host, for example `0.0.0.0` on the policy machine. |
| `policy_server_ip` | IP or hostname that the environment client uses to reach the policy server. |
| `additional_info` | Comma-separated runtime overrides passed to the eval client, for example `ckpt_name=...,action_type=joint`. |

```bash
cd XPolicyLab/policy/Pi_05
# Terminal 1 on the policy machine: start the policy server.
bash setup_eval_policy_server.sh \
  <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <policy_uv_env> <policy_server_port> <policy_server_host>

# Example: bind the policy server to all interfaces on port 5000.
bash setup_eval_policy_server.sh \
  RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 \
  0 uv 5000 0.0.0.0

# Terminal 2 on the environment machine: connect RoboDojo to the policy server.
bash setup_eval_env_client.sh \
  <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <env_gpu_id> <eval_env_conda_env> <additional_info> \
  <policy_server_port> <policy_server_ip>

# Example: connect to a policy server reachable at <policy_server_ip>:5000.
bash setup_eval_env_client.sh \
  RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 \
  0 <eval_env_conda_env> "ckpt_name=RoboDojo-cotrain-arx_x5-joint-0,action_type=joint" \
  5000 <policy_server_ip>
```

Set `EVAL_ENV_TYPE=debug` for offline shape/IO checks when the adapter supports it; leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation.

## Important Parameters

Common parameter meanings used across the commands above:

| Parameter | Description |
|---|---|
| `bench_name` | Benchmark or dataset family, usually `RoboDojo`. |
| `task_name` | RoboDojo simulation task to evaluate, for example `stack_bowls`. |
| `ckpt_name` | Checkpoint/run directory name, usually under `checkpoints/`. |
| `env_cfg_type` | Robot/environment configuration, for example `arx_x5`. |
| `action_type` | Action representation, for example `joint`. |
| `seed` | Evaluation seed. |
| `policy_gpu_id` | GPU used by the policy server. |
| `env_gpu_id` | GPU used by the RoboDojo simulation client. |
| `policy_uv_env` | `uv` to use `deploy.yml` `policy_uv_env_path`, or an explicit OpenPI project path for the policy server. |
| `eval_env_conda_env` | Conda environment for RoboDojo simulation/client. |

Policy-specific `deploy.yml` keys worth checking before evaluation:

| Key | Notes |
|---|---|
| `policy_name` | Runtime or checkpoint option consumed by this adapter. |
| `checkpoint_num` | Runtime or checkpoint option consumed by this adapter. |
| `result_dir` | Runtime or checkpoint option consumed by this adapter. |
| `obs_transform_pipeline` | Runtime or checkpoint option consumed by this adapter. |
| `policy_uv_env_path` | Runtime or checkpoint option consumed by this adapter. |
| `train_config_name` | Runtime or checkpoint option consumed by this adapter. |
| `repo_id` | Runtime or checkpoint option consumed by this adapter. |

Frequently used environment variables detected in the adapter scripts:

| Variable | Notes |
|---|---|
| `CONDA_BASE` | Optional override used by the local scripts or upstream runtime. |
| `GIT_LFS_SKIP_SMUDGE` | Optional override used by the local scripts or upstream runtime. |
| `HF_DATASETS_CACHE` | Optional override used by the local scripts or upstream runtime. |
| `JAX_COMPILATION_CACHE_DIR` | Optional override used by the local scripts or upstream runtime. |
| `LOCAL_CACHE_ROOT` | Optional override used by the local scripts or upstream runtime. |
| `OPENPI_DATA_MODE` | Optional override used by the local scripts or upstream runtime. |
| `OPENPI_FSDP_DEVICES` | Overrides the FSDP device count passed to OpenPI training. |
| `OPENPI_LEROBOT_REPO_ID` | Overrides the LeRobot repo id used by `train.sh`; defaults to `<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>`. |
| `OPENPI_LOCAL_CACHE_ROOT` | Optional override used by the local scripts or upstream runtime. |
| `OPENPI_ROOT` | Optional override used by the local scripts or upstream runtime. |
| `OPENPI_SRC` | Optional override used by the local scripts or upstream runtime. |
| `OPENPI_TRAIN_CONFIG_NAME` | Optional override used by the local scripts or upstream runtime. |
| `POLICY_DIR` | Optional override used by the local scripts or upstream runtime. |
| `PYENV` | Optional override used by the local scripts or upstream runtime. |

## Notes

- Keep `ckpt_name` stable between data processing, training, and evaluation. For data-size ablations, encode the subset in `ckpt_name` such as `stack_bowls_50ep`.
- `task_name` is only the evaluation task; multi-task checkpoints can be evaluated on different tasks without renaming the checkpoint directory.
- Prefer running `setup_eval_policy_server.sh` and `setup_eval_env_client.sh` separately when debugging dependency, CUDA, or model-loading issues.
