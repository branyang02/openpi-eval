# RoboLab

[RoboLab](https://research.nvidia.com/labs/srl/projects/robolab) is NVIDIA's Isaac Lab benchmark for multi-task robot manipulation.

This example uses its own Python 3.11 venv. The simulator runs here and talks to the root policy server over WebSocket.

- `main.py`: one or more named RoboLab tasks.
- `eval_all.py`: one subprocess per task in the benchmark set or an explicit task list.

## Example Rollout

<a href="../../docs/assets/rollouts/robolab_one_bottle_square_pail_success.mp4">
  <img src="../../docs/assets/rollouts/robolab_one_bottle_square_pail_success.gif" alt="RoboLab OneBottleInSquarePailTask success rollout" width="420">
</a>

<sub><code>pi05_droid_jointpos</code>, OneBottleInSquarePailTask</sub>

## Setup

```bash
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive third_party/robolab
git -C third_party/robolab lfs pull

cd examples/robolab_env
uv venv --python 3.11
uv sync
```

RoboLab installs Isaac Sim 5.0 and Isaac Lab 2.2.0 through `uv`. Full evaluation expects a Linux host with an NVIDIA GPU, accepted Omniverse EULA, and the RoboLab assets downloaded by Git LFS.

## Configs

Registered configs:

- `pi05_droid_jointpos`
- `pi0_fast_droid_jointpos`

Both configs use the DROID joint-position action space and RoboLab's Pi0-family client.

## Checkpoints

- `pi05_droid_jointpos`: `gs://openpi-assets-simeval/pi05_droid_jointpos`
- `pi0_fast_droid_jointpos`: `gs://openpi-assets-simeval/pi0_fast_droid_jointpos`

The policy server can read these checkpoint paths directly.

## Serve

Start the policy server from the repo root.

```bash
# pi0.5, JAX backend
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid_jointpos \
    --policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos

# pi0-FAST, JAX backend
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_fast_droid_jointpos \
    --policy.dir=gs://openpi-assets-simeval/pi0_fast_droid_jointpos
```

## Evaluate

Run clients from `examples/robolab_env`.

```bash
# Single task
CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES \
uv run python main.py --policy pi05 --task BananaInBowlTask \
    --num-envs 1 --num-runs 1 --video-mode none

# Full benchmark set
CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES \
uv run python eval_all.py --policy pi05 --num-envs 10 --num-runs 1

# Explicit task list
CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES \
uv run python eval_all.py --policy pi05 \
    --tasks BananaInBowlTask OneBottleInSquarePailTask \
    --num-envs 1 --num-runs 1

# pi0-FAST single task
CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES \
uv run python main.py --policy pi0_fast --task BananaInBowlTask \
    --num-envs 1 --num-runs 1 --video-mode none
```

RoboLab vectorizes episodes inside one Isaac Sim process:

```text
episodes per task = --num-envs * --num-runs
```

Increase `--num-envs` before increasing `--num-runs`. Use `--num-episodes-adaptive MAX_N` for RoboLab's adaptive sampling mode. `eval_all.py` launches one `main.py` subprocess per task; keep `--num-workers 1` unless the host has enough CPU/GPU memory for multiple Isaac Sim processes.

Output layout:

```text
examples/robolab_env/output/<policy>-<task_set>/
|-- results.json
|-- episode_results.jsonl
|-- parallel_logs/task_NN_<task_name>.log
`-- <task_name>/
    |-- env_cfg.json
    |-- log_<run>_env<env>.json
    |-- run_<run>.hdf5
    `-- *.mp4  # when --video-mode is not none
```

Generated results are written to `examples/robolab_env/output/` and should be
published only after a fresh release evaluation.

## Results

No RoboLab release evaluation results are included in this release.

Generated RoboLab outputs can also be inspected with the upstream dashboard:

```bash
cd third_party/robolab
uv run robolab-dashboard
```

## Tests

```bash
cd examples/robolab_env
uv run pytest tests/ -v
```

Full simulator evaluation is manual because it requires Isaac Sim, RoboLab assets, and GPU memory.

## Troubleshooting

If the RoboLab checkout is corrupted or stuck with stale LFS pointers, reset only that submodule before reinstalling:

```bash
git submodule deinit -f third_party/robolab
rm -rf third_party/robolab .git/modules/third_party/robolab
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive third_party/robolab
git -C third_party/robolab lfs pull
```
