# RoboLab

[RoboLab](https://research.nvidia.com/labs/srl/projects/robolab) is NVIDIA's Isaac Lab benchmark for multi-task robot manipulation. This client follows the same shape as the other simulator clients:

- `main.py`: evaluate one or more named RoboLab tasks against an OpenPI policy server.
- `eval_all.py`: evaluate a curated subset or all benchmark tasks, with per-task logs and one aggregate `results.json`.
- Outputs: `examples/robolab_env/output/...` when launched through this client, or a user-provided `--output-dir`.

## Example Rollout

<a href="../../docs/assets/rollouts/robolab_one_bottle_square_pail_success.mp4">
  <img src="../../docs/assets/rollouts/robolab_one_bottle_square_pail_success.gif" alt="RoboLab OneBottleInSquarePailTask success rollout" width="420">
</a>

<sub><code>pi05_droid_jointpos</code>, OneBottleInSquarePailTask</sub>

## Setup

Initialize the RoboLab submodule and download its LFS assets:

```bash
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive third_party/robolab
git -C third_party/robolab lfs pull
```

Create the RoboLab client environment:

```bash
cd examples/robolab_env
uv venv --python 3.11
uv sync
```

RoboLab installs Isaac Sim 5.0 and Isaac Lab 2.2.0 through `uv`. Full evaluation expects a Linux host with an NVIDIA GPU, accepted Omniverse EULA, and the RoboLab assets downloaded by Git LFS.

## Serve

Start the policy server from the repo root. JAX serving is the default path.

```bash
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid_jointpos \
    --policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos
```

For the DROID `pi0_fast` checkpoint:

```bash
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_fast_droid_jointpos \
    --policy.dir=gs://openpi-assets-simeval/pi0_fast_droid_jointpos
```

## Evaluate

Run from `examples/robolab_env`.

Single task:

```bash
CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES \
uv run python main.py --policy pi05 --task BananaInBowlTask \
    --num-envs 1 --num-runs 1 --video-mode none
```

Multiple named tasks in one RoboLab runner:

```bash
CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES \
uv run python main.py --policy pi05 \
    --task BananaInBowlTask OneBottleInSquarePailTask \
    --num-envs 4 --num-runs 1 --enable-subtask
```

Curated smoke subset through `eval_all.py`:

```bash
CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES \
uv run python eval_all.py --policy pi05 --num-envs 1 --num-runs 1
```

All 120 benchmark tasks:

```bash
CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES \
uv run python eval_all.py --policy pi05 --task-set all --num-envs 10 --num-runs 1
```

Use the matching client variant when serving `pi0_fast`:

```bash
CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES \
uv run python main.py --policy pi0_fast --task BananaInBowlTask \
    --num-envs 1 --num-runs 1 --video-mode none
```

RoboLab vectorizes episodes inside one Isaac Sim process:

```text
episodes per task = --num-envs * --num-runs
```

Increase `--num-envs` before increasing `--num-runs`. Use `--num-episodes-adaptive MAX_N` for RoboLab's adaptive sampling mode. The default server connection is `0.0.0.0:8000`; pass `--host`, `--port`, or `--remote-uri` for other servers.

## Results

By default, `main.py` writes to `output/<policy>/` and `eval_all.py` writes to `output/<policy>-<task_set>/`. `main.py` writes `episode_results.jsonl` plus one task directory under the run directory. `eval_all.py` uses the same layout and adds `results.json` plus `parallel_logs/`:

```text
<output_dir>/
├── results.json
├── episode_results.jsonl
├── parallel_logs/task_NN_<task_name>.log
└── <task_name>/
    ├── env_cfg.json
    ├── log_<run>_env<env>.json
    ├── run_<run>.hdf5
    └── *.mp4  # when --video-mode is not none
```

RoboLab can resume an existing run directory and skip completed episodes. Use a fresh `--output-dir` when you want a clean run. The client refuses to reuse a directory that already contains results from a different policy.

No RoboLab release evaluation results are included in this release. Publish RoboLab numbers only after a fresh run from this client. The upstream dashboard can inspect generated RoboLab outputs:

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
