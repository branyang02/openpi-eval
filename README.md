# openpi-eval

Focused OpenPI evaluation fork for pretrained `pi05` and `pi0_fast` policies in
MetaWorld, LIBERO, and RoboCasa.

This release keeps the simulator integrations, parallel evaluation runners,
training data configs, websocket serving behavior, and serve-time JAX-to-PyTorch
conversion. It intentionally excludes non-evaluation research integrations.

## Rollout Clips

Example qualitative rollouts from release checkpoints:

<table>
  <tr>
    <th>MetaWorld</th>
    <th>LIBERO</th>
    <th>RoboCasa</th>
  </tr>
  <tr>
    <td>
      <video src="docs/assets/rollouts/metaworld_reach_success.mp4" controls muted loop playsinline width="260"></video>
      <br><sub><code>pi05_metaworld</code>, reach-v3</sub>
    </td>
    <td>
      <video src="docs/assets/rollouts/libero_bbq_sauce_success.mp4" controls muted loop playsinline width="260"></video>
      <br><sub><code>pi05_libero</code>, BBQ sauce to basket</sub>
    </td>
    <td>
      <video src="docs/assets/rollouts/robocasa_turn_on_sink_success.mp4" controls muted loop playsinline width="260"></video>
      <br><sub><code>pi05_robocasa</code>, TurnOnSinkFaucet</sub>
    </td>
  </tr>
</table>

## Configs

| Environment | `pi05` | `pi0_fast` |
|---|---|---|
| MetaWorld | `pi05_metaworld` | `pi0_fast_metaworld` |
| LIBERO | `pi05_libero` | `pi0_fast_libero` |
| RoboCasa | `pi05_robocasa` | `pi0_fast_robocasa` |

`pi0_fast_robocasa` is wired for future training and evaluation, but no released
checkpoint is currently available.

## Setup

```bash
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

Simulator environments use separate venvs:

```bash
cd examples/libero_env
uv sync
uv run python setup_libero_config.py

cd ../robocasa_env
uv sync
uv run python -m robocasa.scripts.setup_macros
uv run python -m robocasa.scripts.download_kitchen_assets
```

Use EGL on GPU machines:

```bash
export MUJOCO_GL=egl
```

## Checkpoints

| Config | Checkpoint |
|---|---|
| `pi05_metaworld` | `brandonyang/openpi-metaworld-25000` |
| `pi0_fast_metaworld` | `brandonyang/pi0fast-metaworld-checkpoints/pi0_fast_metaworld_b200_bs512/2500` |
| `pi05_libero` | `brandonyang/openpi-libero-9000` |
| `pi0_fast_libero` | `brandonyang/pi0fast-libero-checkpoints/pi0_fast_libero_b200_bs512/2000` |
| `pi05_robocasa` | `robocasa/robocasa365_checkpoints/pi05_pretrain_human300/multitask_learning/75000` |

Download commands live in the environment READMEs:
[MetaWorld](examples/metaworld/README.md),
[LIBERO](examples/libero_env/README.md), and
[RoboCasa](examples/robocasa_env/README.md).

## Serving

All clients call a WebSocket policy server:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir=/path/to/checkpoint
```

For `pi05`, PyTorch inference is optional. The first PyTorch serve converts the
JAX checkpoint in place:

```bash
uv run scripts/serve_policy.py --pytorch policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir=/path/to/checkpoint
```

## Evaluation

```bash
MUJOCO_GL=egl uv run examples/metaworld/eval_all.py --split train

cd examples/libero_env
MUJOCO_GL=egl uv run python eval_all.py --task_suite_name libero_10 --num_workers 5

cd ../robocasa_env
MUJOCO_GL=egl uv run python eval_all.py --task_set atomic_seen --num_workers 5
```

## Tests

```bash
uv run ruff check .
JAX_PLATFORMS=cpu uv run pytest --strict-markers -m "not manual" src/openpi packages scripts

cd examples/libero_env
uv run pytest tests/

cd ../robocasa_env
uv run pytest --strict-markers -m "not manual" tests/
```
