# Normalization statistics

OpenPI normalizes proprioceptive inputs and action targets during training and
inference. The statistics are computed from each training dataset and stored
with the checkpoint under `assets/<asset_id>/norm_stats.json`.

## Compute statistics

Run normalization before training a focused-release config:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_metaworld
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

RoboCasa checkpoints include their own normalization assets. The RoboCasa README
shows the expected checkpoint asset layout after download.

## Load statistics at inference time

`scripts/serve_policy.py` loads normalization statistics from the checkpoint
directory through the selected config:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir=checkpoints/openpi-libero-9000
```

For custom checkpoints, make sure the checkpoint directory contains the asset ID
expected by the config:

| Config | Asset ID |
|---|---|
| `pi05_metaworld` | `brandonyang/metaworld_ml45` |
| `pi0_fast_metaworld` | `brandonyang/metaworld_ml45` |
| `pi05_libero` | `physical-intelligence/libero` |
| `pi0_fast_libero` | `physical-intelligence/libero` |
| `pi05_robocasa` | `robocasa` |
| `pi0_fast_robocasa` | `robocasa` |

If normalization assets are missing, recompute them with
`scripts/compute_norm_stats.py` for trainable configs or copy the released
checkpoint assets into the expected directory.
