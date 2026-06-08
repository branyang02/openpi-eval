# MetaWorld Wan2.2 + IDM

This environment prototypes a Direct Video-Action style controller on the
`brandonyang/metaworld_ml45` dataset:

```text
current camera image + task text -> Wan2.2 imagined future video -> IDM action chunk
```

The inverse dynamics model (IDM) is trained separately from the video model.
Wan2.2 is the intended inference-time world model; the small conv video model
is only kept for fast smoke tests and CI.

## Temporal alignment

The IDM, the cached Wan2.2 futures, and every eval/ranking run share one
temporal contract: `frame_delta`, `num_future_frames`, and `action_horizon`
must match end to end. The released best checkpoint was trained with
`frame_delta=1`, `num_future_frames=4`, `action_horizon=4`.

For eval and ranking, the IDM checkpoint's `model_config` and dataset
`train_config` are the source of truth: `image_size`, `num_future_frames`,
`action_horizon`, and `frame_delta` are read back from the checkpoint, and
cached/live Wan futures whose selected generated-video frames or recorded
source-frame offsets disagree fail loudly instead of silently falling back. Raw
Wan2.2 is valid here only at `--frame-delta 1`; for larger temporal strides, use
a Wan-LoRA fine-tuned from an export with that `frame_delta`. Legacy checkpoints
without recorded `frame_delta` emit a warning and should be treated as
inspection-only until retrained.

## Setup

```bash
cd examples/world_model_env
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra video
```

The `video` extra installs `flash-attn`. The project config tells uv to build it
against the resolved runtime `torch`; if uv builds `flash-attn` from source, the
host must have a compatible CUDA toolkit with `nvcc` on `PATH`. On systems that
still require non-isolated `flash-attn` builds, rerun setup with
`uv sync --extra video --no-build-isolation-package flash-attn`.

Wan2.2 runs from its own checkout/checkpoint:

```bash
git clone https://github.com/Wan-Video/Wan2.2 /path/to/Wan2.2
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir /path/to/Wan2.2-TI2V-5B
```

## Export Wan2.2 Data

Wan2.2 inputs are the current RGB frame and task text. For DiffSynth-Studio
fine-tuning, each exported video stores the current conditioning frame as frame
0 and future frames after it.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python export_wan_dataset.py \
    --dataset-source lerobot \
    --episodes 0 \
    --max-samples 256 \
    --image-size 128 \
    --num-future-frames 16 \
    --frame-delta 1 \
    --output-dir output/wan_metaworld_corner4
```

This writes:

```text
output/wan_metaworld_corner4/
|-- metadata.csv
|-- manifest.jsonl
|-- videos/
|-- images/
`-- captions/
```

Use `metadata.csv` as DiffSynth's `--dataset_metadata_path` and pass
`--extra_inputs input_image` for Wan2.2-TI2V-5B training.

Keep the same resolution/aspect ratio through export, LoRA training, cache
generation, and live serving. The commands here use square `128x128`, matching
the closed-loop smoke/eval runs in this ledger; use a larger matched square
resolution for quality experiments if compute allows.

To validate the exported dataset and print a DiffSynth LoRA command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python prepare_wan_finetune.py \
    --dataset-dir output/wan_metaworld_corner4 \
    --num-frames 17 \
    --mode lora
```

Run the stricter DiffSynth preflight/launcher. Without `--run`, this only writes
the exact command and environment checks to `wan_finetune_preflight.json`.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python run_wan_finetune.py \
    --dataset-dir output/wan_metaworld_corner4 \
    --diffsynth-repo-dir /tmp/DiffSynth-Studio \
    --checkpoint-dir /tmp/wan2.2-ti2v-5b \
    --preflight-output-dir output/wan_metaworld_lora_preflight \
    --output-path output/wan_metaworld_lora \
    --num-frames 17 \
    --height 128 \
    --width 128 \
    --mode lora
```

Add `--run` to launch `accelerate launch` from the DiffSynth checkout.
When `--checkpoint-dir` is set, the launcher uses the local T5, DiT shards,
VAE, and tokenizer instead of relying on model-id downloads.

Generate a small visual sample from a trained LoRA:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python validate_wan_lora.py generate \
    --diffsynth-repo-dir /tmp/DiffSynth-Studio \
    --checkpoint-dir /tmp/wan2.2-ti2v-5b \
    --lora-path output/wan_metaworld_lora/epoch-0.safetensors \
    --input-image output/wan_metaworld_corner4/images/sample_000000.png \
    --prompt "Robot manipulation in MetaWorld. Generate the near future scene." \
    --output-video output/wan_metaworld_lora/validate_sample.mp4 \
    --height 128 \
    --width 128 \
    --num-frames 17
```

## Dry Run

Before training, validate the whole data path with an untrained IDM and
dataset future frames standing in for Wan output:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python dry_run_pipeline.py \
    --dataset-source synthetic \
    --image-size 32 \
    --frame-delta 1 \
    --num-future-frames 4 \
    --action-horizon 8 \
    --device cuda:0 \
    --output-dir output/dry_run_pipeline
```

This writes `current_frame.png`, `wan_like_future.mp4`, and `dry_run.json`.

For a multi-sample pipeline check without trained Wan, use dataset future frames
as the explicit stand-in for Wan output:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python eval_pipeline.py \
    --dataset-source synthetic \
    --max-samples 64 \
    --image-size 32 \
    --frame-delta 1 \
    --num-future-frames 4 \
    --action-horizon 8 \
    --device cuda:0 \
    --output-dir output/pipeline_eval
```

This writes `pipeline_eval.json`, `current_frame.png`, and a
`*_future_debug.mp4` video.

## Matched Wan Action-Mode Comparison

The planner defaults reproduce the existing matched comparison family scaled to
256 samples. To print the matched command sequence without starting GPU work:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python plan_wan_action_mode_experiment.py \
    --samples-per-episode 32 \
    --output-root output/wan_action_modes_matched_ep16_23_spe32_h4
```

The printed plan keeps the dataset/source/image key/action horizon/sample
fingerprint and matched checkpoint family aligned across the decoded-video IDM
path and the Wan hidden-prefix action-expert paths, writes the comparison JSON
with `run_wan_action_mode_matrix.py`, and suggests a two-GPU split. Treat the
rows as comparable only if the matrix reports `sample_sets_match=true`.
`current_wan_prefix_action_expert` is Pi0.5-style only in the sense that frozen
Wan current-frame/text hidden-prefix memory is run once and reused by the action
expert. True native Wan attention KV-cache reuse is not implemented or exposed
in the current DiffSynth path; hidden-prefix tokens are action-expert inputs, not
cached Wan attention KV.

The strongest current-valid decoded-video modular evidence is
`output/diagnose_loop20_current_valid_cache_recheck_ep16_19_first64/idm_diagnostics.json`,
using generated cache `output/loop16_cache_epoch1_f17_steps8_ep16_19_first64`:
`idm_mse=0.06194690754637122`, mean baseline `0.33408883213996887`,
`future_blind=false`. The same-family prior GT reference is around
`0.04420376801863313`.

Matched 256-sample eval root:
`output/wan_action_modes_matched_ep16_23_spe32_h4_epoch4_steps4_seed7`. Broad
current-prefix+state trained on the 44-task train2 `spe16` cache scored
`dataset_action_mse=0.163603` on the matched ep16-23 eval, roughly tied with
the matched decoded-video smoke checkpoint (`0.160416`) and better than the mean
baseline (`0.352071`). That 256-sample decoded-video result is weaker evidence
for the generated-video bottleneck because same-split GT is basically identical
(`0.160341`); keep it as matched action-mode smoke/comparison evidence, not the
best modular generated-video result.
Fixed deterministic-noise partial-prefix caches improved the partial result to
`0.693690`, but noisy partial future slots still underperform this action
expert.
Cache-style action-memory variants remain behind encoder-style current-prefix
and the matched decoded-video smoke checkpoint. `decoder_arch='suffix_prefix_cache'`
scored `0.195745` and its FiLM point scored `0.192989`; the 3-token
`joint_softmax_prefix_cache` ablations scored `0.285013` additive and
`0.195765` FiLM vs the same mean baseline `0.352071`. The tokenpool4 richer
Wan-prefix ablation uses 12 prefix tokens (4 tokens each from layers
`[0, 14, 29]`) under `current_wan_prefix_action_expert`. It improved
joint-softmax FiLM to `0.189409`, so richer prefix tokens help the action-memory
path a little, but the simple encoder zero-noise result moved from `0.163603`
to `0.169355`; random-noise training eval improved from `~0.270` to `0.253`.
Tokenpool4 still does not beat the matched decoded-video smoke checkpoint
(`0.160416`) or the best 3-token encoder zero-noise result (`0.163603`). Treat
the joint-softmax prefix-cache and tokenpool4 runs as compact Wan-prefix
action-expert benchmarks
with learned reusable prefix memory and joint prefix/action softmax, not faithful
Pi0.5 KV-cache semantics or native Wan attention KV reuse. The local cache
builds learned action-expert prefix K/V from updated prefix tokens and lacks
Pi0.5 positional/RoPE suffix offsets.
The same-task ep0-15 -> ep16-23 prefix+state score (`0.034215`) is assembly-only
and should not be read as broad generalization. See
`EXPERIMENT_LEDGER.md` for the full action-mode ledger and caveats.

## Future Cache

Wan2.2 generation is expensive, so generated futures should be cached once and
then reused for IDM eval/ranking only. IDM training uses real MetaWorld
trajectory futures from the dataset, not generated or cached Wan futures.

Smoke-test the cache format with dataset futures:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python cache_future_rollouts.py \
    --future-source dataset_future \
    --dataset-source synthetic \
    --max-samples 8 \
    --image-size 32 \
    --frame-delta 1 \
    --num-future-frames 4 \
    --action-horizon 8 \
    --output-dir output/future_cache_smoke
```

Score cached futures against dataset futures and write a visual contact sheet:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python evaluate_future_cache.py \
    --cache-dir output/future_cache_smoke \
    --dataset-source synthetic \
    --max-samples 8 \
    --image-size 32 \
    --frame-delta 1 \
    --num-future-frames 4 \
    --action-horizon 8 \
    --output-dir output/future_cache_quality
```

Use cached futures in pipeline eval:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python eval_pipeline.py \
    --future-source cached \
    --cached-future-dir output/future_cache_smoke \
    --dataset-source synthetic \
    --max-samples 8 \
    --image-size 32 \
    --frame-delta 1 \
    --num-future-frames 4 \
    --action-horizon 8 \
    --batch-size 4 \
    --device cuda:0 \
    --output-dir output/pipeline_eval_cached
```

For raw Wan2.2 futures, switch `--future-source wan2_2`, provide
`--wan-repo-dir` plus `--wan-checkpoint-dir`, and keep `--frame-delta 1`. Raw
Wan2.2 is not fine-tuned on subsampled MetaWorld exports, so larger temporal
strides require the `wan_lora` path below.

For a DiffSynth-trained Wan2.2 LoRA, cache generated futures once and score
them before using them for IDM eval:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python cache_future_rollouts.py \
    --future-source wan_lora \
    --dataset-source lerobot \
    --repo-id brandonyang/metaworld_ml45 \
    --image-key corner4.image \
    --episodes 0 \
    --max-samples 64 \
    --image-size 64 \
    --frame-delta 1 \
    --num-future-frames 4 \
    --action-horizon 4 \
    --output-dir output/wan_lora_cache \
    --diffsynth-repo-dir /tmp/DiffSynth-Studio \
    --wan-lora-checkpoint-dir /tmp/wan2.2-ti2v-5b \
    --wan-lora-path output/wan_metaworld_lora/epoch-2.safetensors \
    --wan-lora-height 128 \
    --wan-lora-width 128 \
    --wan-lora-num-frames 17 \
    --wan-lora-num-inference-steps 8 \
    --wan-lora-future-frame-strategy first

UV_CACHE_DIR=/tmp/uv-cache uv run python evaluate_future_cache.py \
    --cache-dir output/wan_lora_cache \
    --dataset-source lerobot \
    --repo-id brandonyang/metaworld_ml45 \
    --image-key corner4.image \
    --episodes 0 \
    --max-samples 64 \
    --image-size 64 \
    --frame-delta 1 \
    --num-future-frames 4 \
    --action-horizon 4 \
    --output-dir output/wan_lora_cache_quality
```

## Rank Wan2.2 LoRA Checkpoints

Pixel future quality alone does not say whether a LoRA's imagined futures still
encode the action. `validate_wan_lora.py rank` scores each LoRA's cached futures
on both axes with an existing IDM checkpoint: pixel future metrics (reusing
`evaluate_future_cache.py`) and IDM action decodability (`idm_mse` on the cached
futures, plus `idm_decodability_gap` against the ground-truth-future reference).
Dataset shape (image size, future frames, action horizon) is read from the IDM
checkpoint and the cached futures are validated against it, so train the IDM and
build the caches with matching `--image-size`, `--num-future-frames`, and
`--action-horizon`. `frame_delta` is read from modern checkpoints when present;
pass `--frame-delta 1` explicitly for matching legacy checkpoints and the
canonical runs here. The run fails loudly on any missing checkpoint, cache
directory, manifest, or `config.json`, and on any cached-future shape mismatch,
instead of silently falling back.

Cache one future directory per candidate LoRA (see "Future Cache" above), then
rank them:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python validate_wan_lora.py rank \
    --idm-checkpoint output/idm_corner4_h4/best_idm_checkpoint.pt \
    --cache-dirs output/wan_lora_cache_epoch2 output/wan_lora_cache_epoch4 \
    --labels epoch2 epoch4 \
    --dataset-source lerobot \
    --repo-id brandonyang/metaworld_ml45 \
    --image-key corner4.image \
    --episodes 0 \
    --max-samples 64 \
    --frame-delta 1 \
    --rank-by idm_decodability_gap \
    --device cuda:0 \
    --output-dir output/wan_lora_ranking
```

This writes `output/wan_lora_ranking/ranking_summary.json` with per-checkpoint
pixel and IDM metrics, the ground-truth-future reference, and rankings by
`idm_mse`, `idm_decodability_gap`, and `future_mse`. `idm_decodability_gap` is
reported per checkpoint as the signed difference `idm_mse(generated) -
idm_mse(ground_truth)` (positive means the imagined futures are harder to decode
the action from than the real ones), but the `idm_decodability_gap` ranking sorts
by its magnitude: the most faithful checkpoint is the one whose futures are *as*
action-decodable as the real futures, i.e. gap closest to zero. A checkpoint whose
futures are artificially easier to decode (negative gap) is no more faithful than
one that is equally harder, so it is not allowed to rank first.

Cached futures can also be used directly for IDM evaluation:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python eval_idm.py \
    --checkpoint output/idm_corner4/idm_checkpoint.pt \
    --dataset-source synthetic \
    --cached-future-dir output/future_cache_smoke \
    --image-size 32 \
    --frame-delta 1 \
    --num-future-frames 4 \
    --action-horizon 8 \
    --output-dir output/idm_cached_eval
```

## Train IDM

The Wan2.2 path defaults to a single selected camera view, so IDM training uses
`corner4.image` by default. The IDM consumes current frames, future frames, and
proprioceptive state; task text stays on the Wan2.2 prompt side. The active IDM
path is a flow-matching diffusion transformer over action chunks.

`train_idm.py` trains only on ground-truth future frames from the MetaWorld
dataset trajectory. Generated/cached Wan futures are eval and ranking inputs
only; passing cache flags to `train_idm.py` fails before training starts.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python train_idm.py \
    --dataset-source lerobot \
    --episodes 0 \
    --max-samples 2048 \
    --image-size 128 \
    --frame-delta 1 \
    --num-future-frames 4 \
    --action-horizon 4 \
    --idm-arch flow_transformer \
    --latent-dim 1024 \
    --idm-transformer-layers 14 \
    --idm-transformer-heads 16 \
    --idm-transformer-ff-dim 4096 \
    --idm-transformer-patch-size 16 \
    --idm-flow-sampling-steps 16 \
    --idm-flow-endpoint-loss-weight 0.1 \
    --idm-future-noise-std 0.12 \
    --early-stopping-patience 12 \
    --early-stopping-min-delta 0.0005 \
    --device cuda:0 \
    --output-dir output/idm_flow_dit_350m_corner4
```

This variant trains a conditional flow velocity over action chunks and samples
actions by integrating the learned velocity field from Gaussian action noise at
inference. `--idm-flow-num-samples` averages several sampled chunks for a more
stable decoded action. Eval uses `--flow-eval-seed 0` by default so flow
checkpoint selection is reproducible. `--idm-flow-endpoint-loss-weight` is an
optional auxiliary action endpoint loss; the training CLI defaults it to `0.1`.
The default flow-DiT profile above is about 356M trainable IDM parameters.

To experiment with frozen Wan2.2-5B VAE latents as the IDM visual encoder,
switch the visual backend and pass the local Wan/DiffSynth paths:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra video python train_idm.py \
    --dataset-source lerobot \
    --episodes 0 \
    --max-samples 2048 \
    --image-size 64 \
    --frame-delta 1 \
    --num-future-frames 4 \
    --action-horizon 4 \
    --idm-arch flow_transformer \
    --idm-visual-encoder wan_vae \
    --wan-vae-repo-dir /tmp/DiffSynth-Studio \
    --wan-vae-checkpoint-path /tmp/wan2.2-ti2v-5b \
    --device cuda:0 \
    --output-dir output/idm_flow_dit_350m_wan_vae_corner4
```

Run a small comparison grid and evaluate both final and best checkpoints:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python run_idm_experiments.py \
    --dataset-source lerobot \
    --episodes 0 \
    --max-samples 512 \
    --image-size 64 \
    --frame-delta 1 \
    --num-future-frames 4 \
    --action-horizons 4 8 \
    --epochs 10 \
    --batch-size 16 \
    --learning-rates 0.0003 \
    --device cuda:0 \
    --output-dir output/idm_metaworld_grid
```

Inspect action-range and per-dimension IDM errors:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python diagnose_idm.py \
    --checkpoint output/idm_corner4/idm_checkpoint.pt \
    --dataset-source lerobot \
    --episodes 0 \
    --max-samples 2048 \
    --image-size 128 \
    --frame-delta 1 \
    --num-future-frames 1 \
    --action-horizon 32 \
    --device cuda:0 \
    --output-dir output/idm_corner4_diagnostics
```

## Wan2.2 + IDM Inference

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python infer_wan_idm.py \
    --idm-checkpoint output/idm_corner4/idm_checkpoint.pt \
    --wan-repo-dir /path/to/Wan2.2 \
    --wan-checkpoint-dir /path/to/Wan2.2-TI2V-5B \
    --episodes 0 \
    --frame-delta 1 \
    --sample-index 0 \
    --task-prompt "complete the MetaWorld task" \
    --device cuda:0 \
    --output-dir output/wan_idm_sample
```

The script saves the Wan2.2 video plus `wan_idm_action.json`.

## Closed-Loop Policy Server (MetaWorld)

`serve_world_model.py` wraps a trained IDM as an [openpi](../../packages/openpi-client)
websocket policy server, so the existing `examples/metaworld` driver can run
closed-loop smoke tests against it. The server itself has no MetaWorld/MuJoCo
dependency — the environment driver stays in the root example and connects as a
client.

The smoke future provider is `repeat_current`: it repeats the current frame as
the IDM's "future", which keeps plumbing tests fast and Wan-free. It is guarded
by `--allow-repeat-current` so it is never mistaken for a real world-model eval.
For DiffSynth-trained Wan2.2 LoRA futures, use the named `wan_lora` provider and
pass the DiffSynth repo, base checkpoint, and LoRA checkpoint paths.

```bash
# Terminal 1 — serve a smoke IDM policy. image-keys count must equal the IDM num_views.
UV_CACHE_DIR=/tmp/uv-cache uv run python serve_world_model.py \
    --idm-checkpoint output/idm_corner4/idm_checkpoint.pt \
    --image-keys observation/image \
    --future-provider repeat_current \
    --allow-repeat-current \
    --device cuda:0 \
    --port 8000

# Terminal 2 — drive MetaWorld against the server (in the metaworld example env).
MUJOCO_GL=egl uv run examples/metaworld/main.py --env_name reach-v3 --port 8000
```

Real Wan-LoRA serving uses the same websocket API:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python serve_world_model.py \
    --idm-checkpoint output/idm_corner4/idm_checkpoint.pt \
    --image-keys observation/image \
    --future-provider wan_lora \
    --diffsynth-repo-dir /tmp/DiffSynth-Studio \
    --wan-lora-checkpoint-dir /tmp/wan2.2-ti2v-5b \
    --wan-lora-path output/wan_metaworld_lora/epoch-3.safetensors \
    --wan-lora-height 128 \
    --wan-lora-width 128 \
    --wan-lora-num-frames 17 \
    --wan-lora-num-inference-steps 8 \
    --wan-lora-future-frame-strategy first \
    --device cuda:0 \
    --port 8000
```

The server speaks the same request/response shape as the other clients: it accepts
batched observations (`observation/image`, `observation/state`, `prompt`) and
returns `{"actions": (batch, action_horizon, action_dim)}`. Missing observation
keys, missing prompts for `wan_lora`, or state-dimension mismatches fail loudly.

For MetaWorld closed-loop eval, set `--replan-steps` no larger than the IDM
checkpoint's `action_horizon`; the canonical Loop-11 checkpoint uses
`action_horizon=4`, so the eval examples use `--replan-steps 4`. The Wan-LoRA
CLI defaults are short smoke defaults; real runs in this README pass
`--wan-lora-num-inference-steps 8` explicitly.

## Smoke Baseline

Use the conv baseline only to verify data loading, losses, checkpointing, and
GPU plumbing quickly:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python train.py \
    --dataset-source synthetic \
    --epochs 2 \
    --batch-size 16 \
    --image-size 32 \
    --num-future-frames 2 \
    --action-horizon 8 \
    --device cuda:0 \
    --output-dir output/smoke_cuda
```

## Tests

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design contract.
