# Wan2.2 + IDM Architecture

This environment follows the Direct Video-Action pattern:

```text
observe current image + task text
-> generate imagined future video with Wan2.2
-> decode the imagined future into actions with a separately trained IDM
-> execute a short action chunk and replan
```

The video model decides the desired visual future. The IDM only translates the
current/future visual transition into executable MetaWorld actions.

## Design Basis

Rhoda's DVA writeup frames robot control as repeated video prediction followed
by inverse dynamics: a causal video model predicts future frames from visual
context, and a separate IDM converts that prediction into robot actions.

1X's world-model writeup uses the same high-level bridge from generated video to
action, and highlights practical concerns we should measure here: generated
video quality versus task success, best-of-N future sampling, inference latency,
and closed-loop replanning.

Wan2.2 is the intended world model for this repo. The official Wan2.2 `TI2V-5B`
path supports text/image-to-video generation via `generate.py --task ti2v-5B`
with `--image`, `--prompt`, and `--ckpt_dir`. We wrap that entrypoint instead
of copying Wan internals into this environment.

Wan2.2 should not receive robot actions or low-dimensional robot state. Its
inputs are the current RGB camera frame and task text. Its output is an imagined
video rollout; the IDM is responsible for turning that visual future into robot
actions.

## Data Contract

The MetaWorld generator writes raw LeRobot keys:

| Key | Shape |
|---|---:|
| `corner.image` | `(H, W, 3)` |
| `corner4.image` | `(H, W, 3)` |
| `gripperPOV.image` | `(H, W, 3)` |
| `observation.state` | `(4,)` |
| `observation.environment_state` | `(39,)` |
| `actions` | `(4,)` |
| `task` | string |

The adapter requests temporal windows through LeRobot `delta_timestamps` and
converts samples to:

| Tensor | Shape | Meaning |
|---|---:|---|
| `current_images` | `(V, 3, H, W)` | current observation view(s) |
| `future_images` | `(K, V, 3, H, W)` | future keyframes for IDM training |
| `future_image_mask` | `(K,)` | valid future-frame positions |
| `state` | `(4,)` | observable robot state |
| `action_chunk` | `(A, 4)` | MetaWorld action chunk |
| `action_mask` | `(A,)` | valid action positions |
| `task_id` | `()` | stable hash of task text |

Defaults are `K=1` future keyframe and `A=32` actions. For the Wan2.2 path,
the default is a single camera view, `corner4.image`, because Wan2.2 produces
one video from one reference image. Multi-view inference should either call
Wan2.2 once per view or train the IDM on a stitched view layout.

## Wan2.2 Contract

Inference inputs:

| Field | Meaning |
|---|---|
| `prompt` | task text, e.g. "Robot manipulation in MetaWorld. Task: ..." |
| `image` | current RGB camera frame |
| `frame_num` | generated clip length, `4n+1` frames |
| `ckpt_dir` | Wan2.2-TI2V-5B checkpoint directory |

Inference output:

| Field | Meaning |
|---|---|
| `mp4` | generated future video from Wan2.2 |
| `future_images` | selected generated frames, converted to `(K, 1, 3, H, W)` for IDM |

The wrapper saves the current frame as an image, calls Wan2.2 with `--image`
and `--prompt`, then samples generated frames for the IDM. If Wan returns the
conditioning frame as frame 0, the wrapper skips it when selecting futures.
Raw Wan2.2 is only valid at `frame_delta=1` in this environment; generated
frame slots are native video steps, so larger dataset source offsets require a
Wan-LoRA fine-tuned from clips exported at that `frame_delta`.

Wan2.2 fine-tuning data is exported for DiffSynth-Studio:

| File | Meaning |
|---|---|
| `metadata.csv` | columns `video,prompt` |
| `videos/*.mp4` | each clip is `[current_frame, future_1, ... future_K]` |
| `images/*.png` | debug copy of the current conditioning frame |
| `captions/*.txt` | debug copy of the prompt |

DiffSynth's Wan2.2-TI2V-5B trainer uses `--extra_inputs input_image`; internally
it sets `input_image = video[0]`. Therefore the current frame must be frame 0 of
the exported training video, and the future frames follow it. The total clip
length must be `4n+1`, so `num_future_frames` should be divisible by 4.

## Wan Action Inference Modes

The mode registry in `world_model/action_modes.py` names the action-side contract
explicitly. `describe_wan_action_modes.py` can render the same specs as JSON or
Markdown for experiment notes.

| Mode | Wan computation | Action input contract |
|---|---|---|
| `decoded_video_idm` | full Wan text+image-to-video generation, future latents, and VAE-decoded future frames | IDM consumes current pixels, decoded future pixels, and current proprio; no reusable Wan memory is exposed |
| `current_wan_prefix_action_expert` | Pi0.5-style frozen Wan current-frame/text hidden-prefix pass runs once; no future-video generation or future latent slots; not native Wan attention KV | flow action expert consumes cached Wan-derived hidden-prefix memory plus current proprio and reuses that memory across action flow denoising steps |
| `partial_wan_prefix_action_expert` (hybrid/research) | target contract is Wan generation or incomplete denoising over future latent slots; the offline prefix-cache producer can use zero/noise placeholders or cached GT/generated future latents in one DiT feature pass | flow action expert consumes reusable Wan-derived prefix/action memory plus current proprio; decoded frames are not controller inputs |

Only `decoded_video_idm` consumes decoded future pixels in the action path.
`current_wan_prefix_action_expert` matches the Pi0.5 pattern most closely: the
frozen Wan current observation/text hidden prefix is computed once, cached, and
reused while the action expert performs flow denoising. The hybrid/research mode
is the intended place to test Wan future-latent generation or incomplete
denoising while exposing reusable Wan-derived memory for the action expert. The
offline prefix-token cache can either add zero/noise future latent placeholders
or join a cached future-latent cache by dataset index and place the cached
future slots into the same one-pass Wan DiT feature extraction. GT Wan VAE
future-latent caches are oracle-only; generated or partial-denoised latent
caches carry their generator provenance in metadata. Live serving still exposes
current-prefix only until this offline hybrid path shows useful signal. In the
current implementation that memory may be learned or projected action-expert
prefix memory from Wan features.

In code, `WanPi05ActionExpert.prepare_action_context(...)` builds an
`ActionDenoisingContext` once per observation/action chunk, and
`forward_with_action_context(...)` reuses that context for every flow step. This
is the action-expert context boundary; it is not native Wan attention KV.

True native Wan attention KV extraction/reuse is not implemented or available in
the current DiffSynth path. Hidden-prefix action modes cache hidden features or
action-expert memory derived from Wan; they are not cached Wan attention KV.

## IDM Contract

IDM training inputs:

| Tensor | Shape | Meaning |
|---|---:|---|
| `current_images` | `(B, V, 3, H, W)` | current camera view(s) |
| `future_images` | `(B, K, V, 3, H, W)` | ground-truth future frames for training, Wan frames for inference |
| `state` | `(B, 4)` | observable MetaWorld robot state |

`state` is current-timestep proprio only. Future proprio/state is oracle-only
and prohibited as controller input because Wan emits no proprio and the live
server can only provide the current observation state.

IDM output:

| Tensor | Shape | Meaning |
|---|---:|---|
| `action_chunk` | `(B, A, 4)` | raw MetaWorld `[dx, dy, dz, gripper]` chunk in dataset scale |

The IDM is trained from scratch on LeRobot action chunks. At inference it sees
the same current frame and proprioceptive state, but its future frames come from
Wan2.2 instead of the dataset. Task text belongs to the world model/prompt path;
the IDM does not condition on `task_id`.

Four IDM layouts are supported for compatibility and ablations, but the active
IDM path is `flow_transformer`:

| `--idm-arch` | Status | Visual encoding |
|---|---|---|
| `stacked` | legacy checkpoint-compatible baseline | stack current and future frames along channels, then encode once |
| `delta` | legacy convolutional baseline | encode current frames, future frames, and explicit `future - current` deltas separately |
| `transformer` | experimental direct regressor | patch-token transformer over current, future, and delta frames |
| `flow_transformer` | active IDM architecture | frozen or learned transition encoder plus noisy action-token transformer velocity head |

The `flow_transformer` layout trains a conditional velocity field over action
chunks: during training it corrupts the target action with sampled noise and
time, predicts the target flow velocity, can add an optional auxiliary endpoint
action loss, then samples actions by integrating from Gaussian action noise at
inference. Multiple sampled action chunks can be averaged with
`idm_flow_num_samples`; eval passes fixed sample noise by default so checkpoint
selection is reproducible. Its visual transition backend is selected with
`--idm-visual-encoder`: `patch` learns patch tokens from pixels; `wan_vae`
freezes the Wan2.2-TI2V-5B VAE and conditions the action DiT on Wan latent video
tokens instead of learned pixel patch embeddings.

The current LingBot-VA-inspired optimization direction keeps this modular
contract: Wan text + current image predicts future frames/latents, and a
separate IDM decodes actions. Loop47/48 are post-Loop46 future-ranking
follow-ups: ranking weight `0.5` fixes the current future-blind gate but hurts
action accuracy badly, while ranking weight `0.05` mostly recovers action error
without meaningful future use. Loop49 then enables Wan latent token prefix
conditioning without ranking loss; it improves internal validation but remains
future-blind on external GT/generated-latent eval. Loop50 adds stronger
LingBot-style future-only GT latent noise and scheduled ranking; it slightly
improves selected-checkpoint MSE but remains future-blind, and the final
checkpoint still does not prefer real futures over repeated/shuffled/noisy
futures. Scalar ranking weight, scheduled ranking, and prefix-token conditioning
are therefore insufficient. Prioritize explicit action-token cross-attention over
Wan latent tokens, visual history tokens, or a candidate-consistency objective
that forces correct action differences between futures without relying on easy
negative artifacts. Do not train the IDM on WM-generated frames unless that
constraint is explicitly changed.

History-conditioned IDMs are supported in live serving with per-connection
state/action history buffers. This is training-faithful only when requests arrive
for every environment step, such as `replan_steps=1`. With larger
`replan_steps`, the server only observes the replanning boundary states/actions;
training-faithful history then requires the client to supply the skipped
per-step history.

## Cached Futures

Wan2.2 futures should be cached before IDM eval loops. The cache manifest maps
each generated future back to the source dataset index and stores:

| File | Meaning |
|---|---|
| `manifest.jsonl` | source index, prompt, paths, generation seed, frame strategy, selected frame indices, total video frames, `dataset_frame_delta`, `source_frame_offsets` |
| `futures/*.pt` | generated future tensor shaped like `future_images` |
| `videos/*.mp4` | debug video for visual inspection |
| `images/*.png` | current conditioning frame |

For Wan caches, `future_frame_strategy` names how `selected_frame_indices` are
chosen from generated-video slots with the conditioning frame at index 0.
`first` selects `[1, 2, ..., K]`. `source_offsets` selects slots matching the
dataset source offsets; for example, `frame_delta=4` and `K=4` selects
`[4, 8, 12, 16]` and requires a 17-frame clip. Raw `wan2_2` caches are
restricted to `frame_delta=1`; use `wan_lora` for larger source-frame offsets.

`eval_pipeline.py --future-source cached` wraps the original dataset and
replaces only `future_images`, preserving the real current frame, robot state,
task metadata, and action target.

`evaluate_future_cache.py` compares cached futures against the dataset futures
when those ground-truth futures are available. It writes aggregate MSE/MAE/PSNR
metrics plus a contact sheet of current, ground-truth future, and cached future
frames so Wan quality can be inspected before IDM training.

`cache_future_rollouts.py --future-source wan_lora` uses DiffSynth-Studio to
load the local Wan2.2 checkpoint plus a trained LoRA once, generates videos for
each sample, and stores only the selected future frames expected by the IDM.
This is the preferred path for evaluating fine-tuned Wan checkpoints because it
separates expensive video generation from repeated IDM experiments.
Use `first` for near-term generated futures, or `source_offsets` when cached
decoded futures should match dataset source offsets. Raw official Wan2.2 is only
valid for `frame_delta=1` in this setup; larger dataset strides require a LoRA
fine-tuned from clips exported at that `frame_delta`.

The strongest current-valid decoded-video modular evidence is
`output/diagnose_loop20_current_valid_cache_recheck_ep16_19_first64/idm_diagnostics.json`
with generated cache `output/loop16_cache_epoch1_f17_steps8_ep16_19_first64`:
`idm_mse=0.06194690754637122` versus mean baseline `0.33408883213996887`, with
`future_blind=false`. The same-family prior GT diagnostic is about
`0.04420376801863313`, so generated video remains the measured bottleneck while
still carrying usable future signal.

Generated Wan latent caches can store pre-VAE-decode latents for latent-IDM
experiments. A partial-denoise run may stop before `num_inference_steps`; cache
metadata records the denoise mode, fraction, completed step count, and requested
step config so full and partial generated-latent evals cannot be mixed silently.
Generated-latent cache result JSON also separates generator load, generation, and
write wall time, with per-row generation time in the manifest.

`compare_wan_latent_caches.py` compares generated latent caches against matched
GT Wan VAE latent caches. The latest matched diagnostic shows the conditioning
latent time slice is nearly identical while the future latent slice carries the
gap; full denoise is closest to GT latents, but IDM action error is still the
dominant robustness/domain-adaptation problem. Loop46 reduces absolute MSE with
history-conditioned cached latents, but remains future-blind and
denoise-insensitive; Loop47/48 show scalar future-ranking loss is not enough by
itself. The next work is stronger latent-token future conditioning rather than
more denoise-count tuning.

## Training And Inference

1. Fine-tune Wan2.2 on text + current-frame conditioned MetaWorld video clips.
2. Train the IDM from scratch on ground-truth current/future frame pairs plus
   LeRobot action chunks.
3. Run Wan2.2 from the current image and task prompt to generate imagined
   future frames.
4. Feed current frames, generated future frames, and proprioceptive state to the IDM.
5. Execute the predicted action chunk in receding horizon and replan.

The current IDM predicts raw action chunks of shape `(A, 4)` for MetaWorld
actions `[dx, dy, dz, gripper]`. Closed-loop execution should clip actions only
at the simulator boundary, not inside the training head.

## Baselines

- `train_idm.py` is the main training path for the separately trained IDM.
- `infer_wan_idm.py` is the Wan2.2 -> IDM inference path.
- `train.py` keeps a small conv video model plus IDM only for smoke testing.

The conv model should not be presented as the target world-model architecture.

## Experiments To Run

- IDM trained on ground-truth futures, evaluated on held-out ground-truth futures.
- IDM evaluated on cached Wan2.2 futures to measure the generated-future gap.
- Best-of-N Wan2.2 sampling with a VLM or heuristic video-quality selector.
- Closed-loop receding-horizon rollouts in MetaWorld.
- Latency profiling: Wan generation time, IDM decode time, and action execution
  coverage per generated rollout.

## References

- Rhoda AI, "Causal Video Models Are Data-Efficient Robot Policy Learners":
  https://www.rhoda.ai/research/direct-video-action
- 1X, "World Model | From Video to Action":
  https://www.1x.tech/discover/world-model-self-learning
- Wan2.2:
  https://github.com/Wan-Video/Wan2.2
