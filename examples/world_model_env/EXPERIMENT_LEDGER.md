# Experiment Loop Ledger

Auditable record of the MetaWorld Wan2.2 + IDM experiment loops.

**Objective** (from [README.md](README.md) / [ARCHITECTURE.md](ARCHITECTURE.md)):
`current camera image + task text -> Wan2.2 imagined future video -> IDM action chunk`,
executed closed-loop in MetaWorld with receding-horizon replanning.

**How to read this.** Each clean loop below lists **Command / Artifact / Eval / Fix /
Outcome**. Every metric is quoted from a JSON file under `output/` (git-ignored — the
artifacts are evidence, not committed). Numbers were read directly from the cited JSON.
Loops that no longer satisfy the current invariants are listed separately as
stale/superseded so they are not mistaken for live results.

Date: 2026-06-08.

---

## Current invariant contract

A loop is **valid / current** only if it satisfies all of:

| Axis | Required value |
|---|---|
| View / resolution | `corner4.image`, `image_size=64` |
| Temporal contract | `frame_delta=1`, `num_future_frames=4`, `action_horizon=4` |
| IDM | active research path is `idm_arch=flow_transformer`, `idm_target_source=ground_truth`, `normalize_actions=true`; `idm_visual_encoder=patch` or experimental frozen `wan_vae`. Historical closed-loop results below used delta checkpoints and are retained as baselines, not the preferred IDM architecture. |
| Train/eval split | canonical comparison split: train episodes **0–15**; eval/diagnose/rank on **held-out episodes 16–19**. This is an assembly-only split, not task-diverse ML45. |
| Checkpoint selection | hardened contiguous split (`split_gap≥1`, enforced in `train_lib.split_dataset`), no adjacent-frame leakage |
| Canonical IDM | `output/metaworld_loop_11_delta_ep0_15_2048_h4_noise004_hardened_split/best_idm_checkpoint.pt` |
| Pixel-cache eval | boundary-correct `evaluate_future_cache.py` (mask-skips invalid futures, pixel-weighted MSE) |

The canonical IDM's held-out reference (real GT futures, ep16–19):
**`idm_mse=0.0479` @ n=64**, **`0.0461` @ n=128** — vs a mean-action baseline of
`~0.334`. Every valid Wan-future ranking below is measured against this same IDM and
reference.

The strongest current-valid decoded-video modular generated-Wan evidence is
`output/diagnose_loop20_current_valid_cache_recheck_ep16_19_first64/idm_diagnostics.json`
using cache `output/loop16_cache_epoch1_f17_steps8_ep16_19_first64`:
**`idm_mse=0.06194690754637122`**, mean baseline
`0.33408883213996887`, and `future_blind=false`. The same-family prior GT
reference in `output/diagnose_loop20_noise012_state_norm_gt_ep16_19_first64/idm_diagnostics.json`
is **`0.04420376801863313`**.

### Wan action-mode reporting contract
- `decoded_video_idm`: decoded video generation -> IDM. This path generates/decodes
  future pixels and feeds them to the IDM/action decoder.
- `current_wan_prefix_action_expert`: Pi0.5-style frozen Wan hidden-prefix memory
  run once from the current frame/text -> action expert. This reports current-only
  Wan hidden/prefix tokens as action-expert inputs, not native Wan attention KV.
- `partial_wan_prefix_action_expert`: partial/noisy hidden-prefix Wan -> action expert.
  This reports hidden/prefix tokens from partial or noisy future-latent Wan passes, not
  decoded future pixels as action inputs.
- `joint_softmax_prefix_cache_action_expert`: compact Wan-prefix action expert with
  learned reusable prefix memory and joint prefix/action softmax over Wan hidden-prefix
  tokens. This is a cache-style action-memory benchmark, not faithful Pi0.5 KV-cache
  semantics or native Wan attention KV reuse.
- None of these reports claim true Wan attention KV-cache reuse. Hidden-prefix
  tokens are not cached Wan attention KV, and true native Wan KV extraction/reuse
  is not implemented or available in the current DiffSynth path.

### Latest matched Wan action-mode findings
- **Run-once Wan-prefix audit (2026-06-08)** `current_wan_prefix_action_expert`
  is the Pi0.5-style path: serving calls the current image/text prefix encoder once
  per observation/action chunk, then `WanPi05ActionExpert.prepare_action_context(...)`
  builds reusable action-expert context for all flow denoising steps. This path does
  not sample or decode future video. `decoded_video_idm` remains the explicit
  world-model path that generates/decodes future frames and feeds those pixels to the
  IDM. `partial_wan_prefix_action_expert` is the hybrid research path with future
  latent slots/memory and optional debug decode, but the current prefix-cache producer
  only inserts zero/noise future slots and runs one DiT feature pass; it does not yet
  bridge real incomplete Wan denoising into the prefix-token action-expert schema.
  None of these paths currently expose true native Wan attention KV; the reusable
  memory is Wan-derived prefix/action-expert memory. Focused verification on this audit
  passed locally:
  `uv run pytest -q tests/test_action_modes.py tests/test_pi05_wan_action_expert.py tests/test_wan_kv_cache.py tests/test_wan_prefix_cache.py tests/test_infer_pi05_wan_action_expert.py tests/test_serve_world_model.py tests/test_run_wan_action_mode_matrix.py`
  -> `233 passed`. Claude's broader independent focused verification also passed
  `290 passed, 1 skipped` across action modes, prefix cache, KV scaffold, Pi0.5 Wan
  action expert, serving, and inference; the skip is the opt-in real Wan DiT smoke.
- **Matched eval root**
  `examples/world_model_env/output/wan_action_modes_matched_ep16_23_spe32_h4_epoch4_steps4_seed7`
  covers held-out episodes 16-23 with `samples_per_episode=32`, `action_horizon=4`,
  Wan epoch 4, 4 inference steps, and seed 7.
- **Patch-token cross-attention flow IDM (latest)** Training artifact
  `output/idm_flow_patch_crossattn_futuredelta_gt_ep0_15_spe64_h4_seed7_no_rank/metrics.json`
  ran 80 epochs on train episodes 0-15 with `samples_per_episode=64`, `h=4`, patch visual
  encoder, `flow_transformer`, `visual_token_conditioning_mode=cross_attention`,
  `representation=future_delta`, and no ranking loss. Best internal epoch 71 scored
  `idm_mse=0.04453461678301702`, `idm_smooth_l1=0.020557327348677837`; checkpoint:
  `output/idm_flow_patch_crossattn_futuredelta_gt_ep0_15_spe64_h4_seed7_no_rank/best_idm_checkpoint.pt`.
  Held-out GT eval/diagnose on episodes 16-23 with `samples_per_episode=32`:
  `output/idm_flow_patch_crossattn_futuredelta_gt_ep16_23_eval_no_rank/eval_metrics.json`
  and
  `output/diagnose_idm_flow_patch_crossattn_futuredelta_gt_ep16_23_no_rank/idm_diagnostics.json`
  scored `idm_mse=0.04860944184474647`, `idm_smooth_l1=0.022621081094257534`, mean
  baseline `0.3542788624763489`, `future_blind=false`, current-repeated output delta MSE
  `0.04055627138586715`. Generated Wan future diagnostic on the same held-out sample set
  used cache
  `output/wan_action_modes_matched_ep16_23_spe32_h4_epoch4_steps4_seed7/decoded_video_idm/future_cache`
  and artifact
  `output/diagnose_idm_flow_patch_crossattn_futuredelta_wan_ep16_23_no_rank/idm_diagnostics.json`,
  scoring `idm_mse=0.07012095977552235`, `idm_smooth_l1=0.032311801449395716`, the same
  mean baseline `0.3542788624763489`, `future_blind=false`, and current-repeated output
  delta MSE `0.021656778932083398`. Interpretation: explicit patch visual-token
  cross-attention plus `future_delta` is no longer future-blind and improves broad
  held-out GT action decoding; the generated-Wan gap remains about `0.02151` MSE vs GT
  on the same split. Ranking loss was expensive and not helpful in the earlier
  interrupted comparison, so this no-ranking result is the cleaner checkpoint. This is
  still offline IDM/action-decoding evidence, not closed-loop task success.
- **Prior 256-sample comparison / smoke** All sample sets matched and none used true
  Wan attention KV-cache reuse. `decoded_video_idm` scored
  `dataset_action_mse=0.160416`, while same-split GT scored `0.160341`; because
  generated and GT futures are basically identical for this checkpoint, keep this
  as matched action-mode smoke/comparison evidence rather than the best evidence
  of a generated-video bottleneck. Current prefix-only scored `0.583027`; partial
  prefix-only scored `7.784041`.
- **New ablations** Current prefix+state trained on the small broad train2 spe4 cache
  scored `0.470518`; partial prefix+state scored `6.231323`; original-scale loss scored
  `0.848437`; cross-attention scored `0.761041`.
- **Same-task caveat** Current prefix+state trained on episodes 0-15 with `spe32` and
  evaluated on episodes 16-23 scored `0.034215`, but this is the same task
  (`pick up the nut and place it onto the peg`) and should not be treated as broad
  generalization.
- **Current-prefix action-expert mini sweep (2026-06-08)** On the matched
  ep16-23/spe32/h4 cache, the existing unweighted
  `current_wan_prefix_state` seed109 h512/l6 run scored
  `dataset_action_mse=0.03421521083929278`, per-dim
  `[0.0012778148568080403, 0.007111521435509357, 0.127188858400749, 0.0012826486641047252]`.
  The new best weighted-loss run,
  `output/pi05_wan_action_expert_dit_ep0_15_spe32_eval16_23_spe32_h4_prefixstate_origscale_weighted_seed109_e300_h512_l6/checkpoint.pt`,
  evaluated at
  `output/wan_action_modes_matched_ep16_23_spe32_h4_epoch4_steps8_seed7/current_wan_prefix_state_origscale_weighted_ep0_15_train_action_expert/eval/eval_metrics.json`,
  scored `dataset_action_mse=0.02837979388687737`,
  `smooth_l1=0.011098821650104144`, per-dim
  `[0.0024209396265056256, 0.017416839776929528, 0.09343164173890724, 0.0002497544051670835]`,
  17.06% lower MSE than unweighted. A weighted h512/l6 seed110 repeat at
  `output/wan_action_modes_matched_ep16_23_spe32_h4_epoch4_steps8_seed7/current_wan_prefix_state_origscale_weighted_seed110_ep0_15_train_action_expert/eval/eval_metrics.json`
  scored `dataset_action_mse=0.028556008913602455`, per-dim
  `[0.009191132618664319, 0.013089761064572408, 0.09165461940754964, 0.00028852256362345025]`,
  so the result is robust/tied. A weighted h768/l8 seed109 capacity check regressed
  (`val_model_sample_mse=0.03495163097977638`), so larger capacity was not better on
  this small cache. Decoded-video IDM steps8 remains
  `dataset_action_mse=0.07290960592217743`; current-prefix is the strongest path here,
  but still not native Wan KV. Verification from main: focused `ruff` passed for
  Wan/action-mode files; the latest independent action-mode audit pytest passed
  (`290 passed, 1 skipped`, opt-in real Wan DiT smoke skipped).
- **Broad train2 weighted-loss follow-up (2026-06-08)** Broad current-prefix cache
  `output/pi05_wan_dit_prefix_cache_train2_spe16_h4` contains `1408` rows
  (`44` tasks x `2` episodes/task x `16` samples/episode). The existing broad
  unweighted checkpoint
  `output/pi05_wan_action_expert_dit_train2_spe16_eval16_23_spe32_h4_prefixstate_norm_seed109_e300_h512_l6/checkpoint.pt`
  scored assembly ep16-23/spe32 `dataset_action_mse=0.16360315680503845`, per-dim
  `[0.08032508194446564, 0.21418392658233643, 0.33094385266304016, 0.02895982377231121]`.
  Original-scale weighted broad runs at `lr=5e-4` regressed (`seed109`
  `0.22003406286239624`, `seed110` `0.2854838967323303`); lowering LR fixed the
  optimizer/gradient-scale issue on assembly eval, with `lr=1e-4` seed109 at
  `output/wan_action_modes_matched_ep16_23_spe32_h4_epoch4_steps4_seed7/current_wan_prefix_state_broad_origscale_weighted_seed109_lr1e4/eval/eval_metrics.json`
  scoring `dataset_action_mse=0.11086914266422845`, `smooth_l1=0.051577073838518436`,
  per-dim `[0.04726141992060684, 0.10948591568226787, 0.26494987252570523, 0.021779362528333897]`
  (`lr=5e-5` seed109 `0.12076468265967545`, `lr=2e-4` seed109
  `0.1756603717803955`, `lr=1e-4` seed110 `0.15081612336481573`). A new task-diverse
  eval cache, `output/pi05_wan_dit_prefix_cache_eval44_third_ep_spe16_h4`, contains
  `704` rows (`44` held-out third episodes x `16` samples/episode), `prefix_token_count=3`,
  mode `current_wan_prefix_action_expert`; `spe32` failed because episode `594` has
  only `28` valid windows. On eval44, the unweighted broad checkpoint is best by MSE:
  `output/pi05_wan_dit_prefix_cache_eval44_third_ep_spe16_h4/current_wan_prefix_state_broad_unweighted_seed109/eval/eval_metrics.json`
  scored `dataset_action_mse=2.86207081467861`, `smooth_l1=0.3931520895445747`,
  per-dim `[1.2302685628639567, 1.7873916216135826, 8.350951180201635, 0.07967189403526555]`,
  versus mean baseline `6.453713593332177`. Weighted `lr=1e-4` seed109 is worse by MSE
  but better by smooth L1 (`2.922197008854678`, `0.31739909972523983`, per-dim
  `[0.9556522186055424, 1.6256873443575663, 9.042371812728256, 0.06507665972734805]`);
  seed110 eval44 MSE was `3.2990759737700572`. Interpretation: original-scale
  weighting helps assembly-only held-out eval when LR is lowered, but does not improve
  aggregate task-diverse MSE because the dim-2 tail dominates eval44. Valid claim:
  broad current-prefix beats mean baseline on task-diverse offline action MSE, but this
  is not closed-loop success and not native Wan KV.
- **Robust action-loss weighting follow-up (2026-06-08)** Added explicit
  `clipped_original_scale` and `normalized_original_scale` options in
  `train_pi05_wan_action_expert.py`, with focused tests and metadata recording.
  Verification passed with focused `ruff` and pytest (`34 passed`). On assembly
  ep16-23/spe32, clipped max4 at `lr=1e-4` nearly tied the best raw weighted result:
  `output/wan_action_modes_matched_ep16_23_spe32_h4_epoch4_steps4_seed7/current_wan_prefix_state_broad_clipped_origscale_max4_seed109_lr1e4/eval/eval_metrics.json`
  scored `dataset_action_mse=0.11069199220615875`, `smooth_l1=0.05286687007901354`,
  per-dim `[0.05527931377119688, 0.09168435465130634, 0.27045151641762566, 0.02535278398450616]`;
  clipped max2 scored `0.11510111391544342`, and normalized at `lr=1e-4` scored
  `0.12496356666088104`. On eval44, none of the robust weighted variants beat the
  unweighted broad checkpoint by MSE: clipped max4 scored `3.0777057663433847`,
  clipped max2 scored `2.9575072196826655`, normalized `lr=1e-4` scored
  `2.8984930976194914`, versus unweighted `2.86207081467861`. They do improve
  smooth L1 (`~0.318-0.321` vs unweighted `0.393`), so they reduce many smaller
  errors while still losing to large dim-2 tails. Zero-noise task diagnostics in
  `output/pi05_wan_dit_prefix_cache_eval44_third_ep_spe16_h4/diagnostics/`
  show the worst eval44 tasks are handle/contact-heavy tasks such as
  `press the handle down`, `pull the handle sideways`, `press the handle down sideways`,
  and `pull the handle up`, with the dominant errors in action dim 2. Interpretation:
  clipped/normalized weighting is useful infrastructure and improves assembly-only
  action decoding, but broad task-diverse MSE still needs a tail-robust objective or
  per-task balancing rather than global action-scale weighting alone.
- **Per-task action-normalization follow-up (2026-06-08)** Added explicit
  `--action-normalization-scope per_task` support for cached-prefix action-expert
  training/eval/infer. Per-task checkpoints store task-specific mean/std tensors and
  standalone eval now requires matching task labels instead of falling back silently.
  Verification passed with focused `ruff` and pytest (`42 passed`). On assembly
  ep16-23/spe32, per-task normalization at `lr=1e-4` produced the best current broad
  action-expert result: seed109
  `output/wan_action_modes_matched_ep16_23_spe32_h4_epoch4_steps4_seed7/current_wan_prefix_state_broad_pertask_norm_seed109_lr1e4/eval/eval_metrics.json`
  scored `dataset_action_mse=0.10214722784722179`, `smooth_l1=0.04456001181742712`,
  per-dim `[0.03538499495482712, 0.09161698748535976, 0.27798000412521245, 0.003606924823487777]`;
  seed110 improved assembly further to `0.0964680352128555`, `smooth_l1=0.043158545562463436`,
  per-dim `[0.032485059182251266, 0.09090294193232501, 0.2589850142499874, 0.003499125486858404]`.
  The higher `lr=5e-4` per-task run scored `0.13343936204910278`, so `lr=1e-4` is the
  better setting. On eval44, however, per-task normalization did not beat global
  unweighted MSE: seed109 scored `2.916586852627758`, seed110 scored `3.360497322045793`,
  versus global unweighted `2.86207081467861`. Interpretation: per-task normalization
  is useful and helps the assembly task strongly, but it increases broad task-diverse
  variance; eval44 still needs explicit tail handling for contact-heavy dim-2 tasks
  rather than more normalization alone.
- **Prepared action-context + cross-attention follow-up (2026-06-08)** Added
  `ActionDenoisingContext` so `WanPi05ActionExpert.prepare_action_context(...)`
  runs once per observation/action chunk and `forward_with_action_context(...)`
  reuses that action-expert context for every flow step across all decoder arches.
  This is not native Wan attention KV. Focused verification passed with `ruff`
  plus pytest (`110 passed`), and `eval_pi05_wan_action_expert.py` now records
  `eval_elapsed_ms`, `eval_ms_per_sample`, and `eval_device`. On the broad train2
  `spe16` cache, context-cross-attention at `lr=5e-4` regressed on matched assembly
  eval (`dataset_action_mse=0.24433133006095886`). Lowering to `lr=1e-4` improved
  assembly strongly:
  `output/pi05_wan_action_expert_dit_train2_spe16_eval16_23_spe32_h4_prefixstate_crossattn_norm_seed109_e300_h512_l6_lr1e4_context_refactor/checkpoint.pt`
  evaluated at
  `output/wan_action_modes_matched_ep16_23_spe32_h4_epoch4_steps4_seed7/current_wan_prefix_state_crossattn_seed109_lr1e4_context_refactor/eval/eval_metrics.json`
  scored `dataset_action_mse=0.129132573704121`, `smooth_l1=0.06151751357348739`,
  per-dim `[0.09989926830758172, 0.15219022247913078, 0.2571002572493869, 0.0073405467803846405]`,
  versus encoder timing rerun `0.15759993256395738` and mean baseline `0.35207056286014865`.
  However, eval44 remains the guardrail: the same cross-attention checkpoint scored
  `dataset_action_mse=3.044008037220362`, `smooth_l1=0.31073525562664706`, per-dim
  `[1.0660264776899817, 1.1921706178390135, 9.864939611670136, 0.05289544168231696]`,
  versus encoder timing rerun `2.836926372864771`, `smooth_l1=0.3924286687019146`.
  Interpretation: cross-attention plus prepared context is a useful architecture knob
  and improves the seen assembly eval, but it still worsens task-diverse MSE by the
  dim-2 tail while reducing many smaller errors. Treat it as another tail-robustness
  candidate, not a broad solution yet. Follow-up global action-weighting variants confirm
  that scale weighting alone is not the broad fix. Cross-attention plus normalized
  original-scale weighting at `lr=1e-4`:
  `output/pi05_wan_action_expert_dit_train2_spe16_eval16_23_spe32_h4_prefixstate_crossattn_normed_origscale_seed109_e300_h512_l6_lr1e4/checkpoint.pt`
  scored assembly
  `output/wan_action_modes_matched_ep16_23_spe32_h4_epoch4_steps4_seed7/current_wan_prefix_state_crossattn_normed_origscale_seed109_lr1e4/eval/eval_metrics.json`
  `dataset_action_mse=0.11432412792763325`, `smooth_l1=0.05516456888885875`, per-dim
  `[0.052211113872086046, 0.10126948926077486, 0.28670229654979706, 0.017113612027874963]`,
  but eval44
  `output/pi05_wan_dit_prefix_cache_eval44_third_ep_spe16_h4/current_wan_prefix_state_crossattn_normed_origscale_seed109_lr1e4/eval/eval_metrics.json`
  worsened to `dataset_action_mse=3.1771873075631305`, `smooth_l1=0.32494674449830346`,
  per-dim `[1.2585758959232458, 1.5479915659734025, 9.825349912302613, 0.07683185605326194]`.
  Cross-attention plus clipped original-scale max2 scored assembly `0.12169105869567216`
  and eval44 `3.1973346858822564`. Interpretation: weighted cross-attention sharpens
  the seen assembly slice, but task-diverse eval still needs a task/group-tail objective
  such as worst-task/top-k/CVaR loss, not another global dimension scale.
- **Task-tail action loss objective (2026-06-08)** Added per-sample flow-loss
  components and train-time `action_loss_aggregation` modes: `mean`,
  `task_balanced`, and `task_cvar`. The default `mean` preserves the old
  valid-action-element weighted scalar loss, while task modes aggregate
  per-task valid-element means. Focused verification passed with `ruff` and
  pytest (`120 passed`). On the broad train2 `spe16` cache, the encoder
  `task_cvar` guardrail regressed badly on assembly
  (`dataset_action_mse=0.2446256726816681`) and eval44
  (`3.1848190530046567`), so the tail objective does not rescue the encoder.
  Context-cross-attention showed the useful signal. With `task_cvar_fraction=0.25`
  and `task_cvar_weight=0.5`, assembly worsened to `0.15356916484963684`, but
  eval44 improved to `2.6773954024546645`, `smooth_l1=0.3283850720694381`,
  per-dim `[1.188665413514672, 1.485545660500526, 7.973757059098852, 0.061613476704608955]`.
  A softer `task_cvar_weight=0.25` recovered assembly to `0.11987760178250705`
  while keeping eval44 improved at `2.736983701034551`, `smooth_l1=0.3264332348203167`,
  per-dim `[1.2057641264589856, 1.6149406535537947, 8.071696766067284, 0.055533258058140356]`.
  Sweeping softer/narrower CVaR found the current best tradeoff at
  `task_cvar_fraction=0.25`, `task_cvar_weight=0.125`: assembly
  `0.10923198056585488`, `smooth_l1=0.05306226462308076`, and eval44
  `2.7449897180896197`, `smooth_l1=0.310970172029035`, per-dim
  `[1.0459175328799062, 1.3555138698423939, 8.522198397946546, 0.05632907168963185]`.
  Narrowing the tail to `fraction=0.125`, `weight=0.25` nearly matched the
  aggressive eval44 MSE (`2.6784279858563083`) but badly hurt assembly
  (`0.16255788419677364`).
  Pure `task_balanced` gave the best assembly result so far,
  `0.11409747750285552`, but worsened eval44 to `3.2126523639738838`.
  Interpretation: the CVaR/top-task term is the part that improves task-diverse
  generalization by shrinking the dim-2 tail; pure task balancing is a seen-slice
  optimizer. Next code-level knobs should add a CVaR weight schedule or short
  stage-2 CVaR fine-tune, because full-run CVaR is useful but too blunt.
- **Scheduled task-CVaR follow-up (2026-06-08)** Added optional
  `task_cvar_start_weight` and `task_cvar_warmup_epochs`; when unset, behavior
  is unchanged, and when set the effective CVaR weight linearly ramps from the
  start weight to `task_cvar_weight`. Focused tests passed (`76 passed`) plus
  adjacent eval/infer/cache/action-mode tests (`49 passed`). Scheduling confirmed
  that late strong tail pressure can materially improve eval44, but the tradeoff
  is non-monotonic. Fixed `fraction=0.25`, `weight=0.125` remains the best
  assembly/eval44 balance (`assembly=0.10923198056585488`, `eval44=2.7449897180896197`).
  A scheduled `0 -> 0.5` ramp over `250` epochs is the best eval44 result so far:
  `assembly=0.1485315818350671`, `eval44=2.2408411010693996`, `smooth_l1=0.31217562756196016`,
  per-dim `[1.0644450356639814, 1.399157029014891, 6.403707238908484, 0.0960551006902425]`.
  Later ramps did not dominate: warmup `275` gave `assembly=0.11806870631903099`,
  `eval44=2.8376270364230156`; warmup `290` gave `assembly=0.15262702482539886`,
  `eval44=2.7598302249001576`. A `0 -> 0.25` ramp over `150` epochs failed to
  improve eval44 (`3.093996160643518`). Interpretation: the useful direction is
  not generic warmup; it is late, high-intensity tail correction. Next concrete
  step should be explicit two-stage fine-tuning from the best assembly checkpoint
  or EMA hard-task weights, not more plain warmup sweeps.
- **Strict init-checkpoint / stage-2 fine-tune follow-up (2026-06-08)** Added
  strict `init_checkpoint` support in `train_pi05_wan_action_expert.py` for
  stage-2 `WanPi05ActionExpert` fine-tuning, with strict `model_kwargs` checks
  and action-normalization checks. Focused verification passed locally: `ruff`
  passed and `pytest tests/test_pi05_wan_action_expert.py` passed
  (`80 passed, 37 warnings`). Initial stage-2 runs failed before training because
  checkpoint normalization tensors were on CUDA while current stats were on CPU;
  this was fixed by CPU-normalizing tensors and loading checkpoints with
  `map_location='cpu'`. Stage-2 from fixed CVaR `fraction=0.25`, `weight=0.125`,
  `epochs=50`, `lr=2.5e-5`, target CVaR `fraction=0.25`, `weight=0.5` scored
  assembly/matched `0.09810187667608261`, eval44 `3.052627113340599`,
  `smooth_l1=0.30337769106198337`, per-dim
  `[0.8056289495410692, 1.2918601375277936, 10.059423143971543, 0.05359622232199244]`.
  Stage-2 from task-balanced, `epochs=50`, `lr=2.5e-5`, target CVaR
  `fraction=0.25`, `weight=0.5` scored assembly `0.09651067852973938`,
  eval44 `2.9192938708210594`, `smooth_l1=0.29326518681677466`, per-dim
  `[0.9796963149073534, 1.230028295491605, 9.424369398857248, 0.043081474028031023]`.
  A gentler stage-2 from fixed CVaR `fraction=0.25`, `weight=0.125`,
  `epochs=10`, `lr=1e-5` scored assembly `0.09480413049459457`, eval44
  `3.0329165327178926`, `smooth_l1=0.3017616038005907`, per-dim
  `[0.9554659076018278, 1.2815963521844116, 9.845372970447599, 0.04923090063773125]`.
  A longer stage-2 from task-balanced, `epochs=150`, `lr=2.5e-5` scored assembly
  `0.09354670345783234`, eval44 `2.888337028545386`,
  `smooth_l1=0.2923155241486811`, per-dim
  `[0.9446282578874481, 1.2235276774946529, 9.336543005453128, 0.048649173346315105]`.
  Interpretation: stage-2 fine-tuning on the seen train2 cache strongly improves
  matched assembly MSE (best so far `~0.0935`) but does not beat eval44 fixed-CVaR
  `2.7449897180896197` or scheduled-CVaR best `2.2408411010693996`. This looks
  like seen-cache adaptation/forgetting, not robust task-diverse generalization;
  avoid more plain train2-only stage-2 cycles unless adding task-diverse/hard-task
  data, heldout-aware weighting, or generated-video IDM.
- **Pi0.5 timestep-embedding ablation (2026-06-08)** Added
  `--timestep-embedding-style {diffusion,pi05}` to the cached-prefix action
  expert. The default `diffusion` style preserves the previous geometric
  `max_period=10000` embedding. The new `pi05` style matches OpenPI's Pi0.5
  action-flow posemb periods (`min_period=4e-3`, `max_period=4.0`) with
  float32 angle math for bf16/fp16 safety. Old init checkpoints that lack the
  new model-kwarg validate as `diffusion`; old normalization metadata that lacks
  `scope` validates as `global`. Focused verification passed:
  `pytest tests/test_pi05_wan_action_expert.py tests/test_infer_pi05_wan_action_expert.py tests/test_eval_pi05_wan_action_expert.py`
  -> `105 passed`; focused `ruff` also passed. A backward-compat smoke got past
  the new timestep/scope defaults; a later failure on action-normalization
  mean/std mismatch is separate strict checkpoint-stat validation.

  | Style / seed | matched val sample MSE | eval44 MSE | eval44 smooth L1 |
  |---|---:|---:|---:|
  | diffusion seed109 | `0.16360315680503845` | `2.86207081467861` | `0.3931520895445747` |
  | pi05 seed109 | `0.12022587656974792` | `2.586401891731129` | `0.3532252471404238` |
  | diffusion seed110 | `0.14128677546977997` | `3.2570897162503005` | `0.3725163556952383` |
  | pi05 seed110 | `0.18592765927314758` | `3.4635794603253824` | `0.3608427415056917` |

  Interpretation: Pi0.5 timestep embeddings are a useful parity knob and clearly
  improve seed109, but they regress seed110 by MSE. Keep `diffusion` as the default
  and treat `pi05` as an ablation/selection option rather than a universal fix.
  The next credible use is pairing it with task-tail/CVaR objectives or selecting
  across seeds on a held-out validation set.
- **Next hybrid Wan/action-memory implementation decision (2026-06-08)** A
  read-only sidecar architecture review agreed with the local audit: the next
  credible hybrid experiment is not native Wan KV and not another prefix-pooling
  ablation. First build a GT-oracle real-future-latent prefix producer that
  places cached Wan VAE latents from dataset futures into the existing DiT-hidden
  future latent slots, then train/evaluate the existing action expert on the
  unchanged row schema. This answers the highest-information question first:
  whether the current prefix-action architecture can use real future latent
  content at all. Only if GT future-latent prefixes improve action MSE should we
  spend compute on generated or partially-denoised future-latent prefixes.
  Native Wan attention KV remains out of scope for the next step because the
  current DiffSynth path does not expose reusable KV, and Wan self-attention KV
  is timestep/noise/latent-dependent. Concrete implementation slice:
  `world_model/wan_dit_prefix_encoder.py` should accept validated optional
  future latents for slots `1:`, `cache_pi05_wan_prefix_tokens.py` should join a
  future-latent cache by dataset index and write honest oracle/generated
  provenance metadata, and `tests/test_wan_dit_prefix_encoder.py` plus
  `tests/test_wan_prefix_cache.py` should cover shape validation, cache joins,
  missing/duplicate rows, and metadata. Serving should stay current-prefix-only
  until the offline cache/eval path shows useful signal.
- **GT/generated future-latent prefix bridge implementation (2026-06-08)**
  Implemented the offline producer side of the decision above. `cache_pi05_wan_prefix_tokens.py`
  now accepts `--future-latent-cache-dir` for `prefix_backend='dit_hidden'` with
  `dit_num_latent_frames>1`, joins cached Wan VAE or generated Wan latent rows by
  `dataset_index`, slices off latent slot 0, and passes the remaining future slots
  into `FrozenDiffSynthWanDiTCurrentPrefixEncoder.encode_prefix(...)`. The action
  expert row schema is unchanged. Metadata records whether the cache contains GT
  future latents, the future-slot cache source/path, generated-cache provenance
  when present, and `native_wan_attention_kv_cache=false`. The path is offline-only;
  serving still exposes current-prefix action experts only. Focused verification:
  `pytest tests/test_wan_dit_prefix_encoder.py tests/test_wan_prefix_cache.py`
  -> `48 passed, 1 skipped`; focused `ruff` also passed. No action MSE result is
  claimed yet. Next step is to build a GT-oracle future-latent train/eval cache
  and compare against current-prefix and decoded-video baselines.
- **GT-oracle future-latent prefix cache materialization (2026-06-08)** The new
  bridge successfully produced real Wan DiT hidden-prefix caches using cached
  ground-truth Wan VAE future latents. Train cache:
  `output/pi05_wan_dit_gt_future_prefix_cache_diverse44_train2_spe4_h4`
  (`352` rows: 44 tasks, two train episodes/task, four samples/episode). Eval cache:
  `output/pi05_wan_dit_gt_future_prefix_cache_diverse44_eval2_3_spe2_h4`
  (`176` rows: 44 tasks, two held-out episodes/task, two samples/episode). Both
  caches have `prefix_token_count=3`, `prefix_dim=3072`,
  `contains_future_ground_truth_latents=true`, `future_latent_slot_count=1`,
  `wan_action_mode=partial_wan_prefix_action_expert`, and
  `native_wan_attention_kv_cache=false`. This confirms the bridge works end-to-end
  with the frozen Wan2.2-5B DiT feature path. It is still an oracle feature-cache
  construction result, not an action-MSE or closed-loop result; next step is to train
  the action expert on this cache and compare against current-prefix baselines.
- **GT-oracle future-latent prefix action expert (2026-06-08)** Artifact:
  `output/pi05_wan_action_expert_gt_future_prefix_diverse44_train2_spe4_eval2_3_spe2_h4_prefixstate_norm_seed109_e300_h512_l6/metrics.json`.
  This trained the existing flow-matching action expert on the GT-oracle
  future-latent prefix cache above with `conditioning_mode=wan_prefix_state`,
  `normalize_actions=true`, `epochs=300`, `hidden_dim=512`, `num_layers=6`,
  `num_heads=8`, `sample_steps=16`, and seed `109`. Held-out eval on the matching
  two-episode/task GT-future prefix cache scored `val_model_zero_noise_mse=2.4864377975463867`
  versus `val_mean_action_mse=6.035114765167236`. Interpretation: the action expert
  can use real future-latent Wan prefix content in this oracle setting. This is
  positive architecture signal, not a generated-Wan or closed-loop result. Next
  comparison is a same-split current-prefix control, because the older current-prefix
  small-cache baselines used a different held-out episode/sample mix.
- **Same-split current-prefix control (2026-06-08)** Built matching current-only Wan
  DiT prefix caches for the same train/eval rows:
  `output/pi05_wan_dit_current_prefix_cache_diverse44_train2_spe4_h4`
  (`352` rows) and
  `output/pi05_wan_dit_current_prefix_cache_diverse44_eval2_3_spe2_h4`
  (`176` rows). Artifact:
  `output/pi05_wan_action_expert_current_prefix_diverse44_train2_spe4_eval2_3_spe2_h4_prefixstate_norm_seed109_e300_h512_l6/metrics.json`.
  With the same action-expert hyperparameters as the GT-future run, current-only
  prefixes scored `val_model_zero_noise_mse=3.0317862033843994` against the same
  `val_mean_action_mse=6.035114765167236`. GT-oracle future-latent prefixes therefore
  reduce same-split MSE by about `18%` relative to current-only Wan prefix features
  (`2.4864` vs `3.0318`). Interpretation: future latent slots provide useful oracle
  information to the action expert; the next high-value experiment is to replace
  oracle slots with generated or partially-denoised Wan latents and measure the gap.
- **Narrow generated-latent prefix smoke (2026-06-08)** While preparing the generated
  smoke, the bridge was fixed to accept legacy generated latent cache metadata where
  `dataset_config` contains only `source` and the remaining dataset identity lives at
  top level. Focused verification passed:
  `pytest tests/test_wan_prefix_cache.py` -> `33 passed`; precommit on the touched
  files also passed. Using existing generated Wan latent caches for episodes `[2, 3]`
  with `samples_per_episode=2`, plus a freshly built matching GT Wan VAE latent cache,
  the oracle-trained action expert above was evaluated on four-row prefix caches:

  | Eval prefix cache | dataset action MSE | smooth L1 |
  |---|---:|---:|
  | GT future latents, `output/pi05_wan_dit_gt_future_prefix_cache_ep2_3_spe2_h4_generated_smoke` | `0.060996670148117514` | `0.030498335074058757` |
  | generated full 4/4, `output/pi05_wan_dit_generated_future_prefix_cache_ep2_3_spe2_h4_full_s4_smoke` | `0.49319802502505866` | `0.2375960252596063` |
  | generated partial 2/4, `output/pi05_wan_dit_generated_future_prefix_cache_ep2_3_spe2_h4_partial_s2of4_smoke` | `9.30703801342866` | `1.7251916243343661` |

  The same four-row mean-action baseline is `0.10328263645612615`, so the generated
  full cache is already worse than mean action and partial 2/4 is much worse. Treat
  this as a narrow negative generated-latent smoke, not a 44-task benchmark. It
  suggests the next generated-prefix work should inspect latent alignment/scale and
  task prompts before spending compute on a broad generated cache. Follow-up
  diagnostic artifact:
  `output/diagnose_generated_prefix_alignment_ep2_3_spe2_smoke/metrics.json`.
  Generated full 4/4 has mean future-latent cosine `0.7967510968446732` vs GT and
  mean prefix-token cosine `0.9460775256156921`, while generated partial 2/4 has
  mean future-latent cosine `0.105997359380126` and mean prefix-token cosine
  `0.8191804587841034`. Interpretation: partial latents are plainly misaligned; full
  generated latents are closer in prefix space but still action-brittle on this tiny
  sample, so action-sensitivity and prompt/task alignment need inspection before scale-up.
- **Future-slot timestep ablation (2026-06-08)** Tested whether generated latents were
  failing because the DiT prefix bridge used `dit_timestep=500` even for denoised
  future slots. Built matching GT-oracle t0 prefix caches:
  `output/pi05_wan_dit_gt_future_prefix_cache_diverse44_train2_spe4_h4_t0`
  (`352` rows) and
  `output/pi05_wan_dit_gt_future_prefix_cache_diverse44_eval2_3_spe2_h4_t0`
  (`176` rows), then trained the same action expert:
  `output/pi05_wan_action_expert_gt_future_prefix_diverse44_train2_spe4_eval2_3_spe2_h4_t0_prefixstate_norm_seed109_e300_h512_l6/metrics.json`.
  Broad held-out GT t0 eval scored `val_model_zero_noise_mse=2.674940586090088`,
  worse than the original `t=500` oracle (`2.4864377975463867`). Narrow four-row t0
  evals with the t0-trained checkpoint:

  | Eval prefix cache | dataset action MSE | smooth L1 |
  |---|---:|---:|
  | GT future latents, `output/pi05_wan_dit_gt_future_prefix_cache_ep2_3_spe2_h4_t0_smoke` | `0.028039244527820695` | `0.014019622263910347` |
  | generated full 4/4, `output/pi05_wan_dit_generated_future_prefix_cache_ep2_3_spe2_h4_full_s4_t0_smoke` | `0.6436163760988791` | `0.2890291179116442` |
  | generated partial 2/4, `output/pi05_wan_dit_generated_future_prefix_cache_ep2_3_spe2_h4_partial_s2of4_t0_smoke` | `3.8600330526337836` | `1.0124491977738381` |

  Alignment artifact:
  `output/diagnose_generated_prefix_alignment_ep2_3_spe2_t0_smoke/metrics.json`.
  Generated full 4/4 has mean t0 prefix-token cosine `0.9703584909439087` versus GT
  but still poor action MSE. Interpretation: the simple timestep mismatch hypothesis
  is negative. The generated full cache can be close in pooled Wan prefix space and
  still be action-brittle, so the next fix should target action-head robustness or
  generated-latent quality/conditioning rather than just changing DiT timestep.
- **GT-future prefix cross-attention decoder ablation (2026-06-08)** Tested whether
  the plain encoder action head was too brittle to small Wan-prefix perturbations.
  Artifact:
  `output/pi05_wan_action_expert_gt_future_prefix_diverse44_train2_spe4_eval2_3_spe2_h4_prefixstate_crossattn_norm_seed109_e300_h512_l6/metrics.json`.
  This used the same GT-future prefix train/eval caches and hyperparameters as the
  original oracle run, except `decoder_arch=context_cross_attention`. Broad held-out
  GT eval scored `val_model_zero_noise_mse=2.5952887535095215`, worse than the plain
  encoder's `2.4864377975463867`. Narrow four-row evals with the cross-attention
  checkpoint:

  | Eval prefix cache | dataset action MSE | smooth L1 |
  |---|---:|---:|
  | GT future latents, `output/pi05_wan_dit_gt_future_prefix_cache_ep2_3_spe2_h4_generated_smoke` | `0.1842133637968459` | `0.0916446701145051` |
  | generated full 4/4, `output/pi05_wan_dit_generated_future_prefix_cache_ep2_3_spe2_h4_full_s4_smoke` | `0.7155092835265839` | `0.3397753688932558` |
  | generated partial 2/4, `output/pi05_wan_dit_generated_future_prefix_cache_ep2_3_spe2_h4_partial_s2of4_smoke` | `8.494649018047465` | `1.651319411010427` |

  Interpretation: swapping to cross-attention does not improve generated-latent
  robustness here; it worsens both broad GT and narrow generated-full scores. Keep the
  plain encoder decoder as the stronger baseline for this prefix-token action expert.
- **GT-future prefix dropout robustness ablation (2026-06-08)** Tested whether ordinary
  model dropout makes the plain encoder action expert less brittle to generated-prefix
  perturbations. Artifact:
  `output/pi05_wan_action_expert_gt_future_prefix_diverse44_train2_spe4_eval2_3_spe2_h4_prefixstate_norm_dropout01_seed109_e300_h512_l6/metrics.json`.
  This used the original GT-future prefix caches and action-expert hyperparameters with
  `dropout=0.1`. Broad held-out GT eval scored
  `val_model_zero_noise_mse=2.5729944705963135`, worse than the no-dropout encoder
  baseline (`2.4864377975463867`). Narrow four-row evals:

  | Eval prefix cache | dataset action MSE | smooth L1 |
  |---|---:|---:|
  | GT future latents, `output/pi05_wan_dit_gt_future_prefix_cache_ep2_3_spe2_h4_generated_smoke` | `0.10155734801042866` | `0.05077867400521434` |
  | generated full 4/4, `output/pi05_wan_dit_generated_future_prefix_cache_ep2_3_spe2_h4_full_s4_smoke` | `0.40410634469190204` | `0.1826267271275436` |
  | generated partial 2/4, `output/pi05_wan_dit_generated_future_prefix_cache_ep2_3_spe2_h4_partial_s2of4_smoke` | `9.287772455684122` | `1.8240064680686825` |

  Interpretation: dropout improves generated full versus the no-dropout encoder
  (`0.4041` vs `0.4932`) but hurts broad GT and is still far worse than the four-row
  mean-action baseline (`0.1033`). This is weak positive evidence for explicit
  prefix-perturbation robustness, not a usable fix.
- **GT-future prefix dropout 0.2 ablation (2026-06-08)** Follow-up strength sweep for
  ordinary model dropout. Artifact:
  `output/pi05_wan_action_expert_gt_future_prefix_diverse44_train2_spe4_eval2_3_spe2_h4_prefixstate_norm_dropout02_seed109_e300_h512_l6/metrics.json`.
  With `dropout=0.2`, broad held-out GT eval improved to
  `val_model_zero_noise_mse=2.3751871585845947`, better than the no-dropout encoder
  (`2.4864377975463867`) and dropout 0.1 (`2.5729944705963135`). Narrow four-row evals:

  | Eval prefix cache | dataset action MSE | smooth L1 |
  |---|---:|---:|
  | GT future latents, `output/pi05_wan_dit_gt_future_prefix_cache_ep2_3_spe2_h4_generated_smoke` | `0.14954093352698963` | `0.07477046676349483` |
  | generated full 4/4, `output/pi05_wan_dit_generated_future_prefix_cache_ep2_3_spe2_h4_full_s4_smoke` | `0.6060374949838444` | `0.28842560712601006` |
  | generated partial 2/4, `output/pi05_wan_dit_generated_future_prefix_cache_ep2_3_spe2_h4_partial_s2of4_smoke` | `13.667292885949822` | `2.3486179676650405` |

  Interpretation: dropout 0.2 is the best broad GT-oracle action-MSE checkpoint so
  far, but it worsens generated-latent transfer. Ordinary model dropout is not enough
  for generated prefixes; the next robustness step needs generated-aware prefix
  perturbation or direct generated-prefix training/eval.
- **Generated-prefix failure audit (2026-06-08)** A read-only independent audit checked
  the four-row generated-prefix smoke against row provenance, prompt/task labels,
  cache ordering, future-slot extraction, action normalization, and eval plumbing. The
  failure appears real rather than a row-join or prompt mismatch: all three prefix
  caches score the same actions with the same normalization, the generated rows match
  the GT rows by dataset/task/episode/frame, and the prefix builder consumes the single
  future latent slot that the alignment diagnostic measures. The audit's root-cause
  ranking is: (1) the action expert is overfit to clean GT-oracle future-prefix
  manifolds and is brittle off-manifold; (2) partial generated latents are still noisy
  intermediate denoising states, not clean future predictions; (3) full generated
  future-latent quality is imperfect, including one low-magnitude/mode-collapsed row in
  the four-sample smoke. Recommended next experiments are generated-aware prefix
  perturbation or direct generated-prefix training, storing an `x0` estimate for partial
  latent caches instead of an intermediate noisy sample, an offline GT-to-generated
  prefix interpolation sensitivity sweep, and hardening the prefix/future-cache join
  with episode/frame parity assertions before scaling.
- **GT-future prefix-noise 0.05 ablation (2026-06-08)** Implemented training-only
  Gaussian prefix-token perturbation for `train_pi05_wan_action_expert.py`
  (`--prefix-noise-std`, default `0.0`; optional deterministic
  `--prefix-noise-seed`) and validated the implementation with focused tests plus full
  world-model checks (`ruff check .`, `pre-commit run --all-files`, and
  `pytest -q tests` -> `778 passed, 1 skipped`). Training artifact:
  `output/pi05_wan_action_expert_gt_future_prefix_diverse44_train2_spe4_eval2_3_spe2_h4_prefixstate_norm_prefixnoise005_seed109_e300_h512_l6/metrics.json`.
  With `prefix_noise_std=0.05`, the broad GT-oracle held-out eval improved to
  `val_model_zero_noise_mse=2.223419189453125`, better than no-dropout (`2.4864`)
  and dropout 0.2 (`2.3752`). Narrow four-row evals with this checkpoint:

  | Eval prefix cache | dataset action MSE | smooth L1 |
  |---|---:|---:|
  | GT future latents, `output/pi05_wan_dit_gt_future_prefix_cache_ep2_3_spe2_h4_generated_smoke` | `0.11246896191326297` | `0.05623448095663149` |
  | generated full 4/4, `output/pi05_wan_dit_generated_future_prefix_cache_ep2_3_spe2_h4_full_s4_smoke` | `0.7481807743387987` | `0.3428631495305358` |
  | generated partial 2/4, `output/pi05_wan_dit_generated_future_prefix_cache_ep2_3_spe2_h4_partial_s2of4_smoke` | `10.156493500037111` | `1.9092146272157966` |

  Interpretation: raw isotropic prefix-token noise is useful regularization for broad
  GT-oracle prefixes, but it does not mimic generated-prefix distribution shift and
  worsens generated full/partial transfer on the narrow smoke. The next robustness test
  should be generated-aware perturbation, direct mixed GT+generated prefix training, or
  an interpolation sensitivity sweep rather than larger raw Gaussian noise.
- **GT-future prefix-noise 0.01 ablation (2026-06-08)** A gentler raw Gaussian
  prefix-token noise run was also negative. Artifact:
  `output/pi05_wan_action_expert_gt_future_prefix_diverse44_train2_spe4_eval2_3_spe2_h4_prefixstate_norm_prefixnoise001_seed109_e300_h512_l6/metrics.json`.
  Broad GT-oracle held-out eval scored
  `val_model_zero_noise_mse=2.6350057125091553`, worse than no-noise (`2.4864`),
  dropout 0.2 (`2.3752`), and prefix-noise 0.05 (`2.2234`). Narrow four-row evals:

  | Eval prefix cache | dataset action MSE | smooth L1 |
  |---|---:|---:|
  | GT future latents, `output/pi05_wan_dit_gt_future_prefix_cache_ep2_3_spe2_h4_generated_smoke` | `0.4173261418311921` | `0.1920781599748746` |
  | generated full 4/4, `output/pi05_wan_dit_generated_future_prefix_cache_ep2_3_spe2_h4_full_s4_smoke` | `0.564293572983766` | `0.26687716465882677` |
  | generated partial 2/4, `output/pi05_wan_dit_generated_future_prefix_cache_ep2_3_spe2_h4_partial_s2of4_smoke` | `10.98552744278135` | `2.1334287293193084` |

  Interpretation: smaller raw isotropic noise does not recover generated transfer and
  also hurts broad GT. Do not keep sweeping this scalar without changing the
  perturbation distribution.
- **GT-to-generated prefix interpolation sweep (2026-06-08)** Added
  `interpolate_wan_prefix_caches.py`, a diagnostic utility that validates matched
  cache row alignment and writes blended prefix-token caches
  `(1 - alpha) * GT + alpha * generated`. Focused verification passed:
  `pytest tests/test_interpolate_wan_prefix_caches.py` -> `4 passed`, focused `ruff`
  and pre-commit passed. On the same four-row GT-vs-generated-full smoke, evaluating
  the original no-noise GT-oracle action expert gave:

  | alpha toward generated full | dataset action MSE | smooth L1 |
  |---:|---:|---:|
  | `0.00` | `0.060996670148117514` | `0.030498335074058757` |
  | `0.25` | `0.03343255243623729` | `0.01671627621811864` |
  | `0.50` | `0.06578365645778102` | `0.03289182822889051` |
  | `0.75` | `0.2102514107064981` | `0.10512570535324905` |
  | `1.00` | `0.49319802502505866` | `0.2375960252596063` |

  Interpretation: this is not an immediate off-manifold cliff. A small generated-prefix
  move can improve this tiny held-out assembly slice, and a 50% blend is still near GT.
  The large error appears after the generated contribution dominates. This supports
  training on generated-aware perturbations or mixed GT/generated prefixes, and it makes
  a broader interpolation sweep worth running before investing in larger generated
  caches.
- **Direct mixed GT+generated prefix training (2026-06-08)** Added multi-cache training
  support to `train_pi05_wan_action_expert.py` via repeatable `--extra-cache-path`, with
  deterministic train-cache concatenation, shape checks, cross-cache `wan_action_mode`
  validation, and metadata recording (`train_cache_paths`, `train_cache_sample_counts`,
  `checkpoint["train_caches"]`). Focused verification passed:
  `pytest tests/test_pi05_wan_action_expert.py` -> `97 passed`; focused `ruff` and
  pre-commit passed. The interpolation utility was also hardened after review: blended
  caches now mark `contains_future_ground_truth_latents` true whenever any
  positively-weighted source contains GT future latents, set `cache_kind` consistently,
  and reject manifests without strong alignment metadata; focused interpolation tests
  now pass (`8 passed`). Matching generated-full train/eval prefix caches were
  materialized with Wan2.2 TI2V 5B + LoRA epoch 4 (`4` inference steps, base seed
  `710`):
  `output/pi05_wan_dit_generated_future_prefix_cache_diverse44_train2_spe4_h4_full_s4`
  (`352` rows) and
  `output/pi05_wan_dit_generated_future_prefix_cache_diverse44_eval2_3_spe2_h4_full_s4`
  (`176` rows). Training artifact:
  `output/pi05_wan_action_expert_gt_plus_generated_full_prefix_diverse44_train2_spe4_eval2_3_spe2_h4_prefixstate_norm_seed109_e300_h512_l6/metrics.json`.
  It trained on GT future prefixes plus generated-full future prefixes (`704` total
  train rows) and evaluated on the same held-out GT eval cache as the oracle runs:
  `val_model_zero_noise_mse=2.3415050506591797` vs GT-only no-noise `2.4864`, dropout
  0.2 `2.3752`, and prefix-noise 0.05 `2.2234`.

  | Eval prefix cache / checkpoint | dataset action MSE | smooth L1 |
  |---|---:|---:|
  | four-row GT, mixed checkpoint | `0.054310574865572225` | `0.027155287432786113` |
  | four-row generated full, GT-only checkpoint | `0.49319802502505866` | `0.2375960252596063` |
  | four-row generated full, mixed checkpoint | `0.36170012572505394` | `0.1754124531985617` |
  | four-row generated partial 2/4, mixed checkpoint | `3.19447654103855` | `0.9307870165477307` |
  | broad generated full eval, GT-only checkpoint | `6.3751840588749955` | `1.004933437262866` |
  | broad generated full eval, mixed checkpoint | `4.1327067751560636` | `0.544244186503822` |

  Broad generated-full mean-action baseline is `5.941208772387123`, so GT-only on
  generated-full is worse than mean action, while mixed GT+generated training beats mean
  action and cuts broad generated-full MSE by about `35%` (`6.3752` -> `4.1327`). This
  is the strongest evidence so far that direct generated-prefix exposure improves
  transfer. It does not solve partial/noisy generated latents, and it still trails the
  broad GT-oracle result; next useful work is mix-ratio/source-aware sampling and
  improved generated future quality rather than plain Gaussian prefix noise.
- **Broad train2 result** Current prefix+state trained on the 44-task train2 `spe16`
  cache (`1408` rows) scored `0.163603` on the matched ep16-23 eval, roughly tied with
  the matched decoded-video smoke checkpoint and much better than the mean baseline
  `0.352071`. This remains an offline IDM/action MSE result rather than closed-loop
  task-success evidence.
- **Partial-cache fix** Deterministic per-sample noisy future-slot generation was added
  for partial prefix caches. Focused verification passed with `ruff` plus pytest:
  `38 passed, 1 skipped`.
- **Fixed partial-prefix result** Partial caches now use deterministic per-sample noisy
  future-slot placeholders keyed by dataset index. The 44-task train2 `spe16` train cache
  at `output/pi05_wan_dit_partial_cache_train2_spe16_h4_lat2_noise_perrow_seed0`
  contains `1408` rows; the matched eval cache at
  `output/wan_action_modes_matched_ep16_23_spe32_h4_epoch4_steps4_seed7/partial_wan_prefix_fixed_noise_action_expert/prefix_cache`
  contains `256` rows. Checkpoint
  `output/pi05_wan_partial_action_expert_train2_spe16_eval16_23_spe32_h4_lat2_noise_perrow_prefixstate_norm_seed109_e300_h512_l6/checkpoint.pt`
  evaluated at
  `output/wan_action_modes_matched_ep16_23_spe32_h4_epoch4_steps4_seed7/partial_wan_prefix_fixed_noise_train2_spe16_action_expert/eval/eval_metrics.json`
  scored `dataset_action_mse=0.693690` vs mean baseline `0.352071`. This is much better
  than the old partial prefix-only `7.784041` and partial+state `6.231323`, but remains
  worse than broad current-prefix+state `0.163603` and the matched decoded-video smoke
  checkpoint `0.160416`.
  Broad eval44 follow-up generated the matching partial cache
  `output/pi05_wan_dit_partial_cache_eval44_third_ep_spe16_h4_lat2_noise_perrow_seed0`
  (`704` rows, one deterministic per-sample noisy future-latent slot,
  `wan_backbone_runs_per_observation=1`, no native Wan KV) and evaluated the same
  checkpoint at
  `output/pi05_wan_dit_partial_cache_eval44_third_ep_spe16_h4_lat2_noise_perrow_seed0/partial_wan_prefix_state_train2_spe16_eval44/eval_metrics.json`.
  It scored `dataset_action_mse=6.221486033513933`, `smooth_l1=0.8595603408744797`,
  per-dim `[3.6648001445546483, 4.370102852485178, 16.60190796345154, 0.24913317356436201]`,
  barely better than the broad mean-action baseline `6.453713593332176` and far worse
  than broad current-prefix (`~2.86` MSE).
  Conclusion: deterministic noise fixed a cache artifact, but noisy partial future slots
  remain harmful for this action expert. This is not yet a faithful real partial-denoise
  comparison; the missing bridge is a producer that converts incomplete Wan denoising
  features/latents into the same prefix-token cache schema. Until that exists, prioritize
  current-prefix+state and Pi0.5 parity fixes, or build the real partial-denoise prefix
  bridge before reading more into the hybrid result.
- **Suffix-prefix-cache result** `decoder_arch='suffix_prefix_cache'` uses action-expert
  prefix memory with suffix-only denoising over Wan hidden-prefix tokens. This is not true
  DiffSynth/Wan attention KV reuse. It trained on the broad current-prefix 44-task train2
  `spe16` cache at `output/pi05_wan_dit_prefix_cache_train2_spe16_h4` and evaluated against
  `output/wan_action_modes_matched_ep16_23_spe32_h4_epoch4_steps4_seed7/current_wan_prefix_action_expert/prefix_cache`.
  Checkpoint:
  `output/pi05_wan_action_expert_dit_train2_spe16_eval16_23_spe32_h4_suffix_prefix_cache_prefixstate_norm_seed109_e300_h512_l6/checkpoint.pt`.
  Eval artifact:
  `output/wan_action_modes_matched_ep16_23_spe32_h4_epoch4_steps4_seed7/current_wan_suffix_prefix_cache_train2_spe16_action_expert/eval/eval_metrics.json`.
  It scored `dataset_action_mse=0.195745` vs mean baseline `0.352071`. This proves the
  cache-aware suffix path trains and evals end-to-end and beats baseline, but it is
  currently worse than encoder-style current-prefix+state `0.163603` and the matched
  decoded-video smoke checkpoint `0.160416`; next work should tune the suffix-cache
  architecture rather than treat it as already superior. Verification already passed
  with `ruff`, focused pytest
  (`57 passed`), and real train/eval completion.
- **Joint-softmax prefix-cache ablations** `decoder_arch='joint_softmax_prefix_cache'`
  benchmarks a compact Wan-prefix action expert with learned reusable prefix memory and
  joint prefix/action softmax. The local cache creates learned action-expert prefix K/V
  from updated prefix tokens and lacks Pi0.5 positional/RoPE suffix offsets, so do not
  present it as faithful Pi0.5 KV-cache semantics or native Wan attention KV reuse. Both
  variants use the matched eval root
  `examples/world_model_env/output/wan_action_modes_matched_ep16_23_spe32_h4_epoch4_steps4_seed7`.
  Additive checkpoint
  `examples/world_model_env/output/pi05_wan_action_expert_dit_train2_spe16_eval16_23_spe32_h4_joint_softmax_prefix_cache_prefixstate_norm_seed109_e300_h512_l6/checkpoint.pt`
  evaluated at
  `examples/world_model_env/output/wan_action_modes_matched_ep16_23_spe32_h4_epoch4_steps4_seed7/current_wan_joint_softmax_prefix_cache_train2_spe16_action_expert/eval/eval_metrics.json`
  scored `dataset_action_mse=0.2850127292055413` vs mean baseline
  `0.3520705628601486`. FiLM checkpoint
  `examples/world_model_env/output/pi05_wan_action_expert_dit_train2_spe16_eval16_23_spe32_h4_joint_softmax_prefix_cache_film_prefixstate_norm_seed109_e300_h512_l6/checkpoint.pt`
  evaluated at
  `examples/world_model_env/output/wan_action_modes_matched_ep16_23_spe32_h4_epoch4_steps4_seed7/current_wan_joint_softmax_prefix_cache_film_train2_spe16_action_expert/eval/eval_metrics.json`
  scored `dataset_action_mse=0.1957647576873236` vs the same baseline. FiLM
  dramatically improves joint-softmax over the poor additive variant, but the cache-style
  memory path still does not close the gap to encoder-style current-prefix+state
  `0.163603` or the matched decoded-video smoke checkpoint `0.160416`; it is roughly
  tied with suffix-prefix-cache `0.195745` and slightly behind suffix-prefix-cache
  FiLM `0.192989`.
- **Tokenpool4 richer Wan-prefix ablation** This tests the hypothesis that the 3
  mean-pooled Wan DiT tokens used above are too lossy. The train cache
  `examples/world_model_env/output/pi05_wan_dit_prefix_cache_train2_spe16_h4_tokenpool4_dim3072`
  contains `1408` rows with `prefix_token_count=12`, `hidden_pool='token_pool'`,
  `tokens_per_layer=4`, selected layers `[0, 14, 29]`, and
  `wan_action_mode='current_wan_prefix_action_expert'`. The eval cache
  `examples/world_model_env/output/wan_action_modes_matched_ep16_23_spe32_h4_epoch4_steps4_seed7/current_wan_prefix_action_expert_tokenpool4/prefix_cache`
  contains `256` rows with the same matched eval fingerprint as the previous
  episodes 16-23, `samples_per_episode=32` comparison. The encoder tokenpool
  checkpoint
  `examples/world_model_env/output/pi05_wan_action_expert_dit_train2_spe16_eval16_23_spe32_h4_tokenpool4_dim3072_prefixstate_norm_seed109_e300_h512_l6/checkpoint.pt`
  evaluated at
  `examples/world_model_env/output/wan_action_modes_matched_ep16_23_spe32_h4_epoch4_steps4_seed7/current_wan_prefix_state_tokenpool4_train2_spe16_action_expert/eval/eval_metrics.json`
  scored `dataset_action_mse=0.16935493326086287` vs mean baseline
  `0.3520705628601486`; random-noise training eval was
  `0.25265935560067493 +/- 0.013205588798421306`. The joint-softmax FiLM
  tokenpool checkpoint
  `examples/world_model_env/output/pi05_wan_action_expert_dit_train2_spe16_eval16_23_spe32_h4_tokenpool4_dim3072_joint_softmax_prefix_cache_film_prefixstate_norm_seed109_e300_h512_l6/checkpoint.pt`
  evaluated at
  `examples/world_model_env/output/wan_action_modes_matched_ep16_23_spe32_h4_epoch4_steps4_seed7/current_wan_joint_softmax_prefix_cache_film_tokenpool4_train2_spe16_action_expert/eval/eval_metrics.json`
  scored `dataset_action_mse=0.1894089382359423` vs the same baseline;
  random-noise training eval was `0.33137119313081104 +/- 0.02478261287859776`.
  Tokenpool4 improved the cache-style joint-softmax FiLM decoder (`0.195765` ->
  `0.189409`), so richer prefix tokens help the action-memory path a little. It
  did not improve the simple encoder zero-noise result (`0.163603` -> `0.169355`),
  though random-noise training eval improved vs the prior encoder (`~0.270` ->
  `0.253`). It still does not beat the matched decoded-video smoke checkpoint
  `0.160416` or the best 3-token encoder zero-noise result `0.163603`. Keep the
  caveat that this is a decoder/prefix-feature ablation under
  `current_wan_prefix_action_expert`, not
  native Wan KV reuse.

### Dataset metadata correction — task-diverse sampling is required
- **Finding** `brandonyang/metaworld_ml45` currently reports `4332` episodes and `44`
  task strings in LeRobot metadata. Episode IDs are grouped by task: episode `0..99` is
  assembly (`pick up the nut and place it onto the peg`), episode `100` starts basketball,
  and `reach the goal position` is task id `38`, episodes `3743..3842`.
- **Impact** The historical `episodes 0..15` train and `16..19` eval split is a valid
  held-out-demo split for the assembly task, but it is not ML45 task-diverse. Any broad
  MetaWorld claim needs explicit balanced task sampling.
- **Fix** `DatasetConfig.samples_per_episode` adds deterministic, evenly spaced sampling
  from valid non-terminal windows in each selected LeRobot episode and rejects use with
  `max_samples`. The flag is exposed on train/cache/diagnose/eval inspection CLIs, and
  cache resume now rejects any existing output dir whose `dataset_config` does not exactly
  match the requested split before it skips rows or writes files. This prevents balanced
  logical indices from being silently relabeled across different episode/sample settings.
- **Verification** Focused sampler/cache/validation tests passed (`83 passed`),
  `ruff check .` passed, and full `examples/world_model_env` tests passed:
  `250 passed, 14 warnings`.
- **Next split shape** A cheap task-diverse smoke uses a small fixed number of demos per
  task, for example the first 2 train episodes and next 1 eval episode from each task,
  combined with balanced frame sampling per episode. Plain `--max-samples` is not
  sufficient because it truncates the ordered LeRobot dataset.

### Temporal-contract code fix — Wan generated indices vs dataset offsets
- **Issue** Wan cache/export/serve validation previously conflated generated-video frame
  indices with dataset source-frame offsets. That is harmless at `frame_delta=1`, but for
  `frame_delta>1` the correct Wan-LoRA selected frames are generated-video slots
  `[1, 2, ..., K]`; the dataset cadence is recorded separately as
  `source_frame_offsets=[delta, 2delta, ...]`.
- **Fix** `world_model.data` now exposes both contracts explicitly:
  `expected_wan_selected_frame_indices()` returns `[1..K]`, while
  `expected_wan_source_frame_offsets()` records source offsets. Wan cache config and
  manifest rows now store `dataset_frame_delta` and `source_frame_offsets`; ranking,
  cached-dataset loading, live serving, and export tests validate both. Raw base Wan2.2
  is rejected for `frame_delta>1` because it is not finetuned on the subsampled MetaWorld
  Wan export.
- **Verification**
  ```bash
  UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
    tests/test_future_cache.py tests/test_validate_wan_lora.py \
    tests/test_export_wan_dataset.py tests/test_serve_world_model.py \
    tests/test_wan22.py tests/test_diffsynth_wan.py tests/test_infer_wan_idm.py
  # 122 passed, 1 warning

  UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .
  # All checks passed

  UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
  # 225 passed, 12 warnings
  ```

---

## Valid current-invariant loops

### Loop 11 — hardened IDM retrain (canonical IDM)
- **Command**
  ```bash
  uv run python train_idm.py \
    --dataset-source lerobot --repo-id brandonyang/metaworld_ml45 \
    --image-keys corner4.image --episodes 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
    --max-samples 2048 --image-size 64 --frame-delta 1 \
    --num-future-frames 4 --action-horizon 4 \
    --idm-arch delta --idm-future-noise-std 0.04 \
    --epochs 40 --batch-size 64 --learning-rate 1e-4 \
    --early-stopping-patience 8 --early-stopping-min-delta 0.0005 \
    --device cuda:0 \
    --output-dir output/metaworld_loop_11_delta_ep0_15_2048_h4_noise004_hardened_split
  # split_gap=1 is the train_lib default and is applied automatically (no CLI flag).
  ```
- **Artifact** `output/metaworld_loop_11_delta_ep0_15_2048_h4_noise004_hardened_split/`
  → `best_idm_checkpoint.pt` (12.1 MB), `idm_checkpoint.pt`, `metrics.json`.
- **Eval** (`metrics.json`): 40 epochs, `stopped_early=false`; best epoch 39,
  internal-val `idm_mse=0.0388`; final epoch 40 `idm_mse=0.0510`. Held-out GT
  decodability (`diagnose_loop11_gt_ep16_19_first64/idm_diagnostics.json`):
  `idm_mse=0.0479`, `smooth_l1=0.0230`, mean-action baseline `0.3341`.
  Future-sensitivity: shuffling futures raises error to `0.0990` (~2×), zeroing to
  `0.3775`, current-repeated to `0.5660` → the IDM genuinely reads the future frames.
- **Fix** Retrain after the train/eval split was hardened in `train_lib.split_dataset`
  (contiguous split + `split_gap`), removing the adjacent-frame leakage that made prior
  checkpoint selection optimistic. Aligned the temporal contract to `nff=4, ah=4`.
- **Outcome** Canonical IDM for all downstream eval/ranking. Per-dim GT MSE
  `[dx 0.076, dy 0.049, dz 0.061, grip 0.005]`.

### Loop 12 — all-epoch Wan-LoRA ranking with the new IDM
- **Command**
  ```bash
  uv run python validate_wan_lora.py rank \
    --idm-checkpoint output/metaworld_loop_11_delta_ep0_15_2048_h4_noise004_hardened_split/best_idm_checkpoint.pt \
    --cache-dirs output/wan_lora_cache_ep16_19_64_128_lora3_e5_epoch{0,1,2,3,4}_steps8_first4 \
    --labels lora3_e5_epoch0 lora3_e5_epoch1 lora3_e5_epoch2 lora3_e5_epoch3 lora3_e5_epoch4 \
    --dataset-source lerobot --repo-id brandonyang/metaworld_ml45 --image-key corner4.image \
    --episodes 16 17 18 19 --max-samples 64 --frame-delta 1 \
    --rank-by idm_decodability_gap --device cuda:0 \
    --output-dir output/metaworld_loop_12_lora3_all_epochs_ep16_19_new_idm_rank
  ```
- **Artifact** `output/metaworld_loop_12_lora3_all_epochs_ep16_19_new_idm_rank/ranking_summary.json`
  (+ per-epoch `pixel/future_cache_metrics.json`, contact sheets).
- **Eval** (held-out ep16–19, n=64, GT ref `idm_mse=0.0479`):

  | epoch | idm_mse | decodability_gap | future_mse | PSNR |
  |---|---|---|---|---|
  | 1 | **0.0619** | **0.0141** | 0.00144 | 28.42 |
  | 3 | 0.0756 | 0.0277 | **0.00129** | 28.89 |
  | 4 | 0.0757 | 0.0278 | 0.00130 | 28.86 |
  | 2 | 0.0960 | 0.0482 | 0.00135 | 28.71 |
  | 0 | 0.1242 | 0.0764 | 0.00170 | 27.70 |

- **Fix** Re-ranked every Wan-LoRA epoch against the hardened IDM on held-out episodes,
  on both pixel and action-decodability axes (the older rankings used a pre-Loop-11 IDM).
- **Outcome** **Pixel-best (epoch 3) ≠ decodability-best (epoch 1)** — the dual-axis
  selector matters: the lowest `future_mse` checkpoint is not the most action-decodable
  one. World-model checkpoint selection should rank by `idm_decodability_gap`.

### First-128 ranking / diagnostic — scale check on held-out episodes
- **Command**
  ```bash
  uv run python diagnose_idm.py \
    --checkpoint output/metaworld_loop_11_delta_ep0_15_2048_h4_noise004_hardened_split/best_idm_checkpoint.pt \
    --cached-future-dir output/wan_lora_cache_ep16_19_128_128_lora3_e5_epoch3_steps8_first4 \
    --dataset-source lerobot --episodes 16 17 18 19 --max-samples 128 \
    --image-size 64 --frame-delta 1 --num-future-frames 4 --action-horizon 4 \
    --device cuda:0 --output-dir output/diagnose_loop11_wan_epoch3_ep16_19_first128
  # ranking counterpart: wan_lora_ranking_lora3_e5_epoch3_ep16_19_first128_new_idm/
  ```
- **Artifact** `output/diagnose_loop11_wan_epoch3_ep16_19_first128/idm_diagnostics.json`,
  `output/wan_lora_ranking_lora3_e5_epoch3_ep16_19_first128_new_idm/ranking_summary.json`,
  `output/diagnose_loop11_gt_ep16_19_first64/idm_diagnostics.json`.
- **Eval** Wan-LoRA epoch3 @ n=128: `idm_mse=0.0751` (identical in both diagnostic and
  ranking JSON), GT ref `idm_mse=0.0461`, `decodability_gap=0.0290`. Compared to n=64
  (gap `0.0277`) the gap is stable as sample count doubles. Per-dim MSE at n=128
  `[dx 0.049, dy 0.035, dz 0.210, grip 0.006]`: **dz dominates the Wan-future error**
  (vs `0.061` on GT futures). Shuffled-future sensitivity `0.167` (~2.2× real).
- **Fix** Confirmed the n=64 ranking is not a small-sample artifact, and surfaced the
  episode-boundary case handled by the pixel-eval fix below.
- **Outcome** Loop 12's ordering holds at larger n; the residual Wan→action gap is
  concentrated in the z-axis.

### Pixel-eval boundary fix — `evaluate_future_cache.py`
- **Command** (runs inside the rank pixel pass, or standalone)
  ```bash
  uv run python evaluate_future_cache.py \
    --cache-dir output/wan_lora_cache_ep16_19_128_128_lora3_e5_epoch3_steps8_first4 \
    --dataset-source lerobot --episodes 16 17 18 19 --max-samples 128 \
    --image-size 64 --frame-delta 1 --num-future-frames 4 --action-horizon 4 \
    --output-dir output/.../pixel
  ```
- **Artifact** `.../lora3_e5_epoch3_ep16_19_first128/pixel/future_cache_metrics.json`.
- **Eval** `num_total_samples=128`, `num_skipped_samples=1`
  (`cache_index=86`, reason `no_valid_future_frames`), `num_samples=127`,
  pixel-weighted `future_mse=0.00131`, `PSNR=28.83`. The IDM eval scores all 128 (it
  consumes cached futures and never needs GT future pixels), so the 128-vs-127 split is
  expected and now explicit rather than a silent miscount.
- **Fix** Pixel eval now (1) reads `future_image_mask` and skips samples with no valid
  GT future frame at episode boundaries, recording them in `skipped_samples`;
  (2) aggregates MSE/MAE pixel-weighted (sum of per-sample error × pixels ÷ total
  pixels) instead of mean-of-means; (3) fails loudly if *all* samples are skipped.
- **Outcome** Pixel metrics are boundary-correct and reconcile with IDM sample counts.
  Backed by `tests/test_future_cache.py::test_evaluate_future_cache_skips_samples_without_valid_future_frames`
  and `::test_evaluate_future_cache_fails_when_all_samples_without_valid_future_frames` (green).

### Repeat-current closed-loop smoke — `serve_world_model.py`
- **Command**
  ```bash
  # Terminal 1 (this env): serve the IDM as an openpi websocket policy.
  uv run python serve_world_model.py \
    --idm-checkpoint output/metaworld_loop_11_delta_ep0_15_2048_h4_noise004_hardened_split/best_idm_checkpoint.pt \
    --image-keys observation/image \
    --future-provider repeat_current --allow-repeat-current \
    --device cuda:0 --port 8000
  # Terminal 2 (metaworld env): drive closed-loop against the server.
  MUJOCO_GL=egl uv run examples/metaworld/main.py --env_name reach-v3 --port 8000
  ```
- **Artifact** No `output/` artifact; the contract is covered by the test suite
  (`tests/test_serve_world_model.py`).
- **Eval** Full client/server round-trip with the real `openpi_client`: metadata
  handshake (`policy=world_model_idm`, `future_provider=repeat_current`,
  `image_keys=[observation/image]`), batched `(batch, action_horizon, action_dim)`
  actions through msgpack, and server-side errors propagated to the client. Provider
  output is validated at the server boundary (tensor type, shape, floating dtype,
  finiteness, `[0,1]` range). **58 tests pass** (`test_serve_world_model.py` +
  `test_future_cache.py`) in 4.41s.
- **Fix** `repeat_current` is non-physical, so it is gated behind an explicit
  `--allow-repeat-current` flag (`Args.allow_repeat_current=False` by default; `main()`
  rejects it otherwise) and named in server metadata — it can never be served silently.
- **Outcome** Closed-loop **plumbing** is verified end-to-end and Wan-free. This is a
  contract/smoke test only — **not** a task-success result (see Gaps).

### Real Wan-LoRA direct policy smoke — `wan_lora` provider
- **Artifact** `output/real_wan_lora_policy_smoke_direct_refined/summary.json`
  and `contact_sheet.png` (800×160).
- **Eval** Direct policy invocation loaded real `/tmp/wan2.2-ti2v-5b` and fused
  `output/wan_metaworld_ep0_15_1424_128_lora3_e5/epoch-3.safetensors`. Metadata
  includes `future_provider_config` with `frame_delta=1`, `num_frames=5`,
  `num_inference_steps=1`, and `device=cuda:0`; no separate `wan_lora_device` field
  is present. Returned actions are finite with shape `[4, 4]`.
- **Outcome** Real Wan-LoRA serving works for a direct policy smoke and generated the
  cited media artifact. This is **not** a closed-loop MetaWorld rollout and does
  **not** establish task success.

### Real Wan-LoRA MetaWorld closed-loop smoke — plumbing / latency / media only
- **Command**
  ```bash
  # Terminal 1, from examples/world_model_env/
  CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=/tmp/uv-cache uv run python serve_world_model.py \
    --idm-checkpoint output/metaworld_loop_11_delta_ep0_15_2048_h4_noise004_hardened_split/best_idm_checkpoint.pt \
    --host 127.0.0.1 --port 8123 --image-keys observation/image \
    --future-provider wan_lora --diffsynth-repo-dir /tmp/DiffSynth-Studio \
    --wan-lora-checkpoint-dir /tmp/wan2.2-ti2v-5b \
    --wan-lora-path output/wan_metaworld_ep0_15_1424_128_lora3_e5/epoch-3.safetensors \
    --wan-lora-height 128 --wan-lora-width 128 --wan-lora-num-frames 5 \
    --wan-lora-num-inference-steps 1 --wan-lora-future-frame-strategy first \
    --wan-lora-output-dir output/serve_world_model/wan_lora_closed_loop_smoke \
    --device cuda:0

  # Terminal 2, from the repo root
  MUJOCO_GL=egl UV_CACHE_DIR=/tmp/uv-cache uv run python examples/metaworld/main.py \
    --host 127.0.0.1 --port 8123 --env-name reach-v3 --num-envs 1 \
    --num-episodes 1 --max-steps 8 --replan-steps 4 --width 64 --height 64 \
    --policy-cameras corner4 gripperPOV --render-camera corner4 --fps 12 \
    --output-dir examples/metaworld/output/world_model_wan_lora_smoke_reach
  ```
- **Artifact** MetaWorld video
  `examples/metaworld/output/world_model_wan_lora_smoke_reach/episode_000.mp4`
  and `contact_sheet.png`; Wan request artifacts under
  `output/serve_world_model/wan_lora_closed_loop_smoke/request_000000/` and
  `request_000001/` (each with `mp4`, input, and contact sheet).
- **Eval** Client logs report `mean_reward=13.26`, `success_rate=0.00` (0/1), and
  two real Wan requests.
- **Outcome** Real Wan-LoRA closed-loop plumbing runs through MetaWorld into the IDM
  and emits inspectable media/request artifacts. This is a **closed-loop
  plumbing/latency/media smoke only**, not a meaningful task-success rollout.

### 50-step real Wan-LoRA reach-v3 probes — tiny n=1, not a benchmark
- **Artifact / Eval** All three runs are single-episode, 50-step closed-loop
  MetaWorld probes with `max_steps=50`, `replan_steps=4`, and 13 real Wan requests
  (`request_000000`..`request_000012`). They are longer than the smoke above, but
  still **tiny n=1 probes**, not full benchmark/task-success evidence.

  | Run | Config | Client artifacts | Server artifacts | Result |
  |---|---|---|---|---|
  | Cheap latency-quality epoch3 | `epoch-3.safetensors`, `num_frames=5`, `num_inference_steps=1` | `examples/metaworld/output/world_model_wan_lora_reach50_eval/episode_000.mp4`, `contact_sheet.png` | `examples/world_model_env/output/serve_world_model/wan_lora_reach50_eval/`, including `representative_contact_sheet.png` | `mean_reward=151.41`, `success_rate=0.00` (0/1) |
  | Validated-quality epoch3 | `epoch-3.safetensors`, `num_frames=17`, `num_inference_steps=8` | `examples/metaworld/output/world_model_wan_lora_reach50_steps8_eval/episode_000.mp4`, `contact_sheet.png` | `examples/world_model_env/output/serve_world_model/wan_lora_reach50_steps8_eval/` | `mean_reward=162.15`, `success_rate=0.00` (0/1) |
  | Decodability-best epoch1 | `epoch-1.safetensors`, `num_frames=17`, `num_inference_steps=8` | `examples/metaworld/output/world_model_wan_lora_reach50_epoch1_steps8_eval/episode_000.mp4`, `contact_sheet.png` | `examples/world_model_env/output/serve_world_model/wan_lora_reach50_epoch1_steps8_eval/`, including `representative_contact_sheet.png` | `mean_reward=192.41`, `success_rate=0.00` (0/1) |

- **Outcome** No run succeeded yet, but epoch1 improves reward over epoch3 at the
  validated-quality setting (`192.41` vs `162.15`), consistent with the offline
  decodability ranking being useful. This remains an early signal only: task success
  still needs full-horizon, multi-episode closed-loop evaluation (run next).

### Full-horizon multi-episode real Wan-LoRA reach-v3 eval — first benchmark-shaped run (n=3, 300 steps)
- **Command**
  ```bash
  # Terminal 1, from examples/world_model_env/ — decodability-best epoch1, validated quality.
  CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=/tmp/uv-cache uv run python serve_world_model.py \
    --idm-checkpoint output/metaworld_loop_11_delta_ep0_15_2048_h4_noise004_hardened_split/best_idm_checkpoint.pt \
    --host 127.0.0.1 --port 8123 --image-keys observation/image \
    --future-provider wan_lora --diffsynth-repo-dir /tmp/DiffSynth-Studio \
    --wan-lora-checkpoint-dir /tmp/wan2.2-ti2v-5b \
    --wan-lora-path output/wan_metaworld_ep0_15_1424_128_lora3_e5/epoch-1.safetensors \
    --wan-lora-height 128 --wan-lora-width 128 --wan-lora-num-frames 17 \
    --wan-lora-num-inference-steps 8 --wan-lora-future-frame-strategy first \
    --wan-lora-output-dir output/serve_world_model/wan_lora_reach300_epoch1_steps8_eval_n3 \
    --device cuda:0

  # Terminal 2, from the repo root — eval_all.py scoped to one task writes
  # results.json + a reach-v3/ subdir (main.py alone would write episodes flat).
  MUJOCO_GL=egl UV_CACHE_DIR=/tmp/uv-cache uv run python examples/metaworld/eval_all.py \
    --host 127.0.0.1 --port 8123 --tasks reach-v3 --num-envs 1 \
    --num-episodes 3 --max-steps 300 --replan-steps 4 --width 64 --height 64 \
    --policy-cameras corner4 gripperPOV --render-camera corner4 --fps 12 \
    --output-dir examples/metaworld/output/world_model_wan_lora_reach300_epoch1_steps8_eval_n3
  ```
  The verifiable run shape — 3 episodes, `max_steps=300`, `replan_steps=4`, `num_envs=1`,
  75 requests/episode — is read back from the artifacts; the render-side flags mirror the
  50-step reach probes above. The `--host`/`--port` shown (`127.0.0.1:8123`) is an example
  free port; the port is not encoded in any artifact, so treat it as illustrative rather
  than artifact-proven.
- **Artifact**
  - Client: `examples/metaworld/output/world_model_wan_lora_reach300_epoch1_steps8_eval_n3/results.json`
    (`mean_success_rate=0.0`, `per_task={reach-v3: 0.0}`) plus
    `reach-v3/episode_00{0,1,2}.{json,mp4}`. Each per-episode JSON now carries
    `server_timing_ms` and `client_timing_ms` (see Eval).
  - Server: `examples/world_model_env/output/serve_world_model/wan_lora_reach300_epoch1_steps8_eval_n3/`
    with 225 request dirs (`request_000000`..`request_000224` = 75×3), each
    `sample_000/wan_lora_view0.mp4` + `wan_lora_view0_input.png` (450 files, ~5.3 MB; no
    top-level summary or contact sheet for this run).
- **Eval** Decodability-best Wan-LoRA **epoch1**, validated-quality generation
  (`num_frames=17`, `num_inference_steps=8`, `128×128`, future-frame strategy `first`)
  decoded by the canonical Loop-11 IDM. Three full-horizon episodes,
  **`success_rate=0.0` (0/3)**. Per-episode cumulative dense reward
  `1911.44 / 1479.42 / 748.78` (reach-v3 dense reward summed over 300 steps; it varies per
  episode and is **not** a success signal, and is not comparable to the 50-step probe
  rewards — different horizon). Latency (server `infer_ms`, now preserved client-side):
  warm requests are sub-second — combined **warm mean `936.7 ms`, median `932.3 ms`,
  range `915–1122 ms` over 224 warm requests**; ep1/ep2 means `941.1 / 935.4 ms`. Ep0's
  reported mean `1261.2 ms` is inflated by a single one-time **cold model-load first
  request of `25511 ms` (~25.5 s)**; excluding it, ep0 warm mean is `933.5 ms`, in line
  with ep1/ep2.
- **Outcome** First **benchmark-shaped** closed-loop run: full 300-step horizon, multiple
  episodes, real Wan-LoRA → IDM, with latency now instrumented end-to-end. Still **no task
  success (0/3)** — this validates the full pipeline at horizon and characterizes
  warm/cold latency, but reach-v3 success is unachieved. Larger n, more tasks, and
  best-of-N future selection remain open.

### Matched full-horizon epoch3 comparison — pixel-best checkpoint, same reach-v3 shape
- **Command** Same 300-step `eval_all.py` shape as the epoch1 full-horizon run above:
  `reach-v3`, `num_episodes=3`, `max_steps=300`, `replan_steps=4`, `num_envs=1`,
  `num_frames=17`, `num_inference_steps=8`, `128×128`, strategy `first`, canonical
  Loop-11 IDM. Only the Wan-LoRA checkpoint changes to `epoch-3.safetensors`
  (pixel-best in Loop 12) and the output dirs change to
  `wan_lora_reach300_epoch3_steps8_eval_n3`.
- **Artifact**
  - Client: `examples/metaworld/output/world_model_wan_lora_reach300_epoch3_steps8_eval_n3/results.json`
    plus `reach-v3/episode_00{0,1,2}.{json,mp4}`.
  - Server: `examples/world_model_env/output/serve_world_model/wan_lora_reach300_epoch3_steps8_eval_n3/`
    with 225 request dirs.
- **Eval** Pixel-best Wan-LoRA **epoch3**, same validated-quality generation and IDM as
  epoch1. Three full-horizon episodes, **`success_rate=0.0` (0/3)**. Per-episode
  cumulative dense reward `1430.30 / 1280.96 / 988.36` (mean `1229.87`) versus epoch1's
  `1911.44 / 1479.42 / 748.78` (mean `1379.88`). Dense reward is not a success signal,
  but on this tiny matched comparison epoch1 remains directionally better. Latency is in
  the same range: ep0 has one cold request (`25271.91 ms`), then warm means are
  `942.7 / 948.0 / 948.8 ms`.
- **Outcome** The matched full-horizon comparison does **not** produce success for either
  checkpoint, so the selector is still unvalidated against task success. It is
  directionally consistent with the offline decodability ranking: decodability-best
  epoch1 has higher dense reward than pixel-best epoch3 at the same run shape.

### Debug follow-up — `replan_steps=1` full-horizon reach-v3 (n=1, 300 steps)
- **Command** Same model stack and server command as the n=3 run above (epoch1,
  `num_frames=17`, `num_inference_steps=8`, `128×128`, strategy `first`, decoded by the
  canonical Loop-11 IDM), changing
  `--wan-lora-output-dir output/serve_world_model/wan_lora_reach300_epoch1_steps8_replan1_eval`
  and the server/client port — this run was executed on `8124` (the port is not encoded in
  any artifact). The client uses `main.py` directly (one episode → flat
  `episode_000.{json,mp4}`, no `results.json`/`reach-v3/` subdir):
  ```bash
  MUJOCO_GL=egl UV_CACHE_DIR=/tmp/uv-cache uv run python examples/metaworld/main.py \
    --host 127.0.0.1 --port 8124 --env-name reach-v3 --num-envs 1 \
    --num-episodes 1 --max-steps 300 --replan-steps 1 --width 64 --height 64 \
    --policy-cameras corner4 gripperPOV --render-camera corner4 --fps 12 \
    --output-dir examples/metaworld/output/world_model_wan_lora_reach300_epoch1_steps8_replan1_eval
  ```
  The verifiable run shape — 1 episode, `max_steps=300`, `replan_steps=1`, `num_envs=1`,
  `num_inference_requests=300` (1 request/step, vs 75/episode at `replan_steps=4`) — is
  read back from `episode_000.json` and the 300 server request dirs; render-side flags
  mirror the reach probes above.
- **Artifact**
  - Client: `examples/metaworld/output/world_model_wan_lora_reach300_epoch1_steps8_replan1_eval/`
    → `episode_000.json` + `episode_000.mp4`.
  - Server: `examples/world_model_env/output/serve_world_model/wan_lora_reach300_epoch1_steps8_replan1_eval/`
    — 300 request dirs (`request_000000`..`request_000299`; no top-level summary or
    contact sheet for this run).
  - Visual debug: `examples/world_model_env/output/visual_debug/wan_lora_reach300_epoch1_steps8_replan1_eval/`
    → `episode_000_rollout_grid.png` plus per-request Wan strips
    (`request_00{0000,0075,0150,0225,0299}_wan_strip.png`).
- **Eval** Single full-horizon episode, **`success_rate=0.0` (0/1)**, cumulative dense
  reward **`1649.14`** over 300 steps. Latency (server `infer_ms`, 300 requests): full
  mean `1009.89 ms` / p50 `926.35 ms` / max `25328.52 ms` is inflated by the same
  one-time cold model-load first request (`25328.52 ms`, ~25.3 s); excluding it, **warm
  mean `928.55 ms`, p50 `926.29 ms`, max `1011.30 ms`** over 299 warm requests —
  essentially matching the n=3 run's combined warm latency (`936.7 ms`).
- **Outcome** Replanning every step (`replan_steps=1`, execute only the first IDM action
  before re-imagining) still yields **0/1**. The dense reward `1649.14` falls inside the
  n=3 `replan_steps=4` per-episode range (`748.78`–`1911.44`, same horizon/task, and dense
  reward is not a success signal), so this is **not** a decisive improvement. It makes the
  open-loop 4-action chunk **less likely** to be the dominant bottleneck and points back at
  first-action fidelity / the world-model→IDM action mapping. Still a tiny n=1 debug probe,
  not benchmark evidence.

### Loop 13 — h1 IDM replan-1 hypothesis check
- **Question** If `replan_steps=1` executes only the first action, would an IDM trained
  directly with `action_horizon=1` improve first-action decoding versus taking the first
  element of the h4 IDM chunk?
- **Artifact**
  - Training: `output/metaworld_loop_13_delta_ep0_15_2048_h1_noise004_hardened_split/`
    (`best_idm_checkpoint.pt`; internal best epoch 31, `idm_mse=0.01970`).
  - Held-out GT diagnostic:
    `output/horizon_loop13_h1_gt_ep16_19_first64/idm_horizon_diagnostics.json`.
  - h1-compatible Wan caches:
    `output/wan_lora_cache_ep16_19_64_128_lora3_e5_epoch1_steps8_first4_h1/` and
    `output/wan_lora_cache_ep16_19_64_128_lora3_e5_epoch3_steps8_first4_h1/`.
  - Cached-Wan diagnostics:
    `output/horizon_loop13_h1_wan_epoch1_ep16_19_first64/idm_horizon_diagnostics.json`
    and `output/horizon_loop13_h1_wan_epoch3_ep16_19_first64/idm_horizon_diagnostics.json`.
- **Eval** Held-out ep16–19 first64, `frame_delta=1`, `num_future_frames=4`,
  `action_horizon=1`. GT futures decode well: **`idm_mse=0.03536`** with per-dim MSE
  `[dx 0.0755, dy 0.0318, dz 0.0337, grip 0.0005]`. Wan futures decode much worse:
  epoch1 **`0.08926`** (`[0.1189, 0.0810, 0.1436, 0.0134]`), epoch3 **`0.08751`**
  (`[0.1121, 0.0619, 0.1616, 0.0144]`).
- **Outcome** The h1 IDM improves held-out GT first-action decoding, but it is less
  robust to Wan-generated futures than the h4 first-action slice (h4 first-action MSE was
  `0.0512` for epoch1 and `0.0499` for epoch3 on the same held-out sample count).
  Therefore h1 is **not** a good online replan-1 candidate yet. The bottleneck remains
  generated-future/action fidelity, especially z and grip under Wan futures.

### Loop 14 — stochastic Wan seed sweep / naive best-of-N check
- **Question** Does sampling multiple Wan-LoRA futures for the same held-out rows produce
  a better action-decodable future, and is pixel quality a usable proxy for choosing it?
- **Artifact**
  - Seed caches: `output/wan_lora_cache_ep16_19_64_128_lora3_e5_epoch1_steps8_first4/`
    (`generation_seed=7`) plus `_seed17/`, `_seed23/`, `_seed1007/`, `_seed5007/`
    refreshed/generated with fixed dataset seed 7 and varied Wan `generation_seed`.
  - Ranking:
    `output/metaworld_loop_14_epoch1_seed_sweep_ep16_19_first64_rank5/ranking_summary.json`.
  - Tooling fix: `cache_future_rollouts.py --generation-seed` decouples stochastic Wan
    generation from `DatasetConfig.seed` and rejects same-output-dir resumes with a
    different generation seed.
- **Eval** Held-out ep16–19 first64, Wan-LoRA epoch1, Loop-11 h4 IDM, GT ref
  `idm_mse=0.0478629`.

  | seed | idm_mse | decodability_gap | future_mse | PSNR |
  |---|---|---|---|---|
  | 7 | **0.0619469** | **0.0140840** | 0.0014388 | 28.420 |
  | 23 | 0.0636633 | 0.0158005 | 0.0014543 | 28.373 |
  | 5007 | 0.0660957 | 0.0182328 | 0.0014427 | 28.408 |
  | 1007 | 0.0669561 | 0.0190932 | **0.0014052** | **28.523** |
  | 17 | 0.0700337 | 0.0221709 | 0.0014848 | 28.283 |

- **Outcome** Stochastic variance is real, but this 5-seed sweep did **not** find a
  better action-decodable sample than the original seed7. The pixel-best sample
  (seed1007) is **not** action-best; it improves `future_mse` while worsening the IDM
  decodability gap. A naive best-of-N selector based on pixel quality is therefore not
  justified yet. A future online best-of-N design should first add an action-aware or
  uncertainty-aware chooser rather than merely generating multiple futures.

### Loop 15 — in-distribution closed-loop panel / per-stage timing
- **Question** Do the current Loop-11 IDM + epoch1 Wan-LoRA policy produce success on
  additional MetaWorld tasks, and where does server-side latency actually go?
- **Artifact**
  - Server:
    `serve_world_model.py --future-provider wan_lora --wan-lora-path
    output/wan_metaworld_ep0_15_1424_128_lora3_e5/epoch-1.safetensors
    --wan-lora-num-inference-steps 8`.
  - Eval:
    `examples/metaworld/output/loop15_panel_assembly_epoch1_n3/results.json`.
  - Episode artifacts:
    `assembly-v3/episode_000.json` and `disassemble-v3/episode_000.json`.
- **Eval** Full-horizon 300-step rollouts, `replan_steps=4`, `num_envs=3`,
  `num_episodes=1` per task.

  | task | success | reward | requests | server mean | future mean | IDM mean |
  |---|---:|---:|---:|---:|---:|---:|
  | `assembly-v3` | 0/1 | 837.87 | 75 | 3082.03 ms | 3075.92 ms | 4.60 ms |
  | `disassemble-v3` | 0/1 | 122.49 | 75 | 2773.03 ms | 2767.91 ms | 3.59 ms |

- **Outcome** The broader panel still has **no task success**. The new per-stage timing
  is useful, though: server latency is dominated almost entirely by Wan future
  generation, while the IDM decode is only ~3-5 ms/request. `assembly-v3` includes a
  one-time cold first request (`27.4 s` total, `27.3 s` future-provider); warm requests
  are in the same ~2.7-2.8 s range as `disassemble-v3`.

### Loops 16–20 — Wan latency/quality sweep + state-normalized IDM robustness
- **Question** Can cheaper Wan settings or stronger IDM robustness training improve the
  generated-future action-decodability gap without violating the current data contract?
- **Artifacts**
  - Wan cache/rank:
    `output/loop16_epoch1_latency_quality_rank/ranking_summary.json`.
  - State-normalized IDM runs:
    `output/metaworld_loop_17_delta_ep0_15_2048_h4_noise004_state_norm/`,
    `output/metaworld_loop_18_delta_ep0_15_2048_h4_noise008_state_norm/`,
    `output/metaworld_loop_20_delta_ep0_15_2048_h4_noise012_state_norm/`.
  - Diagnostics:
    `output/diagnose_loop17_state_norm_{gt,wan_epoch1_f17_s8}_ep16_19_first64/`,
    `output/diagnose_loop18_noise008_state_norm_{gt,wan_epoch1_f17_s8}_ep16_19_first64/`,
    `output/diagnose_loop20_noise012_state_norm_{gt,wan_epoch1_f17_s8}_ep16_19_first64/`,
    plus the current-valid generated-cache recheck
    `output/diagnose_loop20_current_valid_cache_recheck_ep16_19_first64/idm_diagnostics.json`.
- **Eval**

  | loop | setting | internal best | held-out GT | held-out Wan epoch1 f17/s8 |
  |---|---|---:|---:|---:|
  | 11 | canonical, no state norm, noise 0.04 | 0.0388 | 0.0479 | 0.0619 |
  | 17 | state norm, noise 0.04 | 0.0316 | **0.0369** | 0.0736 |
  | 18 | state norm, noise 0.08 | 0.0453 | 0.0549 | 0.0728 |
  | 20 | state norm, noise 0.12 | **0.0376** | 0.0442 | **0.0534** |

  Loop 16 Wan generation sweep:

  | Wan setting | idm_mse | decodability gap | future_mse | PSNR |
  |---|---:|---:|---:|---:|
  | epoch1 f17 steps8 | **0.0619** | **0.0141** | **0.00144** | **28.42** |
  | epoch1 f17 steps4 | 0.0699 | 0.0221 | 0.00156 | 28.06 |
  | epoch1 f5 steps4 | 0.1851 | 0.1372 | 0.00242 | 26.16 |

- **Fix / contract** Dedicated IDM training remains GT-futures-only. `train_idm.py`
  rejects cached/generated futures, the IDM accepts only current 4D state, temporal
  state tensors are rejected, and checkpoint-loaded state normalizers are applied exactly
  once by a forward hook. Fresh audit found no active leakage/double-normalization bug;
  tests were added for the hook-active normalize-once path and serving-side
  state-normalized inputs.
- **Outcome** Cheap Wan settings are not good enough: f17/steps8 remains the best
  future cache. The current-valid generated-cache recheck at
  `output/diagnose_loop20_current_valid_cache_recheck_ep16_19_first64/idm_diagnostics.json`
  uses cache `output/loop16_cache_epoch1_f17_steps8_ep16_19_first64` and scores
  `idm_mse=0.06194690754637122` versus mean baseline `0.33408883213996887`, with
  `future_blind=false`. The same-family prior GT reference is `0.04420376801863313`.
  The older Loop20 state-normalized diagnostic (`0.0534` generated, `0.0442` GT) is
  retained above as prior evidence, but the recheck is the current decoded-video
  modular result to cite. This remains offline evidence, not a proven
  rollout-success checkpoint.

### Loop 20 closed-loop smoke — offline win, still no short-horizon success
- **Command**
  ```bash
  CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=/tmp/uv-cache uv run python serve_world_model.py \
    --idm-checkpoint output/metaworld_loop_20_delta_ep0_15_2048_h4_noise012_state_norm/best_idm_checkpoint.pt \
    --host 127.0.0.1 --port 8133 --image-keys observation/image \
    --future-provider wan_lora --diffsynth-repo-dir /tmp/DiffSynth-Studio \
    --wan-lora-checkpoint-dir /tmp/wan2.2-ti2v-5b \
    --wan-lora-path output/wan_metaworld_ep0_15_1424_128_lora3_e5/epoch-1.safetensors \
    --wan-lora-height 128 --wan-lora-width 128 --wan-lora-num-frames 17 \
    --wan-lora-num-inference-steps 8 --wan-lora-future-frame-strategy first \
    --wan-lora-output-dir output/serve_world_model/loop20_wan_epoch1_reach50_smoke \
    --device cuda:0

  MUJOCO_GL=egl UV_CACHE_DIR=/tmp/uv-cache uv run python examples/metaworld/main.py \
    --host 127.0.0.1 --port 8133 --env-name reach-v3 --num-envs 1 \
    --num-episodes 1 --max-steps 50 --replan-steps 4 --width 64 --height 64 \
    --policy-cameras corner4 gripperPOV --render-camera corner4 --fps 12 \
    --output-dir examples/metaworld/output/loop20_wan_epoch1_reach50_smoke
  ```
- **Artifact** Client:
  `examples/metaworld/output/loop20_wan_epoch1_reach50_smoke/episode_000.json` and
  `episode_000.mp4`; server:
  `output/serve_world_model/loop20_wan_epoch1_reach50_smoke/request_*/`.
- **Eval** 50-step `reach-v3`, `replan_steps=4`, 13 inference requests:
  `success_rate=0.00` (0/1), `mean_reward=116.11`. Server mean `2862.26 ms`; first
  cold request `26223.48 ms`, warm p50 `909.29 ms`. Stage timing: future provider mean
  `2851.56 ms`, IDM mean `9.67 ms` (warm IDM ~3.4 ms).
- **Outcome** Loop 20 improves offline Wan decodability but still does **not** produce
  short-horizon closed-loop task success. Latency remains dominated by Wan future
  generation, with the first request paying model-load cost and warm requests around
  `0.9 s`.

### Loop 20 in-distribution assembly-v3 closed loop — first task success
- **Command**
  ```bash
  CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=/tmp/uv-cache uv run python serve_world_model.py \
    --idm-checkpoint output/metaworld_loop_20_delta_ep0_15_2048_h4_noise012_state_norm/best_idm_checkpoint.pt \
    --host 127.0.0.1 --port 8134 --image-keys observation/image \
    --future-provider wan_lora --diffsynth-repo-dir /tmp/DiffSynth-Studio \
    --wan-lora-checkpoint-dir /tmp/wan2.2-ti2v-5b \
    --wan-lora-path output/wan_metaworld_ep0_15_1424_128_lora3_e5/epoch-1.safetensors \
    --wan-lora-height 128 --wan-lora-width 128 --wan-lora-num-frames 17 \
    --wan-lora-num-inference-steps 8 --wan-lora-future-frame-strategy first \
    --wan-lora-output-dir output/serve_world_model/loop20_wan_epoch1_assembly300_eval \
    --device cuda:0

  MUJOCO_GL=egl UV_CACHE_DIR=/tmp/uv-cache uv run python examples/metaworld/main.py \
    --host 127.0.0.1 --port 8134 --env-name assembly-v3 --num-envs 1 \
    --num-episodes 1 --max-steps 300 --replan-steps 4 --width 64 --height 64 \
    --policy-cameras corner4 gripperPOV --render-camera corner4 --fps 12 \
    --output-dir examples/metaworld/output/loop20_wan_epoch1_assembly300_eval
  ```
- **Artifact** Client:
  `examples/metaworld/output/loop20_wan_epoch1_assembly300_eval/episode_000.json` and
  `episode_000.mp4`; server:
  `output/serve_world_model/loop20_wan_epoch1_assembly300_eval/request_*/`.
  Visual contact sheets:
  `output/visual_debug_loop20_assembly/assembly_rollout_every24_contact.jpg` and
  `output/visual_debug_loop20_assembly/request{000,035,071}_future_contact.jpg`.
- **Eval** 300-step `assembly-v3`, `replan_steps=4`, 72 inference requests:
  `success_rate=1.00` (1/1), `mean_reward=494.03`. Server mean `1288.80 ms`; first
  cold request `26322.61 ms`, warm p50 `~935 ms`. Stage timing: future provider p50
  `931.05 ms`, IDM p50 `3.55 ms`.
- **Important caveat** Current `episodes 0..19` appear to be assembly-style demos
  (`pick up the nut and place it onto the peg`). This result is therefore
  in-distribution for the current train/eval slice. It is the first closed-loop success
  for the prototype, but it is **not** evidence of broad ML45 generalization.
- **Outcome** The full Wan2.2 -> IDM -> MetaWorld control loop can solve at least one
  in-distribution assembly rollout. The open gap is now task-diverse closed-loop
  robustness, especially on out-of-distribution tasks such as `reach-v3`.

### Loop 21 task-diverse IDM smoke — balanced GT-future baseline
- **Question** Does the small `delta` IDM architecture transfer beyond the assembly-only
  split when trained on a cheap task-diverse subset?
- **Split**
  - Train: first 2 demos from each metadata task (`88` episodes total).
  - External eval: next 1 demo from each task (`44` episodes), with
    `--samples-per-episode 8` for `352` balanced held-out samples.
- **Command shape**
  ```bash
  CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=/tmp/uv-cache uv run python train_idm.py \
    --dataset-source lerobot --repo-id brandonyang/metaworld_ml45 \
    --image-keys corner4.image --episodes <first_2_episodes_per_task> \
    --image-size 64 --frame-delta 1 --num-future-frames 4 --action-horizon 4 \
    --idm-arch delta --idm-future-noise-std 0.12 \
    --epochs 20 --batch-size 512 --num-workers 8 --learning-rate 1e-4 \
    --early-stopping-patience 5 --early-stopping-min-delta 0.0005 \
    --device cuda:0 \
    --output-dir output/metaworld_loop_21_diverse44_train2_delta_h4_noise012_state_norm_fast

  CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=/tmp/uv-cache uv run python diagnose_idm.py \
    --checkpoint output/metaworld_loop_21_diverse44_train2_delta_h4_noise012_state_norm_fast/best_idm_checkpoint.pt \
    --dataset-source lerobot --repo-id brandonyang/metaworld_ml45 \
    --image-keys corner4.image --episodes <third_episode_per_task> \
    --samples-per-episode 8 --image-size 64 --frame-delta 1 \
    --num-future-frames 4 --action-horizon 4 --batch-size 256 \
    --device cuda:0 --output-dir output/diagnose_loop21_diverse44_train2_gt_eval1_spe8
  ```
- **Artifact**
  - Training:
    `output/metaworld_loop_21_diverse44_train2_delta_h4_noise012_state_norm_fast/metrics.json`
    and `best_idm_checkpoint.pt`.
  - External GT diagnostic:
    `output/diagnose_loop21_diverse44_train2_gt_eval1_spe8/idm_diagnostics.json`.
- **Eval** Internal best was epoch 20 with `idm_mse=6.4072`. External balanced
  held-out GT-future diagnostic: `idm_mse=5.0121`, `first_action_mse=4.6045`,
  `last_action_mse=6.0828`, mean-action baseline `idm_mse=5.4063`. The largest error is
  still the z dimension (`per_action_dim_mse[2]=11.6373`). Future sensitivity did not
  collapse (`future_blind=false`), but the margin over mean-action is small.
- **Outcome** This is a useful negative result. The current small `delta` IDM plus 2 demos
  per task is only slightly better than a mean-action baseline on task-diverse held-out
  GT futures. Before expensive Wan task-diverse finetuning, the next IDM work should
  improve cross-task action scaling/conditioning, likely by training on more demos per
  task, adding task/language conditioning to the world model side only as needed, and
  revisiting a larger transformer or flow-transformer IDM with a balanced split.

### Loop 22 task-diverse closed-loop panel — Loop20 stack
- **Question** Does the best assembly-slice Loop20 stack generalize to a small
  task-diverse closed-loop MetaWorld panel?
- **Stack / run shape** Run name: Loop22 task-diverse closed-loop panel with Loop20
  stack. IDM checkpoint
  `output/metaworld_loop_20_delta_ep0_15_2048_h4_noise012_state_norm/best_idm_checkpoint.pt`;
  Wan-LoRA
  `output/wan_metaworld_ep0_15_1424_128_lora3_e5/epoch-1.safetensors`; one-view
  `observation/image`; `frame_delta=1`, `num_future_frames=4`, `action_horizon=4`;
  `replan_steps=4`, `max_steps=300`, `num_episodes=2` per task. Two Wan/IDM servers ran
  on ports `8140` and `8141`, driven by two client/GPU task panels.
- **Artifact**
  - GPU0 client results:
    `examples/metaworld/output/loop22_taskdiverse_loop20_gpu0/results.json`
    (`mean_success_rate=0.125`; `assembly-v3=0.5`, `disassemble-v3=0.0`,
    `reach-v3=0.0`, `push-v3=0.0`).
  - GPU1 client results:
    `examples/metaworld/output/loop22_taskdiverse_loop20_gpu1/results.json`
    (`mean_success_rate=0.0`; `pick-place-v3=0.0`, `peg-insert-side-v3=0.0`,
    `door-open-v3=0.0`, `sweep-v3=0.0`).
  - Visual debug:
    `examples/world_model_env/output/visual_debug_loop22_taskdiverse/*episode_*_every24.jpg`
    and
    `examples/world_model_env/output/visual_debug_loop22_taskdiverse/futures/*_wan17.jpg`.
- **Eval**

  | task | success | reward mean | server mean | server p50 | requests |
  |---|---:|---:|---:|---:|---:|
  | `assembly-v3` | 0.50 | 543.50 | 1106.4 ms | 925.7 ms | [72, 75] |
  | `disassemble-v3` | 0.00 | 129.16 | 938.3 ms | 927.9 ms | [75, 75] |
  | `push-v3` | 0.00 | 26.76 | 926.5 ms | 924.7 ms | [75, 75] |
  | `reach-v3` | 0.00 | 1257.39 | 926.2 ms | 922.4 ms | [75, 75] |
  | `door-open-v3` | 0.00 | 203.94 | 926.1 ms | 925.1 ms | [75, 75] |
  | `peg-insert-side-v3` | 0.00 | 5.60 | 926.0 ms | 925.2 ms | [75, 75] |
  | `pick-place-v3` | 0.00 | 2.05 | 1102.4 ms | 927.4 ms | [75, 75] |
  | `sweep-v3` | 0.00 | 14.49 | 929.4 ms | 927.5 ms | [75, 75] |

- **Outcome** The Loop20 stack remains assembly-slice / in-distribution competent
  (`assembly-v3` 1/2), but does not robustly generalize across task-diverse MetaWorld
  (only 1 success across 16 episodes; all seven non-assembly tasks are 0/2). Visual
  futures are near-static on these examples, so the failures are consistent with
  insufficient task-diverse WM/IDM training rather than a server/runtime issue. Warm
  server p50 remains about `922-928 ms` and is still dominated by Wan future generation.

### Loop22 prompt/seed serving ablation — visual check
- **Serving patch** Added explicit Wan-LoRA ablation controls in
  `serve_world_model.py`: `--wan-lora-prompt-template` and `--wan-lora-seed`.
  Validation requires `{task}`, rejects blank templates and negative seeds, threads
  `prompt_template`/`base_seed` into `DiffSynthWanLoraConfig`, passes explicit seeds to
  `generate_future_stack`, and records prompt template/base seed/provider seed in
  metadata. Verification passed:
  `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check serve_world_model.py tests/test_serve_world_model.py`
  and
  `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_serve_world_model.py -q`
  (`88 passed, 10 warnings`).
- **Visual ablation** Used the failing Loop22 reach current image
  `output/serve_world_model/loop22_taskdiverse_loop20_gpu0/request_000297/sample_000/wan_lora_view0_input.png`
  with LoRA `output/wan_metaworld_ep0_15_1424_128_lora3_e5/epoch-1.safetensors`,
  checkpoint dir `/tmp/wan2.2-ti2v-5b`, 17 frames, 8 steps, seed 7. Compared the
  default prompt template against:
  `Robot manipulation in MetaWorld. Current task: {task}. Generate a short near-future video where the robot end effector visibly moves toward the goal, and any manipulated object moves closer to completing the task. Keep the same camera view, same scene, realistic robot motion, no cuts.`
  Artifacts:
  `output/prompt_seed_ablation_reach_loop22_request297/generic_current_plus_futures.jpg`
  and
  `output/prompt_seed_ablation_reach_loop22_request297/motion_explicit_current_plus_futures.jpg`.
- **Metrics / interpretation** Generic `motion_vs_current_mae` was
  `[0.014375892467796803, 0.015416443347930908, 0.01715032011270523, 0.018950898200273514]`
  and `adjacent_future_mae`
  `[0.006038890685886145, 0.0035250394139438868, 0.0040305545553565025]`.
  Motion-explicit `motion_vs_current_mae` was
  `[0.015119404532015324, 0.016606828197836876, 0.018577346578240395, 0.0200569499284029]`
  and `adjacent_future_mae`
  `[0.0047250790521502495, 0.0033495137467980385, 0.0048029483295977116]`.
  The motion-explicit prompt slightly increases pixel difference from the current frame,
  but contact sheets remain essentially static and not task-directed. Prompt wording
  alone is not the fix; next work should target world-model fine-tuning/data/objectives
  or stronger task-diverse IDM evidence, not simply prompt changes.

### Loop77 explicit decoded-video IDM closed-loop assembly smoke
- **Stack / run shape** `decoded_video_idm` using checkpoint
  `output/idm_flow_patch_crossattn_futuredelta_gt_ep0_15_spe64_h4_seed7_no_rank/best_idm_checkpoint.pt`;
  Wan-LoRA `output/wan_metaworld_ep0_15_1424_128_lora3_e5/epoch-4.safetensors`;
  `num_frames=17`, `num_inference_steps=4`, seed `7`, image key
  `observation/image`. Client task `assembly-v3`, `num_envs=1`, `num_episodes=1`,
  `max_steps=300`, `replan_steps=4`; server port `8160`.
- **Artifacts** Client JSON/video:
  `examples/metaworld/output/loop77_patch_futuredelta_epoch4_steps4_assembly300_eval/episode_000.json`
  and `.mp4`. Server futures:
  `examples/world_model_env/output/serve_world_model/loop77_patch_futuredelta_epoch4_steps4_assembly300_eval/request_*/sample_000/wan_lora_view0.mp4`
  plus input PNGs, with 27 request directories. Visual sheet:
  `examples/world_model_env/output/visual_debug_loop77_patch_futuredelta/assembly_success_rollout_and_wan_futures.jpg`.
- **Eval / interpretation** `episode_000.json` reports `success_rate=1.0`,
  `success=[true]`, `mean_reward=301.7853852395811`, and
  `num_inference_requests=27`. Server mean was `1657.3440405988583 ms`, server p50
  `788.7278888374567 ms`, future-provider p50 `671.6894679702818 ms`, and IDM p50
  `116.03108700364828 ms`; the first-request max `24234.541 ms` includes model load,
  so p50/min better represent warm latency. This is the first closed-loop success for
  the latest explicit patch-token cross-attention `future_delta` decoded-video IDM on
  assembly, but it is still n=1 assembly-only evidence, not broad task-diverse success.

### Loop78 explicit decoded-video IDM closed-loop reach contrast
- **Stack / run shape** Same `decoded_video_idm` stack as Loop77: checkpoint
  `output/idm_flow_patch_crossattn_futuredelta_gt_ep0_15_spe64_h4_seed7_no_rank/best_idm_checkpoint.pt`;
  Wan-LoRA `output/wan_metaworld_ep0_15_1424_128_lora3_e5/epoch-4.safetensors`;
  `num_frames=17`, `num_inference_steps=4`, seed `7`, image key
  `observation/image`. Client task `reach-v3`, `num_envs=1`, `num_episodes=1`,
  `max_steps=300`, `replan_steps=4`; server port `8161` on GPU1.
- **Artifacts** Client JSON/video:
  `examples/metaworld/output/loop78_patch_futuredelta_epoch4_steps4_reach300_eval/episode_000.json`
  and `.mp4`. Server futures:
  `examples/world_model_env/output/serve_world_model/loop78_patch_futuredelta_epoch4_steps4_reach300_eval/request_*/sample_000/wan_lora_view0.mp4`
  plus input PNGs, with 75 request directories. Visual sheet:
  `examples/world_model_env/output/visual_debug_loop78_patch_futuredelta/reach_failure_rollout_and_wan_futures.jpg`.
- **Eval / interpretation** `success_rate=0.0`, `success=[false]`,
  `mean_reward=1005.1715246465805`, `num_inference_requests=75`. Server mean was
  `1101.5752180789907 ms`, server p50 `788.151300046593 ms`, future-provider p50
  `671.6568609699607 ms`, and IDM p50 `116.21057009324431 ms`; first-request max
  `24297.10986185819 ms` includes model load. The Loop77 stack succeeds on assembly
  but does not solve reach in this n=1 full-horizon contrast, so broad task success
  remains open.

### Loop79 task-diverse patch/cross-attention future-delta IDM — train2/spe15
- **Question** Can the explicit decoded-video patch-token cross-attention
  `future_delta` IDM scale from the assembly slice to a cheap ML44 task-diverse
  split?
- **Stack / run shape** GT-future IDM training on MetaWorld ML44 train2 episodes with
  `samples_per_episode=15`, `image_keys=corner4.image`, `image_size=64`,
  `frame_delta=1`, `num_future_frames=4`, `action_horizon=4`, patch visual encoder,
  visual-token conditioning enabled with `cross_attention`, `future_only`,
  `future_delta`, flow sampling steps `16`, sample noise scale `0.0`, `epochs=80`,
  `batch_size=64`, `lr=1e-4`, seed `7`. The initial train2/spe16 attempt failed
  before writing outputs because episode `1780` has only 15 valid nonterminal windows
  for the h4 future/action contract.
- **Artifacts** Train:
  `output/idm_flow_patch_crossattn_futuredelta_gt_train2_spe15_h4_seed7_no_rank`.
  Held-out eval44:
  `output/eval_idm_flow_patch_crossattn_futuredelta_gt_train2_spe15_eval44_spe15_h4_seed7_no_rank/eval_metrics.json`.
  Diagnostic:
  `output/diagnose_idm_flow_patch_crossattn_futuredelta_gt_train2_spe15_eval44_spe15_h4_seed7_no_rank/idm_diagnostics.json`.
- **Metrics / interpretation** Training wrote 80 metric rows; best internal epoch 31
  scored `idm_mse=5.799701424206004`, `idm_smooth_l1=1.0635229103705461`;
  final epoch 80 scored `6.0090044035631065`, `1.1072153904858757`. Held-out
  eval44 third-demo `spe15` scored `idm_mse=5.909213927297881`,
  `idm_smooth_l1=0.808262417533181`, n=660, versus mean-action baseline
  `6.483566931522254`. The diagnostic marked `future_blind=true`:
  current-repeated output delta MSE was only `0.000504582342832829`, and
  real-vs-current-repeated degradation was `2.323497425393839e-05`. This learns
  slightly over the mean baseline but mostly collapses to state/action priors; train2
  is not enough broad future-conditioned IDM evidence.

### Loop80 task-diverse patch/cross-attention future-delta IDM — train8/spe8
- **Question** Does broader per-task data coverage fix the Loop79 future-blind
  train2 result without changing architecture?
- **Stack / run shape** Same explicit decoded-video IDM contract as Loop79, but using
  the existing ML44 train8 episode split with `samples_per_episode=8`. Training ran
  `epochs=80`, `batch_size=64`, `lr=1e-4`, seed `7`, on GPU1 while Loop79 eval and
  diagnostics used GPU0.
- **Artifacts** Train:
  `output/idm_flow_patch_crossattn_futuredelta_gt_train8_spe8_h4_seed7_no_rank`.
  Held-out eval44:
  `output/eval_idm_flow_patch_crossattn_futuredelta_gt_train8_spe8_eval44_spe15_h4_seed7_no_rank/eval_metrics.json`.
  Diagnostic:
  `output/diagnose_idm_flow_patch_crossattn_futuredelta_gt_train8_spe8_eval44_spe15_h4_seed7_no_rank/idm_diagnostics.json`.
- **Metrics / interpretation** Training wrote 80 metric rows; best internal epoch 52
  scored `idm_mse=5.554099767980441`, `idm_smooth_l1=1.0670977041754923`;
  final epoch 80 scored `6.114046338578345`, `1.1157865524291992`. Held-out
  eval44 third-demo `spe15` improved to `idm_mse=4.942809457489939`,
  `idm_smooth_l1=0.7019280881592722`, n=660, versus mean-action baseline
  `6.478426626956824`. The diagnostic marked `future_blind=false`:
  current-repeated output delta MSE was `0.07054621360518716`, and
  real-vs-current-repeated degradation was `0.01572215918338582`. Extra demos
  materially improve task-diverse eval and produce real but weak future sensitivity.
  This is the best broad explicit decoded-video patch-IDM result so far, but it is still
  far from the narrow assembly run; the next loop should add an explicit future-usage,
  ranking, same-task-delta, or contact/task-tail objective rather than simply training
  longer.

### Loop81 task-diverse temporal-gap IDM — train8/spe4, frame_delta=2
- **Question** Does asking the IDM to decode actions from farther future frames make the
  future signal less ignorable than Loop80's `frame_delta=1` setup?
- **Stack / run shape** Same patch-token cross-attention `future_delta` flow IDM as
  Loop80, but with `frame_delta=2`, `samples_per_episode=4`, `epochs=50`,
  `batch_size=32`, `lr=1e-4`, seed `7`. The fd4 probes were rejected before useful
  training because short episodes had too few valid windows for the required offset.
- **Artifacts** Train:
  `output/idm_flow_patch_crossattn_futuredelta_gt_train8_spe4_fd2_h4_seed7_no_rank`.
  Held-out eval44:
  `output/eval_idm_flow_patch_crossattn_futuredelta_gt_train8_spe4_fd2_eval44_spe4_h4_seed7_no_rank/eval_metrics.json`.
  Diagnostic:
  `output/diagnose_idm_flow_patch_crossattn_futuredelta_gt_train8_spe4_fd2_eval44_spe4_h4_seed7_no_rank/idm_diagnostics.json`.
- **Metrics / interpretation** Training completed 50 epochs; final/internal-best epoch
  50 scored `idm_mse=4.45141326797592`, `idm_smooth_l1=0.9103104117866996`.
  Held-out eval44 third-demo `spe4` scored `idm_mse=4.960944782603871`,
  `idm_smooth_l1=0.6321175640279596`, n=176, versus mean-action baseline
  `6.913035652854226`. The diagnostic marked `future_blind=false` with
  current-repeated output delta MSE `0.8528993779962714`,
  real-vs-current-repeated degradation `1.4892077012495557`,
  teacher-forced rank accuracy `0.4772727272727273`, and
  real-vs-best-negative gap `0.00014665214852853254`. This is a useful broad
  future-use signal, but the real-vs-negative margin is tiny and the action MSE is only
  tied with Loop80 rather than a clear improvement.

### Loop82 ranking and same-task future-delta ablations — negative
- **Scheduled ranking weight 0.05.** Added delayed/ramped future-ranking loss on top of
  Loop80's train8/spe8 setup (`start_epoch=16`, `ramp_epochs=16`, repeated-current,
  shuffled, and zero negatives). Artifact:
  `output/idm_flow_patch_crossattn_futuredelta_gt_train8_spe8_rank_sched_w005_rsz_h4_seed7_no_rank`.
  The run was intentionally stopped after epoch 32 once the ranking weight reached
  `0.05`. At epoch 32, internal eval was future-sensitive but still failed the key
  ranking gate: current-repeated output delta MSE `0.006101900291904597`,
  rank accuracy `0.3626760563380282`, and real-vs-best-negative gap
  `-0.07808239490423403`. Held-out eval44 `spe15` scored
  `idm_mse=5.246247690374201`, `idm_smooth_l1=0.7378458340962728`, n=660, versus
  mean-action baseline `6.478426638516512`, worse than Loop80's `4.942809457489939`.
  Broad diagnostic confirmed `future_blind=false` but still negative real-vs-best-negative
  gap `-0.020187014160734235`. Interpretation: the scalar ranking objective makes the
  model sensitive to future perturbations, but it does not teach preference for the
  correct future and hurts broad action MSE.
- **Same-task future-delta loss.** The same-task donor ablation
  `output/idm_flow_patch_crossattn_futuredelta_gt_train8_spe8_sametaskdelta_w02_near075_mina005_h4_seed7_no_rank`
  was stopped at 10 metric rows. Donors were plentiful (effective donor fraction
  about `0.926`), but predicted action deltas collapsed toward zero while target
  action-delta MSE stayed around `0.85-0.90`; epoch-10 internal `idm_mse=6.436` was
  already worse than Loop80. Interpretation: this formulation did not make the IDM
  learn useful action differences between futures.

### Loops 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48 — task-diverse flow-DiT IDM / Wan VAE probes
- **Scope** These are task-diverse flow-DiT IDM experiments, not the older
  assembly-only canonical split.
- **Patch baseline (Loop 37)** Patch visual encoder with additive flow-DiT IDM reached
  internal best `idm_mse=3.528105`. Heldout GT diagnostic scored
  `idm_mse=6.080973` vs mean baseline `6.126930`; `future_blind=true`, and
  current-repeated `output_delta_mse=0.000000005`.
- **Frozen Wan VAE IDM (Loop 39)** Frozen Wan VAE visual encoder reached best epoch 18,
  internal best `idm_mse=4.348827`, final `4.404776`. Heldout GT diagnostic scored
  `idm_mse=6.196227` vs mean baseline `6.126930`; `future_blind=false`, with
  current-repeated `output_delta_mse=0.004911485`.
- **Wan VAE latent cache** Smoke cache wrote 8 samples in `11.96s`; one-batch cached
  training took `7.22s` vs uncached `12.32s` with matching metrics. Larger cache wrote
  352 samples in `23.83s`.
- **Cached Wan VAE + context warmup (Loop 40)** Best epoch 18, internal best
  `idm_mse=4.862549`, final `5.981010`; active context weights for the first six epochs
  were `[1, 1, 1, 1, 1, 0]`. Negative for MSE vs Loop 39, but positive evidence for
  cache throughput.
- **Cached Wan VAE, no warmup (Loop 41)** Artifact
  `output/metaworld_loop_41_diverse44_train2_spe4_flow_dit350m_wanvae_cached_tmax025_h4_endpoint01`;
  cached Wan VAE, no context warmup, `tmax=0.25`, `h=4`, additive conditioning. Best
  internal epoch 10 `idm_mse=4.920884`; final `idm_mse=5.462894`. The cache path is a
  speed/infrastructure win, but this run did not reproduce uncached Loop 39 MSE.
- **Cached Wan VAE + latent noise (Loop 42)** Artifact
  `output/metaworld_loop_42_diverse44_train2_spe4_flow_dit350m_wanvae_cached_latnoise_p05_s05_10_tmax025_h4_endpoint01`;
  cached Wan VAE latents, additive conditioning, `tmax=0.25`, `h=4`, endpoint `0.1`,
  latent noise `p=0.5`, `s=[0.5, 1.0]`. Best internal epoch 10
  `idm_mse=5.064327`; final `idm_mse=5.403182`; noise fraction was about `0.50` with
  `s_mean` about `0.74-0.76`. Held-out cache
  `output/wan_vae_latent_cache_diverse44_eval2_3_spe2_h4` wrote 176 samples. Held-out
  eval scored `idm_mse=6.498383` vs mean baseline `6.124695`; diagnostic scored
  `future_blind=false`, current-repeated `output_delta_mse=0.000127973`, and
  `real_vs_current_repeated_degradation=0.004718`. Loop 41 on the same held-out eval
  scored `idm_mse=6.578063` vs the same baseline and `future_blind=true`. Noisy latents
  improved future sensitivity and slightly beat Loop 41 held-out, but both are worse than
  the mean-action baseline; not the next accuracy baseline.
- **Cached Wan VAE + gentler latent noise (Loop 43)** Artifact
  `output/metaworld_loop_43_diverse44_train2_spe4_flow_dit350m_wanvae_cached_latnoise_p025_s075_10_tmax025_h4_endpoint01`;
  cached Wan VAE latents, additive conditioning, `tmax=0.25`, `h=4`, endpoint `0.1`,
  latent noise `p=0.25`, `s=[0.75, 1.0]`. Best/final internal epoch 20
  `idm_mse=5.001990`, `smooth_l1=1.060535`; train noise fraction `0.28115`,
  `s_mean=0.87552`. Held-out eval
  `output/eval_loop43_wanvae_cached_latnoise025_gt_eval2_3_spe2` scored
  `idm_mse=6.196259`, `smooth_l1=1.052104` vs mean baseline `6.124695`; diagnostic
  `output/diagnose_loop43_wanvae_cached_latnoise025_gt_eval2_3_spe2` scored
  `future_blind=false`, current-repeated `output_delta_mse=0.00169029`, and
  `real_vs_current_repeated_degradation=0.0177183`. Gentler noise is better than Loop 42
  on held-out and keeps future sensitivity, but remains slightly worse than the
  mean-action baseline; not a solved accuracy baseline.
- **Future-only Wan VAE latent noise (Loop 44)** Added
  `idm_wan_vae_latent_noise_time_mode` with default `all` and `future_only`; for batched
  `(B,C,T,H,W)` latents, `future_only` leaves latent time index 0 unchanged and corrupts
  only `T>=1`, matching the latent-gap diagnostic where `t0` is near-identical and the
  future latent slice carries the gap. Artifact
  `output/metaworld_loop_44_diverse44_train2_spe4_flow_dit350m_wanvae_cached_latnoise_p025_s075_10_futureonly_tmax025_h4_endpoint01`;
  same as Loop 43 except `idm_wan_vae_latent_noise_time_mode='future_only'`. Internal
  final/best epoch 20 improved MSE vs Loop 43 but worsened smooth L1:
  Loop44 `idm_mse=4.857773317609515`, `idm_smooth_l1=1.0834537165505544`; Loop43
  internal `5.001990291050502`, `1.0605349268232074`. Matched external eval on episodes
  2/3, `samples_per_episode=16`, `h=4`, same 32 samples:

  | eval | idm_mse | smooth_l1 |
  |---|---:|---:|
  | Loop43 GT cache, `output/eval_loop43_wanvae_cached_latnoise025_gt_eval2_3_spe16` | 1.4906591773033142 | 0.5090571194887161 |
  | Loop44 GT cache, `output/eval_loop44_wanvae_cached_futureonly_gt_eval2_3_spe16` | 1.6896296739578247 | 0.5718317031860352 |
  | Loop44 generated latent 1/4, `output/eval_loop44_real_generated_wan_latent_lora3_e5_epoch4_ep2_3_spe16_h4_partial_s1of4_timed` | 1.6864711046218872 | 0.5715441107749939 |
  | Loop44 generated latent 2/4, `output/eval_loop44_real_generated_wan_latent_lora3_e5_epoch4_ep2_3_spe16_h4_partial_s2of4_timed` | 1.6885976791381836 | 0.5721623003482819 |
  | Loop44 generated latent 3/4, `output/eval_loop44_real_generated_wan_latent_lora3_e5_epoch4_ep2_3_spe16_h4_partial_s3of4_timed` | 1.6906144618988037 | 0.5727440714836121 |
  | Loop44 generated latent 4/4 full, `output/eval_loop44_real_generated_wan_latent_lora3_e5_epoch4_ep2_3_spe16_h4_full_s4_timed` | 1.6910588145256042 | 0.5720984637737274 |
  | mean-action baseline on this slice | 0.414536714553833 | 0.19033678621053696 |

  Diagnostic
  `output/diagnose_loop44_wanvae_cached_futureonly_gt_eval2_3_spe16/idm_diagnostics.json`
  scored `future_blind=false` by gate, current-repeated output delta MSE
  `0.0002781580697046593`, and real-vs-current-repeated degradation
  `0.0013756752014160156`; however all GT/generated eval scores are nearly flat around
  `1.69`, so the model is not meaningfully exploiting denoise-level latent quality on
  this held-out slice. Future-only latent noise is a useful knob and slightly improves
  internal split MSE, but it is negative externally versus Loop 43 and still far worse
  than the mean-action baseline. Next direction should not be more simple time-mode
  noise; prefer stronger future-conditioning objectives/architecture such as a
  larger/healthier latent IDM, contrastive/action endpoint losses targeted to future
  latents, action/state history tokens, or a patch+Wan latent hybrid. Keep the constraint
  of not training on generated WM frames.
- **Cached Wan VAE + future contrastive loss (Loop 45)** Artifact
  `output/metaworld_loop_45_diverse44_train2_spe4_flow_dit350m_wanvae_cached_latnoise_p025_s075_10_contrast_w05_m01_tmax025_h4_endpoint01`;
  same as Loop 43 plus cached Wan VAE latent-compatible future contrastive loss
  (`idm_future_contrastive_weight=0.5`, margin `0.1`). Best checkpoint is epoch 15:
  `idm_mse=4.958617510114397`, `idm_smooth_l1=1.064666325705392`; final epoch 20 was
  worse at `idm_mse=5.4515644073486325`, `idm_smooth_l1=1.1133646965026855`. Training
  contrastive real/corrupted endpoint MSE stayed close, so separation was weak. Matched
  external eval on episodes 2/3, `samples_per_episode=16`, `h=4`, same 32 samples:

  | eval | idm_mse | smooth_l1 |
  |---|---:|---:|
  | Loop45 GT cache, `output/eval_loop45_wanvae_cached_contrast_w05_gt_eval2_3_spe16` | 1.3849986791610718 | 0.4892744719982147 |
  | Loop45 generated latent 1/4, `output/eval_loop45_real_generated_wan_latent_lora3_e5_epoch4_ep2_3_spe16_h4_partial_s1of4_timed` | 1.3873890042304993 | 0.4890851080417633 |
  | Loop45 generated latent 2/4, `output/eval_loop45_real_generated_wan_latent_lora3_e5_epoch4_ep2_3_spe16_h4_partial_s2of4_timed` | 1.387679636478424 | 0.4892708659172058 |
  | Loop45 generated latent 3/4, `output/eval_loop45_real_generated_wan_latent_lora3_e5_epoch4_ep2_3_spe16_h4_partial_s3of4_timed` | 1.3879331350326538 | 0.4895540028810501 |
  | Loop45 generated latent 4/4 full, `output/eval_loop45_real_generated_wan_latent_lora3_e5_epoch4_ep2_3_spe16_h4_full_s4_timed` | 1.3900517225265503 | 0.49070675671100616 |
  | mean-action baseline on this slice | 0.414536714553833 | 0.19033678621053696 |

  Diagnostic
  `output/diagnose_loop45_wanvae_cached_contrast_w05_gt_eval2_3_spe16/idm_diagnostics.json`
  scored `future_blind=true`, current-repeated output delta MSE
  `8.582998998463154e-05`, and real-vs-current-repeated degradation
  `0.006462752819061279`. Loop45 improves external MSE versus Loop43/44, and the
  future contrastive objective is promising, but it remains far worse than the mean-action
  baseline and mostly insensitive to denoise level/future replacement. Next direction:
  history-conditioned IDM plus a stronger future-conditioning objective/architecture.
  Keep the modular Wan -> separate IDM contract and continue avoiding IDM training on
  generated WM frames.
- **History-conditioned cached Wan VAE follow-up (Loop 46)** Current Loop46 eval/diagnose
  evidence uses the same matched episodes 2/3, `samples_per_episode=16`, `h=4` shape with
  history length 2. GT cached Wan VAE latents score `idm_mse=0.866048663854599`,
  `smooth_l1=0.3261895924806595`; the mean-action baseline is still much lower at
  `idm_mse=0.41459181904792786`. Generated Wan latent denoise fractions are nearly flat:

  | denoise fraction | idm_mse |
  |---|---:|
  | 0.25 | 0.8721326291561127 |
  | 0.50 | 0.8718866109848022 |
  | 0.75 | 0.870557963848114 |
  | 1.00 | 0.8681868016719818 |

  Diagnostic
  `output/diagnose_loop46_wanvae_cached_contrast_w05_hist2_gt_eval2_3_spe16/idm_diagnostics.json`
  still shows future-blind behavior: current-repeated `output_delta_mse=9.1487e-05`,
  and current-repeated/shuffled/zero/noise replacements do not meaningfully degrade
  action error versus the real future. Loop46 reduces absolute MSE versus Loop45, but it
  remains worse than the mean-action baseline and does not yet exploit GT-vs-generated or
  partial-vs-full denoise quality. Next LingBot-VA-inspired steps: train an explicit
  future-ranking objective, augment GT Wan latents with partial/noisy-latent corruption,
  and let the action DiT cross-attend over Wan latent tokens instead of compressing them
  too early. Live serving supports history-conditioned checkpoints with per-connection
  history buffers, but that history is exact only when requests arrive every env step
  (`replan_steps=1`); larger `replan_steps` need client-supplied per-step history for
  training-faithful conditioning.
- **Future-ranking implementation verification after Loop46** The follow-up implementation
  passed:
  `uv run ruff check world_model/config.py world_model/train_lib.py train_idm.py run_idm_experiments.py diagnose_idm.py tests/test_training_smoke.py tests/test_run_idm_experiments.py tests/test_diagnose_idm.py`
  and
  `uv run pytest -q tests/test_training_smoke.py tests/test_run_idm_experiments.py tests/test_diagnose_idm.py`
  (`80 passed, 7 warnings`).
- **Loop46 future-ranking diagnostic follow-up** New diagnostic artifact:
  `output/diagnose_loop46_wanvae_cached_contrast_w05_hist2_gt_eval2_3_spe16_future_ranking/idm_diagnostics.json`.
  It keeps `future_blind=true`. Teacher-forced future ranking scores
  `rank_accuracy=0.34375`, `mean_real_candidate_rank=2.375`, and
  `real_vs_best_negative_gap=-0.0009136747685261071`; endpoint MSE for the real
  candidate is `0.12640448659658432` versus shuffled `0.12616709619760513`.
- **Future-ranking weight 0.5 all negatives (Loop 47)** This is a post-Loop46 follow-up
  using artifact
  `output/metaworld_loop_47_diverse44_train2_spe4_flow_dit350m_wanvae_cached_latnoise_p025_s075_10_contrast_w05_rank_w05_allnegs_hist2_tmax025_h4_endpoint01`
  with ranking weight `0.5` over all negatives. Best epoch 11 internal
  `idm_mse=3.1745423998151505`, `rank_accuracy=0.8456591655203767`; final epoch 20
  internal `idm_mse=4.30932366507394`, `rank_accuracy=0.9421221864951769`.
  External GT eval `output/eval_loop47_rank_w05_allnegs_hist2_gt_eval2_3_spe16`
  scored `idm_mse=1.5603640675544739`. Generated-latent MSE by denoise fraction:
  `0.25=2.826761484146118`, `0.50=2.41923451423645`,
  `0.75=1.9342970848083496`, `1.00=1.668363630771637`. Diagnostic
  `output/diagnose_loop47_rank_w05_allnegs_hist2_gt_eval2_3_spe16/idm_diagnostics.json`
  scored `future_blind=false`, current-repeated `output_delta_mse=0.13728578388690948`,
  future-ranking `rank_accuracy=0.4375`, and
  `real_vs_best_negative_gap=0.027186322957277298`. Interpretation: Loop47 fixed the
  future-blind gate but hurt action accuracy badly.
- **Future-ranking weight 0.05 all negatives (Loop 48)** This is a post-Loop46 follow-up
  using artifact
  `output/metaworld_loop_48_diverse44_train2_spe4_flow_dit350m_wanvae_cached_latnoise_p025_s075_10_contrast_w05_rank_w005_allnegs_hist2_tmax025_h4_endpoint01`
  with ranking weight `0.05` over all negatives. Best epoch 19 internal
  `idm_mse=3.1908015932355607`, `rank_accuracy=0.2154340837450273`; final epoch 20
  internal `idm_mse=3.3458907808576313`, `rank_accuracy=0.22829582007943242`.
  External GT eval `output/eval_loop48_rank_w005_allnegs_hist2_gt_eval2_3_spe16`
  scored `idm_mse=0.9088110327720642`. Generated-latent MSE by denoise fraction:
  `0.25=0.9142842292785645`, `0.50=0.9132027328014374`,
  `0.75=0.9128563404083252`, `1.00=0.9147288203239441`. Diagnostic
  `output/diagnose_loop48_rank_w005_allnegs_hist2_gt_eval2_3_spe16/idm_diagnostics.json`
  scored `future_blind=false` by the current gate, but sensitivity is weak:
  current-repeated `output_delta_mse=0.0017336404998786747`, future-ranking
  `rank_accuracy=0.28125`, and `real_vs_best_negative_gap=-0.001250106026418507`.
  Interpretation: Loop48 mostly recovers action error versus Loop47, but it does not
  meaningfully solve future use and is still worse than Loop46 on GT/generated evals.
- **Wan latent token conditioning, no ranking loss (Loop 49)** This is the clean
  architecture ablation after Loop46/47/48, using artifact
  `output/metaworld_loop_49_diverse44_train2_spe4_flow_dit350m_wanvae_cached_visualtokens_latnoise_p025_s075_10_contrast_w05_hist2_tmax025_h4_endpoint01`
  with `idm_flow_visual_token_conditioning=true`, history length 2, ranking weight 0,
  and the same GT Wan VAE latent cache as Loop46. Best epoch 19 internal
  `idm_mse=2.8739586421421595`; final epoch 20 internal
  `idm_mse=3.2564889090401787`. External GT eval
  `output/eval_loop49_visualtokens_hist2_gt_eval2_3_spe16` scored
  `idm_mse=0.8640031516551971`, roughly tied with Loop46. Generated-latent MSE by
  denoise fraction: `0.25=0.8602502048015594`, `0.50=0.8611972332000732`,
  `0.75=0.8614047169685364`, `1.00=0.8634567856788635`. Diagnostic
  `output/diagnose_loop49_visualtokens_hist2_gt_eval2_3_spe16/idm_diagnostics.json`
  still scored `future_blind=true`, current-repeated
  `output_delta_mse=5.351210893422831e-05`, future-ranking `rank_accuracy=0.1875`,
  and `real_vs_best_negative_gap=-0.0017558876425027847`. Interpretation: prefixing
  Wan latent tokens improves internal validation, but does not make the IDM use future
  quality; partial/full generated futures remain flat.
- **Scheduled ranking + LingBot-style future-only latent noise (Loop 50)** This combines
  Loop49 token conditioning with stronger GT Wan-latent noise (`p=0.5`,
  `s_aug in [0.5,1.0]`, `future_only`) and scheduled ranking (`weight=0.1`,
  `start_epoch=8`, `ramp_epochs=8`, all negatives). Artifact:
  `output/metaworld_loop_50_diverse44_train2_spe4_flow_dit350m_wanvae_cached_visualtokens_latnoise_p05_s05_10_futureonly_contrast_w05_rank_w01_start8_ramp8_hist2_tmax025_h4_endpoint01`.
  The schedule was added in code via `idm_future_ranking_start_epoch` and
  `idm_future_ranking_ramp_epochs`; focused `ruff` and
  `pytest tests/test_training_smoke.py tests/test_run_idm_experiments.py` passed
  (`63 passed, 7 warnings`). Best epoch 10 internal `idm_mse=3.1447611127580917`
  with active ranking weight `0.025`; final epoch 20 internal
  `idm_mse=3.214320809500558` with active ranking weight `0.1`.
  External best-checkpoint GT eval
  `output/eval_loop50_visualtokens_latnoise_p05_futureonly_rank_sched_w01_hist2_gt_eval2_3_spe16`
  scored `idm_mse=0.8426768779754639`, a small improvement over Loop49. Generated-latent
  MSE by denoise fraction:
  `0.25=0.836714118719101`, `0.50=0.8371986448764801`,
  `0.75=0.8380154669284821`, `1.00=0.8414197862148285`. Diagnostic
  `output/diagnose_loop50_visualtokens_latnoise_p05_futureonly_rank_sched_w01_hist2_gt_eval2_3_spe16/idm_diagnostics.json`
  still scored `future_blind=true`, current-repeated
  `output_delta_mse=1.647020053496817e-05`, future-ranking `rank_accuracy=0.0`, and
  `real_vs_best_negative_gap=-0.0040329869370907545`. The final checkpoint was also
  checked: GT `idm_mse=0.873070627450943`, `future_blind=true`, current-repeated
  `output_delta_mse=0.00017608541384106502`, `rank_accuracy=0.3125`, and
  `real_vs_best_negative_gap=-0.001096394204068929`. Interpretation: Loop50 improves
  absolute action error for the selected checkpoint, but scheduled ranking at `0.1` still
  does not make real futures preferable to repeated/shuffled/noisy futures.
- **Loop47/48/49/50 next step** Scalar ranking weight alone, prefix token conditioning
  alone, and a gentle scheduled-ranking warmup are all insufficient. The next
  implementation target should move beyond prefix tokens toward explicit action-token
  cross-attention over Wan latent tokens, visual history tokens, or a candidate-consistency
  objective that rewards correct action differences between futures without letting the
  model exploit easy negative artifacts.
- **LingBot-VA plan note (arXiv 2601.21998)** Inspired by shared video/action latent and
  closed-loop causal WM framing, prioritize future-ranking supervision, partial/noisy GT
  Wan VAE latent IDM training, generated-latent eval, and richer attention over Wan
  latent tokens while keeping the causal WM -> IDM design. Loop49 tested prefix-token
  conditioning and remained future-blind; Loop50 added stronger future-only noisy-latent
  training plus scheduled ranking and still remained future-blind. A stronger architecture
  or objective is required. Treat variable chunk-size later as action-prefix masking only;
  consider an offline Wan DiT hidden-token ablation.
- **Partial-denoised generated Wan latent support** Implemented generated-latent cache
  support for returning Wan latents before VAE decode, inspired by LingBot-VA:
  `stop_after_steps` may be less than `num_inference_steps`. Cache metadata now records
  `denoise_steps_run`, `completed_denoise_steps`, `denoise_fraction`, and
  `denoise_mode`, and cached row metadata is validated against cache config. Fake
  full/partial eval smoke with the Loop43 checkpoint passed plumbing checks:
  full fake-cache MSE `2.551856279373169`, partial fake-cache MSE
  `2.537911891937256` (random fake latents, not quality evidence). One real-sample
  partial cache succeeded at
  `output/generated_wan_latent_cache_real_lora3_e5_epoch4_ep2_spe1_h4_partial_s1of2`;
  eval output `output/eval_loop43_real_generated_wan_latent_lora3_e5_epoch4_ep2_spe1_h4_partial_s1of2`
  scored MSE `1.7400734424591064` with `denoise_fraction=0.5` and
  `denoise_steps_run=1/2`. This is one sample only, not a benchmark. Next: run a real
  paired full-vs-partial sweep on the same samples with more than one sample, comparing
  quality and latency.
- **Paired real generated Wan latent sweep / timing** Same 4 real samples
  (`episodes=2,3`, `samples_per_episode=2`, LoRA
  `output/wan_metaworld_ep0_15_1424_128_lora3_e5/epoch-4.safetensors`, checkpoint
  `/tmp/wan2.2-ti2v-5b`, `num_frames=5`, `h=4`, `image_size=64`, `base_seed=710`).
  Cache dirs:
  `output/generated_wan_latent_cache_real_lora3_e5_epoch4_ep2_3_spe2_h4_full_s4`
  and
  `output/generated_wan_latent_cache_real_lora3_e5_epoch4_ep2_3_spe2_h4_partial_s2of4`.
  `/usr/bin/time` outer cache times were `33.57s` full vs `32.86s` partial (`1.02x`);
  the short job is dominated by model load / non-generation overhead. Loop43 eval dirs
  were
  `output/eval_loop43_real_generated_wan_latent_lora3_e5_epoch4_ep2_3_spe2_h4_full_s4`
  and
  `output/eval_loop43_real_generated_wan_latent_lora3_e5_epoch4_ep2_3_spe2_h4_partial_s2of4`.
  Eval through Loop43 IDM: full `idm_mse=2.0394465923309326`,
  `smooth_l1=0.670330822467804`; partial `idm_mse=2.0259952545166016`,
  `smooth_l1=0.6699042320251465`; baseline MSE `0.47247135639190674`. This tiny
  `n=4` paired sweep is plumbing/timing evidence only, not a performance claim.
- **Generated-latent cache timing instrumentation** Result JSON now reports
  `elapsed_wall_seconds`, `generator_load_wall_seconds`, `generation_wall_seconds`,
  `generation_wall_seconds_mean`, and `write_wall_seconds`; manifest rows include
  `generation_wall_seconds`. Timed preload smoke after separating load/generation:
  full one-sample cache
  `output/generated_wan_latent_cache_real_lora3_e5_epoch4_ep2_spe1_h4_full_s4_preload_timed`
  had load `24.01099926792085s` and generation `0.8410241459496319s`; partial
  `output/generated_wan_latent_cache_real_lora3_e5_epoch4_ep2_spe1_h4_partial_s2of4_preload_timed`
  had load `23.934744254220277s` and generation `0.6858347128145397s`. An earlier
  two-sample timed run before preload separation showed warm-row full
  `0.3788937502540648s` vs partial `0.22213194612413645s`. Next: use a warmed/preloaded
  generator over a larger sample count (`32+`) or server path so load is amortized, and
  compare partial denoise levels `1/4`, `2/4`, `3/4`, `4/4` on the same samples.
- **32-sample generated Wan latent denoise sweep** Same 32 real samples
  (`episodes=2,3`, `samples_per_episode=16`, `h=4`, `image_size=64`, `base_seed=740`,
  LoRA `output/wan_metaworld_ep0_15_1424_128_lora3_e5/epoch-4.safetensors`, checkpoint
  `/tmp/wan2.2-ti2v-5b`, `num_frames=5`). Cache dirs:
  `output/generated_wan_latent_cache_real_lora3_e5_epoch4_ep2_3_spe16_h4_partial_s1of4_timed`,
  `output/generated_wan_latent_cache_real_lora3_e5_epoch4_ep2_3_spe16_h4_partial_s2of4_timed`,
  `output/generated_wan_latent_cache_real_lora3_e5_epoch4_ep2_3_spe16_h4_partial_s3of4_timed`,
  and
  `output/generated_wan_latent_cache_real_lora3_e5_epoch4_ep2_3_spe16_h4_full_s4_timed`;
  eval dirs use the same suffixes under
  `output/eval_loop43_real_generated_wan_latent_lora3_e5_epoch4_ep2_3_spe16_h4_*_timed`.
  Summary artifact:
  `output/generated_wan_latent_denoise_sweep_ep2_3_spe16_h4_summary.json`.

  | denoise | gen mean | speedup vs full | idm_mse | smooth_l1 |
  |---|---:|---:|---:|---:|
  | 1/4 | 0.15733013988938183s | 2.6040902724477273x | 1.4723454117774963 | 0.5041956752538681 |
  | 2/4 | 0.23713099883752875s | 1.7277449547179966x | 1.4754501581192017 | 0.505203127861023 |
  | 3/4 | 0.31095940466912s | 1.317541391889168x | 1.481014370918274 | 0.5071443021297455 |
  | 4/4 | 0.4097018868487794s | 1.0x | 1.4951329231262207 | 0.5108166933059692 |

  Mean-action baseline MSE was `0.414536714553833`, so all generated-latent IDM outputs
  remain much worse than baseline. This is useful latency/plumbing evidence, not a solved
  model-quality result. Next: prioritize IDM quality/domain gap by training/evaluating
  the IDM on generated-latent-like noisy/partial latents at larger scale and/or improving
  the generated Wan LoRA, rather than only optimizing denoise count.
- **Matched generated-vs-GT Wan latent gap diagnostic** Added
  `compare_wan_latent_caches.py` with `tests/test_compare_wan_latent_caches.py`.
  Matched GT cache:
  `output/wan_vae_latent_cache_eval2_3_spe16_h4_matched` (32 samples, episodes 2 and
  3, `samples_per_episode=16`, `h=4`, image 64, same dataset indices as the generated
  sweep). Diagnostic output:
  `output/latent_gap_eval2_3_spe16_h4_generated_sweep/latent_gap_summary.json` and
  `.md`.

  | denoise | latent_mse | normalized_mse | cos | gen/ref norm | t0_mse | future_mse |
  |---|---:|---:|---:|---:|---:|---:|
  | 1/4 partial | 0.7333109515504052 | 1.2678196935551698 | 0.40858491096395483 | 1.0669369821641852 | 3.132800670650517e-05 | 1.466590575094104 |
  | 2/4 partial | 0.5993694460628528 | 1.036248518895201 | 0.48311012621102656 | 1.0005018311920884 | 3.132800670650517e-05 | 1.1987075641189993 |
  | 3/4 partial | 0.3821384387358955 | 0.6606783074348951 | 0.640943607277247 | 0.898623879792035 | 3.132800670650517e-05 | 0.7642455494650845 |
  | 4/4 full | 0.11888106770563066 | 0.20553321685623854 | 0.9073967237728605 | 0.9552873810704272 | 3.132800670650517e-05 | 0.2377308074045548 |

  The first latent time slice is near-identical because it carries/encodes the
  conditioning frame; the future slice carries almost all of the gap. Full denoise is
  much closer to GT latents, but earlier IDM action MSE was still poor and not monotonic
  with latent closeness. This points to generated-latent IDM robustness/domain adaptation
  as the next bottleneck, not denoise count alone. LingBot-VA-inspired next direction:
  keep the causal modular design (`Wan text + current image -> future frames/latents ->
  separate IDM`), but borrow noisy-history/partial-denoise action robustness by
  training/evaluating IDM on GT Wan VAE latents with calibrated flow-noise or
  partial-latent augmentation, and possibly history tokens. Do not train IDM on
  WM-generated frames unless that constraint is explicitly changed.
- **Verification** Focused checks passed:
  `uv run ruff check world_model/diffsynth_wan.py world_model/data.py cache_generated_wan_latents.py eval_idm.py tests/test_diffsynth_wan.py tests/test_generated_wan_latent_cache.py tests/test_cache_generated_wan_latents.py tests/test_eval_idm.py`;
  `uv run pytest -q tests/test_diffsynth_wan.py tests/test_generated_wan_latent_cache.py tests/test_cache_generated_wan_latents.py tests/test_eval_idm.py`
  passed with `67 passed, 3 warnings`;
  `uv run pytest -q tests/test_compare_wan_latent_caches.py` passed with `4 passed`;
  `uv run ruff check compare_wan_latent_caches.py tests/test_compare_wan_latent_caches.py`
  passed with all checks clean; the real compare command completed and wrote the summary
  artifacts. Loop44 follow-up checks passed:
  `uv run pytest -q tests/test_training_smoke.py -k "wan_vae_latent_noise or train_idm_main_forwards_wan_vae_latent_noise_options or train_idm_main_forwards_flow_train_time_range"`
  passed with `9 passed, 34 deselected`;
  `uv run ruff check world_model/config.py world_model/train_lib.py train_idm.py tests/test_training_smoke.py compare_wan_latent_caches.py tests/test_compare_wan_latent_caches.py`
  passed with all checks clean; and
  `uv run pytest -q tests/test_training_smoke.py tests/test_compare_wan_latent_caches.py`
  passed with `47 passed, 4 warnings`. Loop45 `eval_idm.py` and `diagnose_idm.py`
  commands completed and wrote the cited eval/diagnostic JSON artifacts. Loop46 eval and
  diagnose artifacts are present under the cited `output/eval_loop46_*` and
  `output/diagnose_loop46_*` paths.
- **Evidence** Metrics/diagnostics:
  `output/metaworld_loop_37_diverse44_train2_spe4_flow_dit350m_patch_additive_tmax025_h4_endpoint01/metrics.json`,
  `output/diagnose_loop37_patch_tmax025_gt_eval2_3_spe2/idm_diagnostics.json`,
  `output/metaworld_loop_39_diverse44_train2_spe4_flow_dit350m_wanvae_additive_tmax025_h4_endpoint01/metrics.json`,
  `output/diagnose_loop39_wanvae_tmax025_gt_eval2_3_spe2/idm_diagnostics.json`,
  `output/wan_vae_latent_cache_smoke_ep0_1_spe4`,
  `output/idm_cached_wanvae_smoke_ep0_1_spe4`,
  `output/idm_uncached_wanvae_smoke_ep0_1_spe4`,
  `output/wan_vae_latent_cache_diverse44_train2_spe4_h4`,
  `output/metaworld_loop_40_diverse44_train2_spe4_flow_dit350m_wanvae_cached_context_w1_warmup5_tmax025_h4_endpoint01/metrics.json`,
  `output/metaworld_loop_41_diverse44_train2_spe4_flow_dit350m_wanvae_cached_tmax025_h4_endpoint01/metrics.json`,
  `output/metaworld_loop_42_diverse44_train2_spe4_flow_dit350m_wanvae_cached_latnoise_p05_s05_10_tmax025_h4_endpoint01/metrics.json`,
  `output/wan_vae_latent_cache_diverse44_eval2_3_spe2_h4`,
  `output/eval_loop41_wanvae_cached_gt_eval2_3_spe2`,
  `output/diagnose_loop41_wanvae_cached_gt_eval2_3_spe2`,
  `output/eval_loop42_wanvae_cached_latnoise_gt_eval2_3_spe2`,
  `output/diagnose_loop42_wanvae_cached_latnoise_gt_eval2_3_spe2`,
  `output/metaworld_loop_43_diverse44_train2_spe4_flow_dit350m_wanvae_cached_latnoise_p025_s075_10_tmax025_h4_endpoint01/metrics.json`,
  `output/eval_loop43_wanvae_cached_latnoise025_gt_eval2_3_spe2`,
  `output/diagnose_loop43_wanvae_cached_latnoise025_gt_eval2_3_spe2`,
  `output/generated_wan_latent_cache_real_lora3_e5_epoch4_ep2_spe1_h4_partial_s1of2`,
  `output/eval_loop43_real_generated_wan_latent_lora3_e5_epoch4_ep2_spe1_h4_partial_s1of2`,
  `output/generated_wan_latent_cache_real_lora3_e5_epoch4_ep2_3_spe2_h4_full_s4`,
  `output/generated_wan_latent_cache_real_lora3_e5_epoch4_ep2_3_spe2_h4_partial_s2of4`,
  `output/eval_loop43_real_generated_wan_latent_lora3_e5_epoch4_ep2_3_spe2_h4_full_s4`,
  `output/eval_loop43_real_generated_wan_latent_lora3_e5_epoch4_ep2_3_spe2_h4_partial_s2of4`,
  `output/generated_wan_latent_cache_real_lora3_e5_epoch4_ep2_spe1_h4_full_s4_preload_timed`,
  `output/generated_wan_latent_cache_real_lora3_e5_epoch4_ep2_spe1_h4_partial_s2of4_preload_timed`,
  `output/generated_wan_latent_cache_real_lora3_e5_epoch4_ep2_3_spe16_h4_partial_s1of4_timed`,
  `output/generated_wan_latent_cache_real_lora3_e5_epoch4_ep2_3_spe16_h4_partial_s2of4_timed`,
  `output/generated_wan_latent_cache_real_lora3_e5_epoch4_ep2_3_spe16_h4_partial_s3of4_timed`,
  `output/generated_wan_latent_cache_real_lora3_e5_epoch4_ep2_3_spe16_h4_full_s4_timed`,
  `output/generated_wan_latent_denoise_sweep_ep2_3_spe16_h4_summary.json`,
  `output/wan_vae_latent_cache_eval2_3_spe16_h4_matched`,
  `output/latent_gap_eval2_3_spe16_h4_generated_sweep/latent_gap_summary.json`,
  `output/latent_gap_eval2_3_spe16_h4_generated_sweep/latent_gap_summary.md`,
  `output/metaworld_loop_44_diverse44_train2_spe4_flow_dit350m_wanvae_cached_latnoise_p025_s075_10_futureonly_tmax025_h4_endpoint01/metrics.json`,
  `output/eval_loop43_wanvae_cached_latnoise025_gt_eval2_3_spe16`,
  `output/eval_loop44_wanvae_cached_futureonly_gt_eval2_3_spe16`,
  `output/eval_loop44_real_generated_wan_latent_lora3_e5_epoch4_ep2_3_spe16_h4_partial_s1of4_timed`,
  `output/eval_loop44_real_generated_wan_latent_lora3_e5_epoch4_ep2_3_spe16_h4_partial_s2of4_timed`,
  `output/eval_loop44_real_generated_wan_latent_lora3_e5_epoch4_ep2_3_spe16_h4_partial_s3of4_timed`,
  `output/eval_loop44_real_generated_wan_latent_lora3_e5_epoch4_ep2_3_spe16_h4_full_s4_timed`,
  `output/diagnose_loop44_wanvae_cached_futureonly_gt_eval2_3_spe16/idm_diagnostics.json`,
  `output/metaworld_loop_45_diverse44_train2_spe4_flow_dit350m_wanvae_cached_latnoise_p025_s075_10_contrast_w05_m01_tmax025_h4_endpoint01/metrics.json`,
  `output/eval_loop45_wanvae_cached_contrast_w05_gt_eval2_3_spe16`,
  `output/eval_loop45_real_generated_wan_latent_lora3_e5_epoch4_ep2_3_spe16_h4_partial_s1of4_timed`,
  `output/eval_loop45_real_generated_wan_latent_lora3_e5_epoch4_ep2_3_spe16_h4_partial_s2of4_timed`,
  `output/eval_loop45_real_generated_wan_latent_lora3_e5_epoch4_ep2_3_spe16_h4_partial_s3of4_timed`,
  `output/eval_loop45_real_generated_wan_latent_lora3_e5_epoch4_ep2_3_spe16_h4_full_s4_timed`,
  `output/diagnose_loop45_wanvae_cached_contrast_w05_gt_eval2_3_spe16/idm_diagnostics.json`,
  `output/eval_loop46_wanvae_cached_contrast_w05_hist2_gt_eval2_3_spe16`,
  `output/eval_loop46_real_generated_wan_latent_lora3_e5_epoch4_ep2_3_spe16_h4_hist2_partial_s1of4_timed`,
  `output/eval_loop46_real_generated_wan_latent_lora3_e5_epoch4_ep2_3_spe16_h4_hist2_partial_s2of4_timed`,
  `output/eval_loop46_real_generated_wan_latent_lora3_e5_epoch4_ep2_3_spe16_h4_hist2_partial_s3of4_timed`,
  `output/eval_loop46_real_generated_wan_latent_lora3_e5_epoch4_ep2_3_spe16_h4_hist2_full_s4_timed`,
  and `output/diagnose_loop46_wanvae_cached_contrast_w05_hist2_gt_eval2_3_spe16/idm_diagnostics.json`.

---

## Stale / invalid / superseded loops

Do **not** cite these as current results.

| Loop(s) | What ran | Why excluded | Evidence |
|---|---|---|---|
| 01–05 | IDM on episode 0 only, n=80 (unclamped → action-norm → horizon4 → unipi/resnet) | Collapsed to the mean-action baseline; learned no future signal | diagnostics `idm_mse` 0.343 / 0.343 / 0.347 / 0.344 / 0.343 ≈ baseline 0.33–0.35 |
| 06–08 (delta) | GT-future IDM, train ep0–15, eval ep16–19 — right direction, good numbers | Predate the hardened split (`split_gap=null`) → leaky internal val + optimistic checkpoint selection; records internally inconsistent (`train_config.model` nff/ah disagree with top-level `model_config`). Superseded by Loop 11 | loop06 diag 0.0409, loop07 0.0367, loop08 delta best 0.0468 |
| 08 (flow_transformer) | Flow-matching IDM variant | Worse than delta (internal best 0.1018; held-out recheck 0.0675). Experimental, not canonical | `diagnose_loop08_flow_gt_ep16_19_first128_recheck` 0.0675 |
| 09–10 | IDM trained on cached / mixed Wan futures, episode 0 only | Near-baseline (best 0.3322 / 0.2401). Approach abandoned — IDM now trains on GT futures only; `train_idm.py` rejects cache flags | loop09 best 0.3322, loop10 best 0.2401 |
| Older `wan_lora_ranking_*` (ep16_19 first64 `gt_ref=0.0521`; `ep0_*`; `existing_first16` `gt_ref=0.0157`) | Wan-LoRA rankings against a pre-Loop-11 IDM or single-episode data | Wrong/old IDM and/or no held-out episodes; superseded by the `_new_idm` rankings (`gt_ref` 0.0479 @ n64 / 0.0461 @ n128) | ranking_summary `ground_truth_reference.idm_mse` |

---

## Remaining gaps toward the objective

1. **One in-distribution task success exists, but broad success does not.** Wan
   evidence now includes offline cached Wan-LoRA futures scored against the IDM, a real
   Wan-LoRA direct policy smoke with finite `[4, 4]` actions, an 8-step MetaWorld smoke,
   three 50-step reach-v3 closed-loop probes, matched **full-horizon 3-episode reach-v3
   evals** (`max_steps=300`, epoch1/steps8 and epoch3/steps8) logging `success_rate=0.0`
   (0/3 each), a `replan_steps=1` full-horizon debug probe (n=1, also 0/1), and a
   two-task Loop-15 panel (`assembly-v3`, `disassemble-v3`, 0/1 each), and a
   Loop-20 50-step `reach-v3` smoke using the best offline Wan-decodable IDM (also
   0/1), and the Loop22 8-task task-diverse panel (2 episodes/task). Loop 20 produced one
   300-step in-distribution `assembly-v3` success (1/1), and Loop22 repeats that
   assembly-only competence at 1/2, but all seven non-assembly Loop22 tasks are 0/2. The
   300-step runs are benchmark-shaped closed loops (full horizon, multiple episodes,
   latency instrumented), but broad task-diverse success is now negative at small n.
   Offline stochastic Wan seed sweeps exist, but no online best-of-N run has been
   attempted because the pixel-best seed was not action-best.
2. **Task-success metrics remain open beyond assembly.** Every strong broad-comparison
   number here is still offline (IDM MSE, pixel MSE/PSNR, decodability gap). Real
   Wan-LoRA closed-loop evidence logs `success_rate=0.00` for every non-assembly run:
   0/1 for the 8-step smoke and each of the three
   50-step probes, **0/3 for both full-horizon 300-step reach-v3 evals**, and 0/1 for the
   `replan_steps=1` full-horizon debug probe, plus 0/1 each for `assembly-v3` and
   `disassemble-v3` in Loop 15, and 0/1 for the Loop-20 50-step `reach-v3` smoke.
   Loop22 adds 0/2 each for `disassemble-v3`, `reach-v3`, `push-v3`, `door-open-v3`,
   `peg-insert-side-v3`, `pick-place-v3`, and `sweep-v3`. Loop 20's later `assembly-v3`
   300-step run is the first success (1/1), and Loop22 logs `assembly-v3` at 1/2. Of the
   ARCHITECTURE "Experiments To Run", full closed-loop receding-horizon rollouts over
   multiple episodes and multiple tasks are now demonstrated at small n; **online
   best-of-N future selection and broader episode coverage remain open**, and only the
   in-distribution assembly runs have produced task success. Loop 14 is an offline
   best-of-N precursor and argues against a pixel-only selector.
3. **Wan→action gap is material.** The strongest current-valid decoded-video modular
   generated-cache diagnostic scores `idm_mse=0.06194690754637122` with
   `future_blind=false` against mean baseline `0.33408883213996887`; the same-family
   prior GT reference is `0.04420376801863313`. The matched 256-sample decoded-video
   smoke result (`0.160416`) is not the best bottleneck evidence because same-split GT
   is essentially identical (`0.160341`). Loop 20's 50-step `reach-v3` smoke still
   scores 0/1, so the offline gap has not yet converted into task success. The matched
   generated-vs-GT latent diagnostic shows full denoise is much closer to GT latents than
   partial denoise, while IDM action MSE remains poor and not monotonic with latent
   closeness. Loop46 reduces matched external MSE versus Loop45 but is future-blind;
   Loop47/48 show scalar ranking loss is expensive and not enough by itself, while
   Loop49/50 prefix-token and scheduled-ranking variants still remain future-blind. The
   no-ranking patch-token cross-attention plus `future_delta` flow IDM now scores GT
   `idm_mse=0.04860944184474647` and generated-Wan
   `idm_mse=0.07012095977552235` on episodes 16-23 with `samples_per_episode=32`, with
   `future_blind=false`; the remaining generated-Wan gap is about `0.02151` MSE, and no
   closed-loop success has been demonstrated for this checkpoint.
4. **Task-diverse IDM is near-baseline.** Loop 21 adds balanced per-episode sampling and
   trains on 2 demos per task, but external GT-future eval on the next demo per task is
   only slightly better than mean-action (`idm_mse=5.0121` vs baseline `5.4063`). This
   means broad failure is not only a Wan problem; the IDM/data/normalization recipe is
   not yet strong enough across task families.
5. **Narrow high-quality held-out coverage.** The strong offline Wan numbers still come
   from episodes 16–19, n≤128, in an assembly-style slice. Balanced task-diverse sampling
   exists, and Loop22 adds closed-loop task-diverse evidence, but the task-diverse result
   is negative beyond assembly and still only n=2 per task. Comparable task-diverse
   Wan-cache, IDM, and WM training results are still missing.
6. **Selector unvalidated against success.** Ranking by `idm_decodability_gap` is the
   chosen world-model selector, but since pixel-best ≠ decodability-best, the choice has
   not been confirmed against real closed-loop task success. The 50-step n=1 probes and
   matched 300-step n=3 comparison are directionally consistent with the ranking
   (epoch1 higher dense reward than epoch3), but both full-horizon checkpoints still score
   0/3, and Loop 20's improved offline Wan-decodability still produced 0/1 on a 50-step
   smoke. Loop 14 repeats the same mismatch under stochastic seeds (seed1007 is
   pixel-best, seed7 is action-best), so the selector remains unvalidated against success
   and naive pixel-only best-of-N is not supported.
7. **Latency is now instrumented end-to-end and per-stage.** The MetaWorld client records
   both the server-reported `server_timing.infer_ms` and client round-trip per request
   into each `episode_XXX.json` (`server_timing_ms` / `client_timing_ms`), and the server
   now reports stage timings under `server_timing_ms.stages`. Loop 15 shows Wan future
   generation dominates (`~2.77-3.08 s/request` mean), while IDM decode is only
   `~3-5 ms/request`; Loop22's warm server p50 is lower but still Wan-dominated at about
   `922-928 ms`. Still open: use the per-stage timing to evaluate smaller Wan step
   counts, lower resolution, caching, or batched future generation.
8. **Single-view / resolution mismatch untested.** The IDM is one view (`corner4`) at
   64px; Wan inference resolution/view alignment is not yet stress-tested end-to-end.

---

## Reproduction notes

- Metrics were read directly from the JSON files cited per loop; cross-checks pass —
  e.g. `diagnose_loop11_wan_epoch3_ep16_19_first128` and the first-128
  `ranking_summary.json` both report `idm_mse=0.0751`, and the Loop 12 / first-64
  `gt_ref` (0.0479) equals `diagnose_loop11_gt_ep16_19_first64`.
- 300-step reach-v3 run: per-episode rewards (`1911.44 / 1479.42 / 748.78`) and
  `success_rate=0.0` were read from `reach-v3/episode_00{0,1,2}.json` and aggregate
  `results.json`; the 225 server request dirs equal 75 requests/episode × 3, and ep0's
  `25511 ms` first request is a one-time model load (warm mean excluding it is `933.5 ms`,
  matching ep1/ep2's `941.1 / 935.4 ms`).
- `replan_steps=1` debug probe: `success_rate=0.0` and `mean_reward=1649.14` read from
  `world_model_wan_lora_reach300_epoch1_steps8_replan1_eval/episode_000.json`; the 300
  server request dirs equal 300 steps × 1 replan, and the `25328.52 ms` first request is a
  one-time model load (warm mean excluding it is `928.55 ms`, p50 `926.29 ms`).
- Matched epoch3 full-horizon run: per-episode rewards (`1430.30 / 1280.96 / 988.36`)
  and `success_rate=0.0` were read from
  `world_model_wan_lora_reach300_epoch3_steps8_eval_n3/reach-v3/episode_00{0,1,2}.json`
  and aggregate `results.json`; the 225 server request dirs equal 75 requests/episode × 3.
- Loop 13 h1 diagnostic: metrics were read from
  `horizon_loop13_h1_{gt,wan_epoch1,wan_epoch3}_ep16_19_first64/idm_horizon_diagnostics.json`.
  The original h4 Wan caches were correctly rejected for h1 (`action_horizon: 4 != 1`),
  so h1-compatible Wan caches were regenerated before scoring.
- Loop 14 seed sweep: metrics were read from
  `metaworld_loop_14_epoch1_seed_sweep_ep16_19_first64_rank5/ranking_summary.json`.
  The seed17/seed23 cache configs were refreshed after adding `--generation-seed` so their
  dataset seed stays fixed at 7 while their per-row Wan generation seeds remain varied.
- Loop 15 panel: aggregate success was read from
  `examples/metaworld/output/loop15_panel_assembly_epoch1_n3/results.json`, and per-task
  reward/request/stage timing was read from
  `assembly-v3/episode_000.json` and `disassemble-v3/episode_000.json`.
- Current-valid decoded-video modular recheck: generated-cache metrics were read from
  `output/diagnose_loop20_current_valid_cache_recheck_ep16_19_first64/idm_diagnostics.json`
  and cache metadata from `output/loop16_cache_epoch1_f17_steps8_ep16_19_first64`; the
  same-family prior GT reference was read from
  `output/diagnose_loop20_noise012_state_norm_gt_ep16_19_first64/idm_diagnostics.json`.
- Loop 20 assembly success: `success_rate=1.0`, `mean_reward=494.03`, and 72 requests
  were read from `examples/metaworld/output/loop20_wan_epoch1_assembly300_eval/episode_000.json`.
  Visual sheets were generated from the client MP4 and server Wan request MP4s into
  `output/visual_debug_loop20_assembly/`.
- Loop 21 task-diverse IDM smoke: training best (`idm_mse=6.4072`) was read from
  `output/metaworld_loop_21_diverse44_train2_delta_h4_noise012_state_norm_fast/metrics.json`;
  external balanced GT eval (`idm_mse=5.0121`, baseline `5.4063`) was read from
  `output/diagnose_loop21_diverse44_train2_gt_eval1_spe8/idm_diagnostics.json`.
- Loop22 task-diverse closed-loop panel: aggregate success was read from
  `examples/metaworld/output/loop22_taskdiverse_loop20_gpu0/results.json` and
  `examples/metaworld/output/loop22_taskdiverse_loop20_gpu1/results.json`; per-task
  reward/request/latency summary was recorded from the two client/GPU panels and server
  timings. Visual rollout/future sheets were generated under
  `examples/world_model_env/output/visual_debug_loop22_taskdiverse/`.
- `output/` is git-ignored; this ledger references artifacts as evidence without
  committing them.
- Smoke/contract verification: focused cache-seed tests
  (`test_cache_future_rollouts_writes_wan_lora_cache`,
  `test_cache_future_rollouts_can_vary_wan_generation_seed_without_dataset_seed`,
  `test_cache_future_rollouts_rejects_wan_lora_resume_with_different_generation_seed`,
  `test_build_wan_generators_propagate_conditioning_frame_contract`) passed; full
  `uv run pytest -q` was rerun after the Loop 14 updates.
