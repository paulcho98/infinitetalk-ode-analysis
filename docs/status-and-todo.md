# Status & TODO — start here to continue

This is the authoritative "where we are / what remains / how to proceed" doc. Read the root `README.md`
first for the five load-bearing facts, then this.

> **UPDATE (2026-07-26) — steps 1-5 and the Euler-jump factorial are DONE.** The 8-GPU sweep,
> Stage-2a metrics, Stage-2b geometry, the Stage-2c plotters and the **Euler-jump factorial** have
> all completed; see `results/infinitetalk/findings.md` + `results/infinitetalk/figures/`. What remains:
>
> - **← CURRENT PRIORITY: re-run Stage 2b** to pick up the true `{region}_velocity` metric and the
>   `delta_cosine[0]` fix (see the "Corrected in this revision" note in `results/infinitetalk/findings.md`). The
>   committed geometry JSONs and per-config figures predate both. It was deprioritized behind the
>   Euler-jump work, which is now finished — so this is next. Stage 2b is correspondingly off by
>   default in the Euler Stage-2 launchers (`RUN_2B=1` enables it). The perceptual CSVs are
>   unaffected by either fix.
> - **Step 6, the OmniAvatar cross-model comparison** — **UNBLOCKED as of 2026-07-29**, not started.
>   The OmniAvatar results are absent from the *sweep* machine but present and complete on the
>   **OmniAvatar machine** (`/home/work/.local/{ode_analysis,ode_full_trajectories}`), where they were
>   verified schema- and sample-compatible with this repo's CSVs. The Euler-jump Factorial B cells are
>   a 1:1 replication of OmniAvatar's four `14B_textaudio_euler_*` CSVs and are the natural first diff.
>   One transfer is required first: the per-cell euler `metrics.csv` files are gitignored, so only
>   terminal-step values crossed over. **Read `docs/cross-model-comparison.md`** — machine map,
>   inventory, the three gaps, and the normalization caveats.
>
> **Euler-jump summary:** 7 cells × 10 samples × 50 landing steps, ~12.7 h on 7×A100. Both
> pre-flight checks passed. Curvature is created by *audio* guidance at step 0 (text contributes
> ~nothing); Sync-C peaks at landing step 11–15, not at the end. Full write-up in
> `results/infinitetalk/findings.md` and `docs/euler-jump-experiment.md`.

## What has been done and VALIDATED

1. **Environment** (`docs/environment.md`): `infinitetalk` conda env built; all InfiniteTalk deps resolve
   (the tricky ones: `transformers==4.49.0`, `diffusers==0.33.1`, flash-attn prebuilt wheel; `dlib-bin`
   added for Stage-2). `generate_infinitetalk.py --help` and a full 20-step generation both run.
2. **Checkpoints** downloaded (Wan2.1-I2V-14B-480P, MeiGen-AI/InfiniteTalk, chinese-wav2vec2-base).
3. **Stage-1 driver** `scripts/infinitetalk/generate_infinitetalk_ode_pairs_full.py`:
   - Subclasses `InfiniteTalkPipeline`; forks the denoising loop; derives `x0 = x_t + sigma·noise_pred`.
   - Reproduces InfiniteTalk's exact CFG branching (3-call; (1,1) → 1-pass).
   - **FORCE-SQUARE** patch applied and verified: a non-square (2:3) reference now yields `[16,21,80,80]`.
   - VALIDATED end-to-end: decoded `x0` is a coherent talking-head (see `examples/frames/`).
   - Multi-config per model load; shards by sample; caches wav2vec audio per hash.
4. **Stage-1 launcher**: `scripts/infinitetalk/run_infinitetalk_ode_sweep_8gpu.sh` (job-sharded, 8-GPU) ran the
   full 7-config sweep to completion — 70 trajectories, ~11 h, now on disk in
   `ode_full_trajectories_infinitetalk/` (~29 GB, gitignored). `run_infinitetalk_ode_sweep.sh` is
   the older 4-GPU variant.
5. **Stage-2 engines** (`eval_ode_perceptual_v2_infinitetalk.py`, `analyze_ode_trajectory_infinitetalk.py`):
   both ran to completion on the full sweep. For 2b, mask-derivation (dlib mouth-bbox) + WanVAE
   decode/encode paths validated on real data; the GT reader was fixed to center-crop-to-square
   (matches the generator). 2a's GT reader already center-crops.
6. **Euler-jump factorial** (`scripts/infinitetalk/generate_infinitetalk_euler_jump.py` +
   `run_infinitetalk_euler_jump.sh`): all 7 cells complete, ~12.7 h on 7×A100, output in
   `ode_euler_jump_infinitetalk/`. Both pre-flight checks in `docs/euler-jump-experiment.md` passed.
   Stage-2 via `run_stage2_euler_jump_sharded.sh` (shards the SyncNet-bound metrics phase; the
   unsharded `run_stage2_euler_jump.sh` is correct but much slower).

## What is NOT yet done / not validated on final data

> The section that used to sit here ("the full sweep never finished / no trajectories persist on
> disk / plotters untouched") described the state at handoff and was superseded long ago. Everything
> it listed has since run to completion. Accurate remaining list:

- **Stage 2b needs a re-run** to populate the true `{region}_velocity` metric and pick up the
  `delta_cosine[0]` fix. The committed `results/infinitetalk/data/geometry_*.json` and the per-config 2b figures
  predate both, so the velocity panel is the only untrustworthy output in `results/`. The perceptual
  CSVs and the Euler-jump results are unaffected.
- **Stage 2b was not run for the Euler-jump cells at all** (deliberate — `RUN_2B=1` enables it).
  It measures distance-to-GT, which is secondary to distance-to-sequential for that experiment.
- **Cross-model comparison vs OmniAvatar** (original step 6) — not started, but no longer blocked:
  the OmniAvatar results are on the OmniAvatar machine. Needs the 7 euler `metrics.csv` files moved
  off the sweep machine (~3.5 MB) before the Factorial-B diff can be done at per-step granularity.
  See `docs/cross-model-comparison.md`.
- **`results/infinitetalk/data/` does not contain per-step euler perceptual metrics.** Only
  `straightness_*.json` (curvature) and a terminal-step-only, mouth-only
  `figures/euler_jump/euler_terminal_values.csv` are committed. The per-cell
  `ode_analysis_euler_jump/euler_*/perceptual_v2/metrics.csv` are gitignored and exist only on the
  sweep machine — and they hold the step 11–15 peak that the terminal values miss entirely.

## HOW TO PROCEED (ordered)

> **Steps 0–5b below have all been executed** — they are kept as the runbook for reproducing the study
> on a fresh machine, not as a to-do list. The only outstanding items are the Stage-2b re-run (top of
> this doc) and **step 6**, which has been rewritten to reflect its unblocked state.

### 0. Re-point paths
Repo-internal paths are now portable: the launcher derives `$REPO` from its own location and reads
inputs from `data/recon_clips/` + `data/recon_sample_names.txt`; Stage-2 GT dirs default to
`data/recon_clips/`. So on a fresh clone, `--video_dir`/`--audio_dir` and the Stage-2 GT dirs need NO edits.

The EXTERNAL paths you MUST still edit:
- `INFINITETALK_ROOT` (top of the Stage-1 driver and both Stage-2 scripts) -> your InfiniteTalk repo clone.
- `IT=` (weights) and `PY=` (env python) in `run_infinitetalk_ode_sweep.sh`.
- `METRICS_ROOT` in `eval_ode_perceptual_v2_infinitetalk.py` -> your `eval_metrics/` (shape_predictor + syncnet).
- **SyncNet code**: the eval script imports `eval.syncnet.SyncNetEval` / `eval.syncnet_detect.SyncNetDetector`
  (custom wrappers from the original machine / public `syncnet_python`). Put them on `PYTHONPATH`, or pass
  `--skip_metrics sync` to skip Sync-C/D.
- `--output_dir` / output root for each stage (the ~18GB trajectory dump).

### 1. Run the Stage-1 sweep (~8 h, 4 GPUs)
```bash
bash scripts/infinitetalk/run_infinitetalk_ode_sweep.sh 50
```
Sanity while it runs: every `step_000_xt.pt` must be `[16,21,80,80]` (NOT 64×96). If any is non-square,
the force-square patch didn't take — check `generate_infinitetalk_ode_pairs_full.py` (the
`target_h=target_w=640` block). 70 trajectories total (10 samples × 7 configs).

### 2. Validate Stage 2a on ONE finished config (decode → metrics → merge)
- Phase 1 in `infinitetalk` env, Phase 2 in `omniavatar` env (NOT `latentsync-metrics` — no lpips).
- Sanity: reference metrics (MSE/LPIPS) should DECREASE across steps; SyncNet confidence should rise.
  Spot-check that the dlib mouth-bbox actually lands on the mouth (dump one crop).

### 3. Re-validate Stage 2b on ONE finished config (should pass now)
```bash
python scripts/infinitetalk/analyze_ode_trajectory_infinitetalk.py --traj_dir <cfg> --output_dir <out> \
  --gt_mode encode --gt_video_dir <hallo3_bench> --mask_source ref_decode \
  --mouth_mask_cache <cache> --shape_predictor <eval_metrics>/shape_predictor_68_face_landmarks.dat
```
Watch: the earlier crash was a dims mismatch (GT 64×96 vs x0 80×80) — gone now that both are square.

### 4. Run 2a + 2b across all 7 configs (parallel)
2a and 2b only read the `.pt`s and are GPU-light (VAE + small nets). Run them concurrently; shard 2a's
decode across GPUs with `--shard_id/--num_shards`.

### 5. Stage 2c — plotters (see `docs/stage2-audit.md` for exact per-script changes)
- 6 plotters are **PARAM** (config-list repoints): change their hardcoded OmniAvatar variant dicts
  (`14B_cfg3.0` etc.) + `TRAJ_DIR` to the 7 `infinitetalk_t{T}_a{A}/perceptual_v2` dirs. Drop the
  `upper_face` panels (no analogue — regions are `mouth` + `full` only).
- `plot_mouthweight_ode_results.py` is **REWORK** (1-D cfg frontier) — skip or rebuild for 2-D.
- **Build a NEW 2-D `t×a` heatmap / Pareto view** — the natural read of this grid (quality vs the two
  guidance scales). This is the one genuinely new artifact.

### 5b. Euler-jump factorial (NEW — ported, not yet run)
Full design + commands in **`docs/euler-jump-experiment.md`**. Measures ODE straightness: jump from
step 0 straight to each noise level and re-predict, then compare against the sequential trajectory.
Two overlapping 2×2s over (step-0 CFG) × (teacher CFG) with `on=(5,4)`, `noaudio=(5,1)`,
`nocfg=(1,1)` — 7 distinct cells (the on/on cell is shared).
```bash
bash scripts/infinitetalk/run_infinitetalk_euler_jump.sh 50      # generate, 7 cells across 8 GPUs
bash scripts/infinitetalk/run_stage2_euler_jump.sh all           # metrics + geometry (same Stage-2 stack)
python scripts/infinitetalk/plot_euler_jump_factorial.py ...     # figures
```
Prereq: Stage-1 trajectories for t5.0_a4.0, t5.0_a1.0, t1.0_a1.0 (all in the standard sweep).
**Smoke-test `euler_on_on` first** — its step 0 should land very close to the sequential trajectory's.

### 6. Compare against OmniAvatar  ← THE REMAINING WORK
The point of the study: put InfiniteTalk's ODE-straightness/quality-vs-CFG next to OmniAvatar's
(`/home/work/.local/ode_analysis`, on the **OmniAvatar machine**). Same 10 Hallo3 identities, same step
count, same CSV schema — all three verified 2026-07-29.

**`docs/cross-model-comparison.md` is the handoff doc for this step.** It covers which machine holds
what, the full OmniAvatar-side inventory, the three gaps (missing per-step euler CSVs; stale Stage-2b;
no OmniAvatar straightness counterpart, because those runs never saved latents and the cfg-4.5
trajectory dump is gone), which OmniAvatar experiment families have no InfiniteTalk analogue, and why
sharpness/pixel-MSE must be normalized per model before any cross-model claim.

## Deferred re-runs

Three re-runs are scripted and ready but deliberately not executed as part of the unified-repo
work (out of scope per the design spec — no new experiments, no logic changes, numerical behavior
of every ported script must stay unchanged). Each unblocks a specific downstream comparison:

**(a) OmniAvatar cfg=4.5 sequential trajectory + the 4 euler cells, with latents saved.**
The original OmniAvatar study never persisted the cfg-4.5 sequential `.pt` dump (it's gone — see
`docs/cross-model-comparison.md` gap C) and never saved euler-cell latents at all. This is why
`docs/status-and-todo.md`'s / `results/omniavatar/data/`'s Stage-2b (latent geometry) has **no**
OmniAvatar counterpart today, and why the straightness number (`‖x0_euler − x0_sequential‖`,
`scripts/common/measure_euler_straightness.py`) can only be computed on the InfiniteTalk side.
`scripts/omniavatar/generate_ode_trajectories.sh` already writes `step_NNN_xt.pt`/`step_NNN_x0.pt`
per step (no re-run needed there beyond re-executing it); `generate_single_step_predictions.py`
now has `--save_latents` (added in this repo, off by default, dumps `step_NNN_x0.pt` per landing
step) — re-run `scripts/omniavatar/run_single_step_both.sh`'s 4 cells with that flag added to
unblock the OmniAvatar-side straightness measurement. Registry ids: `omni_ta_default` (sequential)
+ the 4 `omni_ta_euler_*` ids. **Unblocks:** cross-model ODE-straightness comparison.

**(b) InfiniteTalk Stage-2b re-run on the sweep machine.**
`results/infinitetalk/data/geometry_*.json` and the per-config Stage-2b figures predate the true
`{region}_velocity` metric and the `delta_cosine[0]` fix (see the top of this doc). Needs the
29 GB trajectory dump, which only exists on the sweep machine. Command:
`python scripts/infinitetalk/analyze_ode_trajectory_infinitetalk.py ...` (full invocation in the
"Quickstart — InfiniteTalk" section of `README.md`), across all 7 configs, then commit the
refreshed `geometry_*.json` files. **Unblocks:** trusting the InfiniteTalk velocity/Δ-cosine
panels; independent of everything else here.

**(c) Gap A — InfiniteTalk's per-step euler perceptual CSVs, from the sweep machine.**
`configs/registry.yaml` marks the 4 per-step euler-metrics entries `status: missing_sweep_machine`:
`it_euler_on_on`, `it_euler_nocfg_on`, `it_euler_nocfg_nocfg`, `it_euler_on_nocfg` (their `csv:`
paths — `results/infinitetalk/data/euler_{on_on,nocfg_on,nocfg_nocfg,on_nocfg}_metrics.csv` — do
not exist yet in this repo). Only 7 files (~500 KB total) need to move off the sweep machine:
```
<repo>/ode_analysis_euler_jump/euler_{on_on,on_noaudio,on_nocfg,noaudio_on,noaudio_noaudio,nocfg_on,nocfg_nocfg}/perceptual_v2/metrics.csv
```
Full detail (why it matters — the terminal-step-only data already committed misses the step
11–15 Sync-C peak entirely) in `docs/cross-model-comparison.md` gap A. **Unblocks:** per-step
Factorial-B diff against OmniAvatar's `14B_textaudio_euler_*` CSVs; independent of (a) and (b).

## Known issues / watch-list
- **Timing**: ~26 min/trajectory; the (1,1) config is 3× faster (1-pass). Plan for ~8 h/sweep.
- **Env-split for 2a**: `latentsync-metrics` env lacks `lpips`; use `omniavatar` for Phase-2 metrics.
- **GT temporal alignment**: GT mp4 is 25 fps and aligned at frame 0 with the generated clip (recon
  setup). Fine for these samples; re-check if new data is added.
- **Per-sample mouth masks**: faces aren't canonically aligned, so mouth-region *pixelwise* variance
  isn't cross-sample comparable (2b reports it on the full grid only).
- **`upper_face` region dropped everywhere** — it was the LatentSync mask complement, no InfiniteTalk
  analogue. Regions are `mouth` (dlib bbox) + `full`.
- **Prompt**: generic `"A person is talking."` for all samples (analog of OmniAvatar's common prompt).
  This affects the text-CFG direction; keep consistent if regenerating.

## Design decisions log (why things are the way they are)
- Path A (fork the loop) not a full FastGenNetwork wrapper: we only need trajectories, not training.
- Force-square: user requirement + uniform-dims analysis (see fact #3 in README).
- x0 derived, not from a head: InfiniteTalk has no x0 head (fact #1).
- CFG combined in x0-space ≡ velocity-space (affine), so trajectories are comparable to OmniAvatar's.
- Mouth region via dlib bbox (points 48-67), not a mask: InfiniteTalk has no LatentSync mouth mask.
