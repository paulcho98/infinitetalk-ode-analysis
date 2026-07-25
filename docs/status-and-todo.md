# Status & TODO — start here to continue

This is the authoritative "where we are / what remains / how to proceed" doc. Read the root `README.md`
first for the five load-bearing facts, then this.

> **UPDATE — steps 1-5 are DONE.** The 8-GPU sweep, Stage-2a metrics, Stage-2b geometry and the
> Stage-2c plotters all completed; see `results/findings.md` + `results/figures/`. What remains from
> the original plan is **step 6 (the OmniAvatar cross-model comparison)**, plus two NEW items:
>
> - **← CURRENT PRIORITY: Euler-jump factorial** (`docs/euler-jump-experiment.md`) — ported,
>   **never run**. Code is syntax-checked only; smoke-test `euler_on_on` before the full sweep.
> - **Re-run Stage 2b** to pick up the true `{region}_velocity` metric and the `delta_cosine[0]`
>   fix (see the "Corrected in this revision" note in `results/findings.md`). The committed
>   geometry JSONs and per-config figures predate both. **DEPRIORITIZED** behind the Euler-jump
>   work — Stage 2b is correspondingly off by default in `run_stage2_euler_jump.sh` (`RUN_2B=1`
>   enables it). The perceptual CSVs are unaffected by either fix.

## What has been done and VALIDATED

1. **Environment** (`docs/environment.md`): `infinitetalk` conda env built; all InfiniteTalk deps resolve
   (the tricky ones: `transformers==4.49.0`, `diffusers==0.33.1`, flash-attn prebuilt wheel; `dlib-bin`
   added for Stage-2). `generate_infinitetalk.py --help` and a full 20-step generation both run.
2. **Checkpoints** downloaded (Wan2.1-I2V-14B-480P, MeiGen-AI/InfiniteTalk, chinese-wav2vec2-base).
3. **Stage-1 driver** `scripts/generate_infinitetalk_ode_pairs_full.py`:
   - Subclasses `InfiniteTalkPipeline`; forks the denoising loop; derives `x0 = x_t + sigma·noise_pred`.
   - Reproduces InfiniteTalk's exact CFG branching (3-call; (1,1) → 1-pass).
   - **FORCE-SQUARE** patch applied and verified: a non-square (2:3) reference now yields `[16,21,80,80]`.
   - VALIDATED end-to-end: decoded `x0` is a coherent talking-head (see `examples/frames/`).
   - Multi-config per model load; shards by sample; caches wav2vec audio per hash.
4. **Stage-1 launcher** `scripts/run_infinitetalk_ode_sweep.sh`: 7 configs, 4-GPU, `--skip_existing`.
   Confirmed all 4 shards start and produce correct output; the SWEEP WAS NOT RUN TO COMPLETION (killed
   to free GPUs — see below).
5. **Stage-2 drafts** (`eval_ode_perceptual_v2_infinitetalk.py`, `analyze_ode_trajectory_infinitetalk.py`):
   both compile and their CLIs parse. For 2b, mask-derivation (dlib mouth-bbox) + WanVAE decode/encode
   paths ran successfully on real data; the GT reader was fixed to center-crop-to-square (matches the
   generator). 2a's GT reader already center-crops.

## What is DRAFTED but NOT yet run to completion / not validated on final data

- **The full sweep** never finished. It was killed twice: once to fix the force-square bug (correct),
  once to yield GPUs (the reason this repo exists). No trajectories currently persist on disk.
- **Stage 2a Phase-1 decode + Phase-2 metrics** were never run on real trajectories (only imports + `--help`).
- **Stage 2b full run** crashed once on the pre-fix mixed-dims data (the crash that *found* the bug);
  it was not re-run after the force-square fix. It should work now (uniform 80×80), but re-validate.
- **Stage 2c plotters**: untouched. Still the OmniAvatar originals in `scripts/plotters_to_adapt/`.
- **2-D CFG heatmap/Pareto view**: does not exist.

## HOW TO PROCEED (ordered)

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
bash scripts/run_infinitetalk_ode_sweep.sh 50
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
python scripts/analyze_ode_trajectory_infinitetalk.py --traj_dir <cfg> --output_dir <out> \
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
bash scripts/run_infinitetalk_euler_jump.sh 50      # generate, 7 cells across 8 GPUs
bash scripts/run_stage2_euler_jump.sh all           # metrics + geometry (same Stage-2 stack)
python scripts/plot_euler_jump_factorial.py ...     # figures
```
Prereq: Stage-1 trajectories for t5.0_a4.0, t5.0_a1.0, t1.0_a1.0 (all in the standard sweep).
**Smoke-test `euler_on_on` first** — its step 0 should land very close to the sequential trajectory's.

### 6. Compare against OmniAvatar
The point of the study: put InfiniteTalk's ODE-straightness/quality-vs-CFG next to OmniAvatar's
(`/home/work/.local/ode_analysis` on the original machine). Same 10 Hallo3 identities, same step count.

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
