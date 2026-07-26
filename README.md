# InfiniteTalk ODE Trajectory Extraction & Analysis

Self-contained code + documentation for **reproducing the OmniAvatar ODE-trajectory analysis on the
InfiniteTalk baseline**. This repo is a handoff package: it holds all the code written so far, the
reference originals it derives from, comprehensive background, and a precise status + TODO so the work
can be finished on another machine (with Claude Code).

> **One-line goal:** For each denoising step of InfiniteTalk's audio-driven diffusion, save the noisy
> state `x_t` and the model's denoised prediction `x0`, across a 2-D classifier-free-guidance (CFG)
> sweep, then analyze how the ODE trajectory geometry and perceptual quality change with CFG — exactly
> as was done for the OmniAvatar model.

---

## TL;DR status (read `docs/status-and-todo.md` for the full version)

| Piece | State |
|---|---|
| Env + checkpoints | ✅ working (`infinitetalk` conda env; see `docs/environment.md`) |
| **Stage 1 — trajectory generation** (`scripts/generate_infinitetalk_ode_pairs_full.py`) | ✅ **complete** — 7-config sweep, 70 trajectories, ~11 h on 8×A100 |
| Stage 1 launcher (`scripts/run_infinitetalk_ode_sweep_8gpu.sh`) | ✅ complete (job-sharded 8-GPU; `run_infinitetalk_ode_sweep.sh` is the older 4-GPU variant) |
| **Stage 2a — perceptual metric engine** (`scripts/eval_ode_perceptual_v2_infinitetalk.py`) | ✅ **complete** — all 7 configs → `results/data/perceptual_*.csv` |
| **Stage 2b — latent trajectory analysis** (`scripts/analyze_ode_trajectory_infinitetalk.py`) | ✅ run for all 7 → `results/data/geometry_*.json` — ⚠️ **needs a re-run**, see below |
| **Stage 2c — plotters** | ✅ complete — per-step curves, 2-D heatmaps, perception–distortion frontier, per-config, default-vs-baseline |
| New **2-D CFG heatmap** view | ✅ `results/figures/cfg_grid_heatmaps_mouth.png` |
| **Euler-jump factorial** (`scripts/generate_infinitetalk_euler_jump.py`) | ✅ **complete** — 7 cells, ~12.7 h on 7×A100; both pre-flights passed; see `docs/euler-jump-experiment.md` |
| Cross-model comparison vs OmniAvatar | 🔴 not started (original TODO step 6) |

**Findings + figures are in `results/`** — start with `results/findings.md`.

### ⚠️ Outstanding before trusting the Stage-2b geometry
The committed `results/data/geometry_*.json` and `results/figures/per_config/` predate two fixes to
`analyze_ode_trajectory_infinitetalk.py`, so **Stage 2b should be re-run**:

1. a genuine `{region}_velocity` = ‖x0(t)−x0(t−1)‖² was added — the old overlay mislabeled the
   *signed* `delta_mse` as velocity and log-plotted it, silently dropping every negative step;
2. `delta_cosine[0]` was seeded with an *absolute* cosine instead of 0, which dominated the Δ-cosine
   bar panels and always won the "top-5 Δ-cosine steps" ranking.

The perceptual CSVs (`results/data/perceptual_*.csv`) are **unaffected** by both.

---

## Background — what this is and why

### The OmniAvatar ODE analysis (the thing we're reproducing)
For the OmniAvatar audio-avatar model we ran a two-stage study:
1. **Generation:** run the 50-step flow-matching (rectified-flow) sampler and, at every step, dump the
   latent `x_t` and the denoised prediction `x0`. Repeat across a sweep of CFG scales.
2. **Analysis:** decode each `x0` and measure perceptual/lip-sync quality *vs step*; measure the ODE
   trajectory geometry (how "straight" the path is, velocity, x0-vs-GT) *vs step*; compare across CFG.
   The research question is how CFG bends the ODE trajectory and trades off quality.

The OmniAvatar results live at `/home/work/.local/ode_full_trajectories` (generation) and
`/home/work/.local/ode_analysis` (analysis) on the original machine. The generation code was in a
separate repo, **FastGen** (`fastgen.networks.OmniAvatar.network.OmniAvatarWan` +
`scripts/generate_omniavatar_ode_pairs_full.py`); the analysis code was in the **OmniAvatar** repo's
`scripts/` (`eval_ode_perceptual_v2.py`, `analyze_ode_trajectory.py`, `plot_*.py`). Both originals are
in `reference/` and `scripts/plotters_to_adapt/` here.

### Porting to InfiniteTalk (this repo)
InfiniteTalk (https://github.com/MeiGen-AI/InfiniteTalk) is a different audio-avatar model. We want the
**same** analysis on it. Only the *generation* stage is model-specific; the *analysis* stage mostly
ports (with dim/mask edits). See `docs/architecture-analysis.md` (renamed
`infinitetalk-architecture-analysis.md`) for the deep model comparison.

---

## The five facts that make this work (do not lose these)

1. **InfiniteTalk has NO x0 head.** Its sampler is a hand-written flow-matching Euler on *velocity*:
   the running `latent` IS `x_t`, and it stores `noise_pred = -v` (velocity, sign-flipped). To get the
   denoised prediction we DERIVE it:
   ```
   sigma  = timesteps[i] / 1000          # timesteps are on a 0..1000 scale
   x0_pred = x_t + sigma * noise_pred    # (noise_pred is already the negated velocity)
   ```
   This is implemented in `scripts/generate_infinitetalk_ode_pairs_full.py`. It matches OmniAvatar's
   x0-space convention (CFG-in-velocity ≡ CFG-in-x0 because the map is affine; and InfiniteTalk's Euler
   update ≡ OmniAvatar's rectified-flow re-noise step), so the two models' trajectories are comparable.

2. **3-call CFG** (default text_scale=5, audio_scale=4):
   `noise_pred = uncond + text_s·(cond − drop_text) + audio_s·(drop_text − uncond)`.
   Special case: `text_scale == 1` uses a 2-call drop_audio formula; and `(1,1)` collapses to just the
   conditional pass (1 forward pass — the no-CFG baseline).

3. **FORCE SQUARE.** InfiniteTalk's `generate_infinitetalk` picks the aspect-ratio bucket closest to
   each reference image, so non-square references (Hallo3 has 2:3 portraits) give NON-square latents
   `[16,21,64,96]`. We require square, so the driver hardcodes `target_h=target_w=640` (→ `[16,21,80,80]`
   for every sample). **This bug bit us once and cost a full sweep — keep it fixed.** Stage-2 GT frame
   readers must use the same center-crop-to-square (not a stretch resize).

4. **Latent = `[16, 21, 80, 80]`** (16 ch, 21 latent frames, 80×80 latent = 640×640 pixels). Decode
   ONLY with InfiniteTalk's `WanVAE` (the trajectories live in its VAE space).

5. **This model is SLOW.** ~10 s per DiT forward pass at 480p → ~26 min per 3-call trajectory →
   the full 7-config × 10-sample × 50-step sweep is **~8 hours on 4 GPUs**. (No teacache — it's off for
   clean trajectories.) Budget accordingly.

---

## The CFG sweep (7 configs)

Naming: `infinitetalk_t{T}_a{A}`. Two families sharing the audio axis {1,2,4,6}:

| audio | Family 1 (text fixed=5) | Family 2 (text scaled) |
|---|---|---|
| 1 | (5,1) | **(1,1)** = no-CFG baseline (1-pass) |
| 2 | (5,2) | (2.5,2) |
| 4 | **(5,4)** = default (shared) | (5,4) |
| 6 | (5,6) | (7.5,6) |

7 distinct: `5:4, 5:1, 5:2, 5:6, 1:1, 2.5:2, 7.5:6`.

---

## Repo map

```
scripts/
  generate_infinitetalk_ode_pairs_full.py   # Stage 1: the ODE-trajectory driver (COMPLETE)
  run_infinitetalk_ode_sweep_8gpu.sh         # Stage 1: 8-GPU job-sharded launcher (the one used)
  run_infinitetalk_ode_sweep.sh              # Stage 1: older 4-GPU sample-sharded launcher
  eval_ode_perceptual_v2_infinitetalk.py     # Stage 2a: decode + perceptual/lip/sync metrics
  analyze_ode_trajectory_infinitetalk.py     # Stage 2b: latent geometry / velocity / x0-vs-GT
  run_stage2_infinitetalk.sh                 # Stage 2 orchestration (2a + 2b per config)
  run_stage2a_metrics_sharded.sh             # Stage 2a metrics, 35-way sharded across 8 GPUs
  run_stage2b.sh                             # Stage 2b across all 7 configs
  plot_ode_curves_infinitetalk.py            # Stage 2c: per-step curves, 7 configs overlaid
  plot_cfg_grid_infinitetalk.py              # Stage 2c: 2-D t×a heatmaps + perception–distortion frontier
  plot_trajectory_geometry_overlay.py        # Stage 2c: geometry overlay (regenerates results/figures/trajectory/)
  plot_default_vs_baseline.py                # Stage 2c: default-vs-ablation, 2 lines/panel, mouth region
  generate_infinitetalk_euler_jump.py        # Euler-jump straightness probe (PORTED, NEVER RUN)
  run_infinitetalk_euler_jump.sh             #   ↳ 7 factorial cells across 8 GPUs
  run_stage2_euler_jump.sh                   #   ↳ straightness + Stage 2a (2b opt-in via RUN_2B=1)
  measure_euler_straightness.py              #   ↳ ‖x0_euler − x0_seq‖ per step — THE curvature number
  plot_euler_jump_factorial.py               #   ↳ per-factorial figures + terminal CSV
  plotters_to_adapt/                         # original OmniAvatar plotters (reference only)
results/                                     # ← THE OUTPUT: findings, figures, raw metrics
  findings.md                                #   written analysis — START HERE for results
  data/perceptual_t{T}_a{A}.csv              #   per-step metrics, 10 samples × 50 steps × 7 configs
  data/geometry_t{T}_a{A}.json               #   Stage-2b latent geometry (see re-run caveat above)
  figures/                                   #   curves, heatmaps, frontier, per_config/, compare_default/
reference/                                   # COMPLETE original OmniAvatar code (for cross-checking the port)
  README.md                                  #   what to diff/verify against
  omniavatar_analysis/                       #   original analysis scripts (our Stage-2 derives from these)
  fastgen_generation/                        #   original ODE driver + OmniAvatarWan model + FastGen core
data/recon_sample_names.txt                  # the 10 Hallo3 recon sample names
data/recon_clips/<hash>.{mp4,wav}            # ALL 10 reference videos + audios (BUNDLED, 7.6MB) — inputs to run
examples/                                    # our smoke-generation video + decoded frames + 2 input clips
docs/
  status-and-todo.md                         # ← START HERE to continue the work
  euler-jump-experiment.md                   # the ported straightness factorial: design + how to run
  environment.md                             # conda env, dep pins, gotchas, checkpoint download
  data.md                                    # Hallo3 sample sources, paths, force-square
  stage2-audit.md                            # per-script portability classification for 2c
  architecture-analysis.md                   # deep InfiniteTalk vs OmniAvatar model comparison
  infinitetalk-ode-port-plan.md              # the working plan doc
```

---

## How to run

Prereqs (see `docs/environment.md` for details): the **InfiniteTalk repo** on `PYTHONPATH`, its
**weights** downloaded, the **`infinitetalk` conda env**, the **metrics models** (`eval_metrics/`), and
the **Hallo3 recon clips**. Repo-internal paths are portable (the launchers derive `$REPO` from their
own location); the launchers' **external** paths — `INFINITETALK_ROOT`, `METRICS_ROOT`, weights, and
the venv `PY` — are absolute and are currently pointed at the **sweep machine**. Re-point them if you
run anywhere else; all are overridable by env var.

**Stage 1 — generate trajectories (~11 h, 8 GPUs):**
```bash
bash scripts/run_infinitetalk_ode_sweep_8gpu.sh 50   # 50 = num steps
# → <OUT>/infinitetalk_t{T}_a{A}/<sample>/{step_NNN_xt.pt, step_NNN_x0.pt, ode_schedule.json, input_latents.pt}
```

**Stage 2a — perceptual metrics** (two-env: Phase-1 decode in `infinitetalk`, Phase-2 metrics in
`omniavatar`; `latentsync-metrics` lacks `lpips`):
```bash
# Phase 1 (infinitetalk env — WanVAE decode):
python scripts/eval_ode_perceptual_v2_infinitetalk.py --phase decode  --traj_dir <cfg_dir> --output_dir <out>
# Phase 2 (omniavatar env — dlib/lpips/syncnet):
python scripts/eval_ode_perceptual_v2_infinitetalk.py --phase metrics --traj_dir <cfg_dir> --output_dir <out>
python scripts/eval_ode_perceptual_v2_infinitetalk.py --merge          --traj_dir <cfg_dir> --output_dir <out>
```

**Stage 2b — latent trajectory analysis** (single env `infinitetalk`, now has WanVAE + dlib):
```bash
python scripts/analyze_ode_trajectory_infinitetalk.py \
  --traj_dir <cfg_dir> --output_dir <out> --gt_mode encode \
  --gt_video_dir <hallo3_benchmark_videos> --mask_source ref_decode \
  --mouth_mask_cache <cache> --shape_predictor <eval_metrics>/shape_predictor_68_face_landmarks.dat
```

**Stage 2c — plots** (all read either a live analysis root or the committed `results/data/`):
```bash
python scripts/plot_ode_curves_infinitetalk.py       --analysis_root <root> --output_dir results/figures
python scripts/plot_cfg_grid_infinitetalk.py         --analysis_root results/data --output_dir results/figures
python scripts/plot_trajectory_geometry_overlay.py   --geometry_dir  results/data --output_dir results/figures/trajectory
python scripts/plot_default_vs_baseline.py           # default vs (t5,a1) and vs (1,1), mouth region
```

**Euler-jump factorial — PORTED BUT NEVER RUN.** Full design in `docs/euler-jump-experiment.md`.
Measures ODE straightness by jumping from step 0 to each noise level and re-predicting. Two
overlapping 2×2s over (step-0 CFG) × (teacher CFG) with `on=(5,4)`, `noaudio=(5,1)`, `nocfg=(1,1)`
— 7 distinct cells. Requires the Stage-1 trajectories for those three configs (all in the sweep).
```bash
bash scripts/run_infinitetalk_euler_jump.sh 50   # generate, 7 cells across 8 GPUs
bash scripts/run_stage2_euler_jump.sh all        # straightness + Stage 2a (2b off; RUN_2B=1 to enable)
python scripts/plot_euler_jump_factorial.py --euler_analysis_root ode_analysis_euler_jump \
    --sequential_analysis_root ode_analysis_infinitetalk --output_dir results/figures/euler_jump
```
The headline curvature number is `‖x0_euler − x0_sequential‖` per step, from
`scripts/measure_euler_straightness.py` (run automatically per cell). It reads only the saved `x0`
tensors — no VAE/GT/model — so it takes seconds and does **not** depend on the Stage-2b re-run.
**Smoke-test one cell first:** `bash scripts/run_stage2_euler_jump.sh euler_on_on 0`. Its step 0
should land very close to the sequential trajectory's — divergence means the conditioning or
schedule doesn't match. This also exercises the `prepare_conditioning()` / `predict_noise()`
refactor of the Stage-1 driver, which is shared by both paths and has not been run against the model.

---

## Example media (`examples/`)

- `videos/infinitetalk_smoke_generation.mp4` — a full 20-step InfiniteTalk generation (proof the stack
  works end-to-end).
- `frames/ode_x0_decoded_frame*.png` — a decoded `x0` from a real ODE trajectory (coherent face at 6
  steps — validates the x0 derivation).
- `inputs/*.mp4|.wav` — **two Hallo3 recon clips** for quick tests. **All 10** are bundled in `data/recon_clips/`.
  *Provenance:* these are from the Hallo3 validation set; keep private / obtain the full set from the
  dataset if redistributing.

---

## External dependencies this repo does NOT bundle (obtain separately)

| Dependency | What / where |
|---|---|
| **InfiniteTalk repo** | `github.com/MeiGen-AI/InfiniteTalk` — the driver subclasses `wan.multitalk.InfiniteTalkPipeline`; put on `PYTHONPATH` (`sys.path.insert(0, INFINITETALK_ROOT)`) |
| **Wan2.1-I2V-14B-480P** | `huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P` (base model) |
| **MeiGen-AI/InfiniteTalk** | `huggingface.co/MeiGen-AI/InfiniteTalk` (audio-condition weights: `single/infinitetalk.safetensors`) |
| **chinese-wav2vec2-base** | `huggingface.co/TencentGameMate/chinese-wav2vec2-base` (audio encoder) |
| **Metrics models** | `eval_metrics/` on the original machine: `shape_predictor_68_face_landmarks.dat`, `checkpoints/auxiliary/syncnet_v2.model`, plus `lpips` pip pkg. SyncNet eval code from the `eval`/`syncnet_python` packages. |
| **Hallo3 recon clips** | BUNDLED in this repo at `data/recon_clips/<hash>.{mp4,wav}` (all 10). Point `--video_dir` and `--audio_dir` there |

See `docs/environment.md` and `docs/data.md` for exact setup and the paths to re-point.
