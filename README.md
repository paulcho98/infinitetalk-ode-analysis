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
| **Stage 1 — trajectory generation** (`scripts/generate_infinitetalk_ode_pairs_full.py`) | ✅ **written + validated** (decodes to coherent faces; uniform `[16,21,80,80]` latents) |
| Stage 1 launcher (`scripts/run_infinitetalk_ode_sweep.sh`) | ✅ works; **full sweep NOT yet completed** (killed to free GPUs; ~8 h run) |
| **Stage 2a — perceptual metric engine** (`scripts/eval_ode_perceptual_v2_infinitetalk.py`) | 🟡 drafted; imports OK; **Phase-1 decode not yet run on real data** |
| **Stage 2b — latent trajectory analysis** (`scripts/analyze_ode_trajectory_infinitetalk.py`) | 🟡 drafted; mask-derivation + VAE paths validated; **full run pending square data** |
| **Stage 2c — plotters** (`scripts/plotters_to_adapt/`) | 🔴 **not started** — need config-list repoint (see audit) |
| New **2-D CFG heatmap** view | 🔴 not started (build fresh) |

**Nothing has produced final metrics/figures yet.** The generation driver is validated on single
samples; the analysis scripts are drafted and partially validated; the full sweep + full analysis
have not been run to completion.

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
  generate_infinitetalk_ode_pairs_full.py   # Stage 1: the ODE-trajectory driver (VALIDATED)
  run_infinitetalk_ode_sweep.sh              # Stage 1: 4-GPU launcher for the 7-config sweep
  eval_ode_perceptual_v2_infinitetalk.py     # Stage 2a: decode + perceptual/lip/sync metrics (DRAFT)
  analyze_ode_trajectory_infinitetalk.py     # Stage 2b: latent straightness/velocity/x0-vs-GT (DRAFT)
  plotters_to_adapt/                         # Stage 2c: OmniAvatar plotters to repoint (TODO)
reference/                                   # COMPLETE original OmniAvatar code (for cross-checking the port)
  README.md                                  #   what to diff/verify against
  omniavatar_analysis/                       #   original analysis scripts (our Stage-2 derives from these)
  fastgen_generation/                        #   original ODE driver + OmniAvatarWan model + FastGen core
data/recon_sample_names.txt                  # the 10 Hallo3 recon sample names
examples/                                    # our smoke-generation video + decoded frames + 2 input clips
docs/
  status-and-todo.md                         # ← START HERE to continue the work
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
the **Hallo3 recon clips**. All paths in the scripts are currently absolute to the original machine and
must be re-pointed.

**Stage 1 — generate trajectories (~8 h, 4 GPUs):**
```bash
bash scripts/run_infinitetalk_ode_sweep.sh 50   # 50 = num steps
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

**Stage 2c — plots:** not written yet. See `docs/stage2-audit.md` — 6 plotters are config-list repoints
(PARAM), and a new 2-D `t×a` heatmap/Pareto view should be built fresh.

---

## Example media (`examples/`)

- `videos/infinitetalk_smoke_generation.mp4` — a full 20-step InfiniteTalk generation (proof the stack
  works end-to-end).
- `frames/ode_x0_decoded_frame*.png` — a decoded `x0` from a real ODE trajectory (coherent face at 6
  steps — validates the x0 derivation).
- `inputs/*.mp4|.wav` — **two Hallo3 recon clips** (reference video + audio) for local testing.
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
| **Hallo3 recon clips** | 10 samples in `data/recon_sample_names.txt`; video `Hallo3_validation/validation_set_for_benchmark/<hash>.mp4`, audio `Hallo3_validation/processed/audios/<hash>.wav` |

See `docs/environment.md` and `docs/data.md` for exact setup and the paths to re-point.
