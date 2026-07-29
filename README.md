# talking-head-ode-analysis

> Current checkout: `/home/work/.local/infinitetalk-ode-analysis`. This repo is being renamed
> `talking-head-ode-analysis` on GitHub (the rename itself is a separate, later task) — the name
> above is the target name, used here in anticipation of it.

Self-contained code + documentation for an **ODE-trajectory analysis of two audio-driven
portrait-animation diffusion models built on Wan 2.1**: **OmniAvatar** (Wan2.1 T2V-14B base +
LoRA + additive-residual audio conditioning, V2V lip-sync) and **InfiniteTalk** (Wan2.1 I2V-14B
base + audio cross-attention). Both are analyzed with the same two-stage pipeline:

1. **Generation (Stage 1):** run the model's flow-matching sampler across a 1-D or 2-D
   classifier-free-guidance (CFG) sweep and, at every denoising step, save the noisy latent `x_t`
   and the model's denoised prediction `x0`.
2. **Analysis (Stage 2):** decode each `x0` and measure perceptual/lip-sync quality *vs step*
   (Stage 2a); measure the ODE trajectory's geometry — straightness, velocity, distance-to-GT —
   *vs step* (Stage 2b); plot both against CFG (Stage 2c).

A `scripts/comparison/` stage joins the two models' per-step CSVs for a direct, CFG-normalized
comparison — **not started yet** (directory is empty); see `docs/cross-model-comparison.md`.

**Full current status, per model:** `docs/status-and-todo.md` (InfiniteTalk-focused — that
study's generation/analysis/euler-jump work is complete, one re-run pending) and
`docs/omniavatar-experiments.md` (OmniAvatar's experiment inventory — a completed historical
study, ported into this repo as committed CSVs + regeneration scripts).

| | OmniAvatar | InfiniteTalk |
|---|---|---|
| Stage 1 (trajectories) | done — 27 CSVs' worth of prior runs; latents were not retained for most of them (see Deferred re-runs below) | done — 7-config sweep, 70 trajectories |
| Stage 2a (perceptual/lip metrics) | done, all groups | done, all 7 configs |
| Stage 2b (latent geometry) | not computed for any OmniAvatar run in this study (no surviving latents) | done, all 7 configs — **stale**, needs a re-run (see `docs/status-and-todo.md`) |
| Euler-jump (ODE straightness) | done, both CFG-drop modes (8 cells) | done, all 7 cells |
| Cross-model comparison | not started (`scripts/comparison/` is empty) | — |

---

## Repo map

```
configs/
  machine-omniavatar.env.example   # path roots for this box
  machine-sweep.env.example        # path roots for the InfiniteTalk sweep machine
  machine.env                      # your copy (gitignored) — cp one of the two examples here
  registry.yaml                    # every experiment: model, guidance knobs, CSV path, registry id

models/                            # vendored — Stage 1/2 run with no external repo checkout
  omniavatar_wan/                  # OmniAvatarWan (DiT + AudioPack + FastGenNetwork base), from FastGen
  wan_vae/                         # Wan VAE decoder + a direct loader (replaces OmniAvatar's ModelManager)

scripts/
  common/
    measure_euler_straightness.py  # ‖x0_euler − x0_sequential‖ per step — model-agnostic, both sides use it
  infinitetalk/                    # Stage 1 driver + 8-GPU launcher, Stage 2a/2b engines, Stage 2c plotters,
                                    #   euler-jump generator + launcher + plotter (17 files)
  omniavatar/                      # ported from the OmniAvatar + FastGen repos (26 files):
                                    #   Stage 1 driver + 3 launchers (cfg4.5 / nocfg / no-audio)
                                    #   euler-jump/fresh-noise generator + launcher (+ --save_latents)
                                    #   Stage 2a/2b engines, decode/visualize/spatial-probe scripts
                                    #   4 metrics/mouthweight launchers, 10 comparison plotters
  comparison/                      # cross-model comparison stage — EMPTY, not started

data/
  recon_sample_names.txt           # the 10 Hallo3 recon sample IDs shared by both models
  recon_clips/<hash>.{mp4,wav}     # all 10 reference videos + audio, bundled (~7.6 MB)
  mask.png                         # LatentSync-style upper-face/mouth mask (OmniAvatar region split)

results/
  omniavatar/data/                 # 27 CSVs + README.md, committed from the original OmniAvatar study
  omniavatar/figures/               # EMPTY — no figures regenerated in this repo yet
  infinitetalk/data/, infinitetalk/figures/   # 7-config sweep CSVs, geometry + straightness JSONs, all figures
  infinitetalk/findings.md         # written analysis — start here for InfiniteTalk results
  comparison/                      # EMPTY — cross-model comparison outputs land here

reference/                         # frozen originals — diff any scripts/omniavatar/* port against these
  README.md
  omniavatar_analysis/             # original OmniAvatar repo scripts/ (8 files)
  fastgen_generation/               # original FastGen driver + OmniAvatarWan_model/ + fastgen_core/

examples/                          # smoke-test media: InfiniteTalk generation, decoded x0 frames, 2 sample clips

docs/
  status-and-todo.md               # START HERE — current state + ordered next steps (InfiniteTalk-focused)
  omniavatar-experiments.md        # OmniAvatar experiment inventory: 6 groups, registry ids, regen commands
  cross-model-comparison.md        # step-6 handoff: machine map, OmniAvatar inventory, gaps, normalization rules
  euler-jump-experiment.md         # the ODE-straightness factorial: design + InfiniteTalk results
  environment.md                   # `infinitetalk` conda env: build steps, pins, gotchas
  data.md                          # the 10 recon samples, force-square, prompt, output layout
  stage2-audit.md                  # per-script portability classification (OmniAvatar → InfiniteTalk Stage 2)
  architecture-analysis.md         # deep InfiniteTalk vs OmniAvatar model comparison
  infinitetalk-ode-port-plan.md    # historical planning doc, predates the two-model repo split
```

---

## Environments

Three conda envs, no single `requirements.txt` (the two models' dependency stacks conflict).

| Env | Used for | Pins / quirks |
|---|---|---|
| **`omniavatar`** | All OmniAvatar Stage-2 metrics (dlib mouth landmarks + lpips + SyncNet) and VAE decode; **also** the euler-jump/fresh-noise generator — `run_single_step_both.sh` runs every phase (generation *and* metrics) under this one env, matching the original single-interpreter launcher | Shares the sibling OmniAvatar repo's conda env. Has the full metrics stack that `latentsync-metrics` lacks (no `lpips` there). |
| **`fastgen`** | OmniAvatar Stage-1 trajectory generation only — `generate_ode_trajectories.sh`, `generate_ode_nocfg.sh`, `generate_ode_no_audio.sh`, and batch 1 (trajectory gen) of `run_mouthweight_generation.sh` | Needs the FastGen-era torch stack `models/omniavatar_wan` was vendored from. **Quirk:** clear `LD_LIBRARY_PATH` before invoking — the system's CUDA 12.8 install shadows the env's own CUDA 12.9 libs: `env LD_LIBRARY_PATH= "$PY_FASTGEN" ...` |
| **`infinitetalk`** | InfiniteTalk Stage-1 generation + Stage-2 decode | `torch==2.4.1`, `torchvision==0.19.1`, `torchaudio==2.4.1`, `xformers==0.0.28`, `flash_attn==2.7.4.post1`, `transformers==4.49.0`, `diffusers==0.33.1`, `dlib-bin`. **`TORCHDYNAMO_DISABLE=1`** (eager execution only — this box lacks the python3.10 dev headers Triton/inductor needs to compile). Full build steps + every gotcha: `docs/environment.md`. |

---

## Machine setup

```bash
cp configs/machine-omniavatar.env.example configs/machine.env   # this box (OmniAvatar + InfiniteTalk both present)
# cp configs/machine-sweep.env.example configs/machine.env      # the InfiniteTalk-only sweep machine
```

Edit `configs/machine.env` (gitignored) for your paths. On this box it's already correct out of
the box: `WEIGHTS_ROOT=/home/work/.local/hyunbin/LipForcing-release/weights` (the Wan2.1-T2V-14B
base weights — **not** `OmniAvatar/pretrained_models`, which only holds the OmniAvatar-specific
checkpoints). Every launcher `source`s `configs/machine.env` if present; every path constant also
has a hardcoded fallback default and can be overridden by env var or CLI flag, so nothing breaks
if you skip this step and just export the two or three vars you need.

---

## Quickstart — OmniAvatar

Full experiment-by-experiment inventory + exact regeneration commands: **`docs/omniavatar-experiments.md`**.

**Generate (Stage 1, `fastgen` env):**
```bash
bash scripts/omniavatar/generate_ode_trajectories.sh          # 50-step trajectory, CFG=4.5 (the default)
bash scripts/omniavatar/generate_ode_nocfg.sh                  # 50-step, no CFG (guidance_scale=1.0, ~2x faster)
bash scripts/omniavatar/generate_ode_no_audio.sh               # audio-ablation (zeroed audio embeddings)
```
Other CFG values, the audio-only CFG-drop mode, or scheduled CFG have no dedicated launcher — call
the driver directly (see `docs/omniavatar-experiments.md` for the exact flags per experiment):
```bash
python scripts/omniavatar/generate_omniavatar_ode_pairs_full.py \
  --guidance_scale <X> --cfg_drop_text {true,false} [--cfg_crossover <tau>] ...
```

**Euler-jump / fresh-noise (`omniavatar` env, single interpreter for gen + metrics):**
```bash
bash scripts/omniavatar/run_single_step_both.sh    # 4 GPUs parallel: fresh_noise, euler on/on, euler nocfg/on, euler nocfg/nocfg
```

**Analyze (Stage 2, `omniavatar` env):**
```bash
bash scripts/omniavatar/run_eval_ode_perceptual_v2.sh    # Stage 2a: decode + perceptual/lip metrics, 4 GPUs
bash scripts/omniavatar/run_all_metrics_sequential.sh    # same engine, 1 GPU sequential (audio-only-CFG + LatentSync groups)
python scripts/omniavatar/analyze_ode_trajectory.py \    # Stage 2b: latent geometry (straightness, velocity, x0-vs-GT)
  --traj_dir <cfg_dir> --mask_path data/mask.png --output_dir <out>
```

MouthWeight checkpoint (separate teacher, `$MOUTHWEIGHT_CKPT`):
```bash
bash scripts/omniavatar/run_mouthweight_generation.sh
bash scripts/omniavatar/run_mouthweight_evaluation.sh
```

**Plot (`omniavatar` env, 10 plotters — full list in `docs/omniavatar-experiments.md`):**
```bash
python scripts/omniavatar/plot_combined_ode_comparison.py --output_dir <out>   # main 14B text+audio overlay
python scripts/omniavatar/plot_cfg_mode_compare.py --output_dir <out>          # text+audio vs audio-only vs noCFG
python scripts/omniavatar/plot_all_models_compare.py --output_dir <out>        # OmniAvatar vs LatentSync
```
Every plotter takes `--analysis_root` (default `$ODE_ANALYSIS_ROOT_OMNI`) and expects the nested
`<config>/<variant>/metrics.csv` shape that `run_eval_ode_perceptual_v2.sh` /
`run_all_metrics_sequential.sh` produce — i.e. they regenerate figures from a **live** analysis
run, not directly from the flat, already-merged CSVs committed at `results/omniavatar/data/`
(those are a separately-exported historical bundle; see `docs/omniavatar-experiments.md`).

---

## Quickstart — InfiniteTalk

**Generate (Stage 1, `infinitetalk` env, ~11 h on 8 GPUs):**
```bash
bash scripts/infinitetalk/run_infinitetalk_ode_sweep_8gpu.sh 50   # 50 = num steps; 7-config sweep, 70 trajectories
```

**Analyze (Stage 2a — two-env: decode in `infinitetalk`, metrics in `omniavatar`; `latentsync-metrics` lacks `lpips`):**
```bash
python scripts/infinitetalk/eval_ode_perceptual_v2_infinitetalk.py --phase decode  --traj_dir <cfg_dir> --output_dir <out>
python scripts/infinitetalk/eval_ode_perceptual_v2_infinitetalk.py --phase metrics --traj_dir <cfg_dir> --output_dir <out>
python scripts/infinitetalk/eval_ode_perceptual_v2_infinitetalk.py --merge          --traj_dir <cfg_dir> --output_dir <out>
```

**Analyze (Stage 2b, `infinitetalk` env — needs a re-run, see `docs/status-and-todo.md`):**
```bash
python scripts/infinitetalk/analyze_ode_trajectory_infinitetalk.py \
  --traj_dir <cfg_dir> --output_dir <out> --gt_mode encode \
  --gt_video_dir <hallo3_benchmark_videos> --mask_source ref_decode \
  --mouth_mask_cache <cache> --shape_predictor <eval_metrics>/shape_predictor_68_face_landmarks.dat
```

**Plot (Stage 2c):**
```bash
python scripts/infinitetalk/plot_ode_curves_infinitetalk.py     --analysis_root <root> --output_dir results/infinitetalk/figures
python scripts/infinitetalk/plot_cfg_grid_infinitetalk.py       --analysis_root results/infinitetalk/data --output_dir results/infinitetalk/figures
python scripts/infinitetalk/plot_trajectory_geometry_overlay.py --geometry_dir  results/infinitetalk/data --output_dir results/infinitetalk/figures/trajectory
python scripts/infinitetalk/plot_default_vs_baseline.py         # default vs (t5,a1) and vs (1,1), mouth region
```

**Euler-jump factorial — complete** (all 7 cells, ~12.7 h on 7×A100). Full design in
`docs/euler-jump-experiment.md`; results in `results/infinitetalk/findings.md` § "ODE straightness".
```bash
bash scripts/infinitetalk/run_infinitetalk_euler_jump.sh 50        # generate, 7 cells across 8 GPUs
bash scripts/infinitetalk/run_stage2_euler_jump_sharded.sh all     # straightness + Stage 2a; shards the SyncNet leg
python scripts/infinitetalk/plot_euler_jump_factorial.py --euler_analysis_root ode_analysis_euler_jump \
    --sequential_analysis_root ode_analysis_infinitetalk --output_dir results/infinitetalk/figures/euler_jump
```
The headline curvature number, `‖x0_euler − x0_sequential‖` per step, comes from
`scripts/common/measure_euler_straightness.py` — model-agnostic, reads only saved `x0` tensors
(no VAE/GT/model), runs in seconds.

### The CFG sweep (7 configs)

Naming: `infinitetalk_t{T}_a{A}`. Two families sharing the audio axis `{1,2,4,6}`:

| audio | Family 1 (text fixed=5) | Family 2 (text scaled) |
|---|---|---|
| 1 | (5,1) | **(1,1)** = no-CFG baseline (1-pass) |
| 2 | (5,2) | (2.5,2) |
| 4 | **(5,4)** = default (shared) | (5,4) |
| 6 | (5,6) | (7.5,6) |

7 distinct: `5:4, 5:1, 5:2, 5:6, 1:1, 2.5:2, 7.5:6`.

### The five facts that make the InfiniteTalk port work (do not lose these)

1. **InfiniteTalk has NO x0 head.** Its sampler is a hand-written flow-matching Euler on
   *velocity*: the running `latent` IS `x_t`, and it stores `noise_pred = -v` (velocity,
   sign-flipped). To get the denoised prediction we DERIVE it:
   ```
   sigma  = timesteps[i] / 1000          # timesteps are on a 0..1000 scale
   x0_pred = x_t + sigma * noise_pred    # (noise_pred is already the negated velocity)
   ```
   This is implemented in `scripts/infinitetalk/generate_infinitetalk_ode_pairs_full.py`. It
   matches OmniAvatar's x0-space convention (CFG-in-velocity ≡ CFG-in-x0 because the map is
   affine; InfiniteTalk's Euler update ≡ OmniAvatar's rectified-flow re-noise step), so the two
   models' trajectories are comparable.

2. **3-call CFG** (default text_scale=5, audio_scale=4):
   `noise_pred = uncond + text_s·(cond − drop_text) + audio_s·(drop_text − uncond)`.
   Special case: `text_scale == 1` uses a 2-call drop_audio formula; and `(1,1)` collapses to just
   the conditional pass (1 forward pass — the no-CFG baseline).

3. **FORCE SQUARE.** InfiniteTalk's `generate_infinitetalk` picks the aspect-ratio bucket closest
   to each reference image, so non-square references (Hallo3 has 2:3 portraits) give NON-square
   latents `[16,21,64,96]`. We require square, so the driver hardcodes `target_h=target_w=640`
   (→ `[16,21,80,80]` for every sample). **This bug bit us once and cost a full sweep — keep it
   fixed.** Stage-2 GT frame readers must use the same center-crop-to-square (not a stretch
   resize).

4. **Latent = `[16, 21, 80, 80]`** (16 ch, 21 latent frames, 80×80 latent = 640×640 pixels).
   Decode ONLY with InfiniteTalk's `WanVAE` (the trajectories live in its VAE space).

5. **This model is SLOW.** ~10 s per DiT forward pass at 480p → ~26 min per 3-call trajectory →
   the full 7-config × 10-sample × 50-step sweep is **~8 hours on 4 GPUs** (~11 h at the
   8-GPU job-sharded launcher's actual utilization). No teacache — it's off for clean
   trajectories. Budget accordingly.

---

## External dependencies (obtain separately)

| Dependency | What / where |
|---|---|
| **Wan2.1-T2V-14B base weights** | `$WEIGHTS_ROOT/Wan2.1-T2V-14B/`: 6 DiT shards (`diffusion_pytorch_model-0000{1..6}-of-00006.safetensors`), `Wan2.1_VAE.pth`, T5 encoder. On this box: `/home/work/.local/hyunbin/LipForcing-release/weights`. |
| **OmniAvatar teacher checkpoint** | `$TEACHER_CKPT` — 14B V2V (LoRA + audio modules), `step-10500.pt`. |
| **OmniAvatar MouthWeight checkpoint** | `$MOUTHWEIGHT_CKPT` — `step-6000.pt`. |
| **Wan2.1-I2V-14B-480P** | `huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P` — InfiniteTalk's base model. |
| **MeiGen-AI/InfiniteTalk weights** | `huggingface.co/MeiGen-AI/InfiniteTalk` — audio-condition weights, `single/infinitetalk.safetensors`. |
| **chinese-wav2vec2-base** | `huggingface.co/TencentGameMate/chinese-wav2vec2-base` — InfiniteTalk's audio encoder (needs an extra `model.safetensors` pull from `refs/pr/1`; see `docs/environment.md`). |
| **InfiniteTalk upstream repo** | `github.com/MeiGen-AI/InfiniteTalk`, put on `PYTHONPATH` via `$INFINITETALK_ROOT`. Pinned at commit **`50aa0a94184315407a991ae804d9b58d6d311ba8`** (the checkout currently at `/home/work/.local/InfiniteTalk` on this box). |
| **`$METRICS_ROOT` tooling** | `shape_predictor_68_face_landmarks.dat` (dlib mouth landmarks), `checkpoints/auxiliary/syncnet_v2.model` (SyncNet), the `lpips` pip package (in the `omniavatar` env). SyncNet inference code from the `eval`/`syncnet_python` packages. On this box: `/home/work/.local/eval_metrics`. |
| **Hallo3 recon clips** | BUNDLED at `data/recon_clips/<hash>.{mp4,wav}` (all 10) — no fetch needed. Point `--video_dir`/`--audio_dir` / `$RECON_DATA_DIR` there. |

See `docs/environment.md` and `docs/data.md` for exact setup and every path to re-point.

---

## Example media (`examples/`)

- `videos/infinitetalk_smoke_generation.mp4` — a full 20-step InfiniteTalk generation (proof the
  stack works end-to-end).
- `frames/ode_x0_decoded_frame*.png` — a decoded `x0` from a real ODE trajectory (coherent face at
  6 steps — validates the x0 derivation).
- `inputs/*.mp4|.wav` — 2 Hallo3 recon clips for quick tests. **All 10** are bundled in
  `data/recon_clips/`. *Provenance:* these are from the Hallo3 validation set; keep private /
  obtain the full set from the dataset if redistributing.
