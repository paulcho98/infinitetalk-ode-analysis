# Euler-jump (ODE straightness) factorial — InfiniteTalk port

Port of the OmniAvatar `euler_{cfg}_{cfg}` experiment to InfiniteTalk. **Not yet run** — code is
written and syntax-checked, but has never executed against the model (no weights on the authoring
machine). Smoke-test one cell before launching the full sweep.

## What it measures

How **straight** the ODE trajectory is. Instead of denoising sequentially, take the step-0
prediction, extrapolate along a *single* Euler jump to each landing noise level, and ask the teacher
to re-predict `x0` there:

```
sigma      = t_i / num_timesteps
eps_euler  = (x_t_0 - (1 - sigma_0) * x0_0) / sigma_0     # noise implied by step 0
x_t        = (1 - sigma) * x0_0 + sigma * eps_euler        # the jump
x0_pred    = x_t + sigma * noise_pred(x_t; teacher CFG)
```

A perfectly straight trajectory would make the jumped `x0` equal the sequential one at every step.
**The Euler-vs-sequential gap is the curvature.** This matters directly for distillation: a straight
path is what few-step students can imitate.

## The factorial

Two independent CFG settings:

| leg | meaning | supplied by |
|---|---|---|
| **step-0 CFG** | guidance for the prediction we jump *from* | `--step0_traj_dir` (read from that trajectory's saved `step_000_{xt,x0}.pt`) |
| **teacher CFG** | guidance for the re-prediction at each landing step | `--text_cfg_teacher` / `--audio_cfg_teacher` |

InfiniteTalk's CFG is 2-D (text, audio), so instead of OmniAvatar's scalar on/off we use three
aliases and run **two overlapping 2×2s**:

| alias | (text, audio) | meaning |
|---|---|---|
| `on` | (5.0, 4.0) | default |
| `noaudio` | (5.0, 1.0) | audio guidance off, text held at 5 |
| `nocfg` | (1.0, 1.0) | all guidance off |

- **Factorial A (audio)**: `on` × `noaudio` — isolates the audio term.
- **Factorial B (allcfg)**: `on` × `nocfg` — isolates guidance as a whole.

The `on/on` cell is shared, so this is **7 distinct runs, not 8**:

```
             teacher ->   on              noaudio            nocfg
  step0 on              euler_on_on     euler_on_noaudio   euler_on_nocfg
  step0 noaudio         euler_noaudio_on  euler_noaudio_noaudio    -
  step0 nocfg           euler_nocfg_on          -          euler_nocfg_nocfg
```

The two "off" flavours were chosen because the Stage-2a comparison figures
(`results/figures/compare_default/`) already showed these are the two ablations that separate:
removing audio guidance collapses lip-sync while *improving* pixel metrics, and removing all
guidance does the same more strongly.

## Prerequisite

The Stage-1 sweep must already have produced trajectories for **t5.0_a4.0, t5.0_a1.0, t1.0_a1.0** —
the step-0 leg is read from them, never recomputed. All three are in the standard 7-config sweep,
so `run_infinitetalk_ode_sweep_8gpu.sh` covers it. The launcher preflights this and aborts if any
are missing.

## Running

```bash
# 1. generate — 7 cells, one per GPU
bash scripts/run_infinitetalk_euler_jump.sh 50

# 2. straightness + Stage 2a  (Stage 2b is SKIPPED by default — see below)
bash scripts/run_stage2_euler_jump.sh all

# 3. figures
python scripts/plot_euler_jump_factorial.py \
    --euler_analysis_root ode_analysis_euler_jump \
    --sequential_analysis_root ode_analysis_infinitetalk \
    --output_dir results/figures/euler_jump
```

### The straightness number (the actual point)

The direct curvature measurement is **‖x0_euler(t) − x0_sequential(t)‖ at the same step** — if the
path were straight, one jump from step 0 would reproduce the sequential result and the gap would be
zero everywhere. Neither Stage 2a nor Stage 2b computes this: 2a compares against GT *pixels*, 2b
against the GT *latent*. Both answer "how good is it", not "how far did the jump miss".

`scripts/measure_euler_straightness.py` fills that gap, and `run_stage2_euler_jump.sh` runs it first
for every cell. Each cell is compared against the sequential trajectory whose CFG matches its
**teacher** leg, so the only difference is *how the state was reached*:

| cell | compared against |
|---|---|
| `euler_*_on` | `infinitetalk_t5.0_a4.0` |
| `euler_*_noaudio` | `infinitetalk_t5.0_a1.0` |
| `euler_*_nocfg` | `infinitetalk_t1.0_a1.0` |

It reports `abs_l2`, `mse` and `rel_l2` (scale-free — the headline number) per step, for `full` and
`mouth`. It only reads the saved `x0` tensors: no VAE, no GT, no model, no dlib (mouth masks come
from the Stage-2b cache if present, else full-region only). Seconds per cell, and it does **not**
depend on the Stage-2b re-run.

### Stage 2b is off by default

`RUN_2B=1 bash scripts/run_stage2_euler_jump.sh all` enables it. It is skipped because it is the
expensive leg (VAE-encodes the GT clip per sample) and measures distance-to-GT, which is secondary
here to distance-to-sequential. The committed sequential geometry JSONs also still predate the
velocity / `delta_cosine[0]` fixes, so 2b output needs its own re-run before it is trustworthy —
tracked in `status-and-todo.md`, deliberately deprioritized behind this experiment.

Validate ONE cell first:

```bash
bash scripts/run_stage2_euler_jump.sh euler_on_on 0
```

`euler_on_on` is the natural smoke test: its teacher CFG equals the source trajectory's, so its
step-0 output should be **very close** to the sequential trajectory's step 0 (not bit-identical —
the jump reconstructs `x_t` from `x0_0` rather than reusing the saved tensor, and the I2V anchor is
re-pinned). Divergence at step 0 means the schedule or conditioning doesn't match.

## Design notes

**Output is trajectory-shaped, not video-shaped.** The OmniAvatar original decoded to mp4 inline.
Ours writes `step_{i:03d}_{xt,x0}.pt` + `ode_schedule.json` + `input_latents.pt`, which means the
existing Stage-2a (`eval_ode_perceptual_v2_infinitetalk.py`) *and* Stage-2b
(`analyze_ode_trajectory_infinitetalk.py`) both run on it unchanged — so the Euler cells get latent
geometry for free, which the OmniAvatar version never had.

**Schedule verification.** The OmniAvatar original hardcoded `if args.cfg_step0 == 4.5` to decide
whether the saved step-0 `x0` could be reused; pointing `--traj_dir` at a trajectory generated with
a different scale silently loaded the wrong tensor. Ours reads the source trajectory's own scales
from its `ode_schedule.json`, records both legs in the output schedule, and hard-fails on any
mismatch in `num_steps` / `shift` / `seed` / `size`.

**Shared conditioning.** `generate_infinitetalk_ode_pairs_full.py` was refactored to expose
`prepare_conditioning()` and `predict_noise()`; the Euler driver reuses both, so guidance semantics
and conditioning can't drift between the sequential and jump paths. That refactor is pure code
motion but is **untested against the model** — it is the main thing to watch in the smoke test.

## Cost

Each cell is 10 samples × 50 landing steps × one teacher forward (1–3 DiT calls depending on CFG
branch). Roughly comparable to one Stage-1 trajectory config, minus the sequential dependency —
call it ~1.5–3 h per cell on one A100, all 7 in parallel on 8 GPUs. The `nocfg` teacher cells are
~3× cheaper (single forward).
