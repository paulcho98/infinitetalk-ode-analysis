# Euler-jump (ODE straightness) factorial — InfiniteTalk port

Port of the OmniAvatar `euler_{cfg}_{cfg}` experiment to InfiniteTalk. **Purpose: reproduce the four
`14B_textaudio_euler_*` CSVs** — the (step-0 CFG × teacher CFG) 2×2 — on the InfiniteTalk baseline.
See "Replication target" below for the exact cell mapping.

> **RUN AND COMPLETE (2026-07-26).** All 7 cells generated (3,500 teacher forwards, ~12.7 h on
> 7×A100) and analysed; results in `results/findings.md` § "ODE straightness — the Euler-jump
> factorial", figures in `results/figures/euler_jump/`, per-step curves in
> `results/data/straightness_*.json`.
>
> **Both pre-flight checks passed.** (1) step-0 `x_t` is bit-identical across configs (`maxdiff=0.0`),
> so reading the step-0 leg from the sweep is valid. Note `t_list[0] = 1000.0` → `σ₀ = 1.0` exactly,
> so `eps_euler` reduces to `x_t_0` and the jump carries no division-amplified error. (2) `euler_on_on`
> step 0 matched the sequential step 0 at `rel_l2 = 2.3e-3`, `cosine = 0.999884` — validating the
> shared `prepare_conditioning`/`predict_noise` refactor against the model.
>
> **One bug was found on first execution:** `load_schedule()` read `ode_schedule.json` from the
> *config* dir, but Stage 1 writes it **per sample** (`<config>/<sample>/ode_schedule.json`). All 7
> cells died on that line in seconds. Fixed with a fallback to the first sample's schedule.
>
> **Headline:** curvature is created by *audio* guidance at step 0 (text contributes ~nothing);
> the matched diagonal gives `on/on` 0.477 vs `nocfg/nocfg` 0.355 terminal `rel_l2` (full). Every
> cell's Sync-C peaks at landing **step 11–15**, not at step 49 — one jump + one teacher call
> recovers ~76% of the 50-step path's peak lip-sync.

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

- **Factorial A (audio)**: `on` × `noaudio` — isolates the audio term. *InfiniteTalk-only extension;
  no OmniAvatar counterpart.*
- **Factorial B (allcfg)**: `on` × `nocfg` — isolates guidance as a whole. **This is the direct
  replication of the OmniAvatar result** (see below).

### Replication target — the OmniAvatar 2×2

Factorial B maps 1:1 onto the four OmniAvatar CSVs this experiment exists to reproduce
(`/home/work/.local/ode_analysis/all_csvs/`, also in the `paper_compat/` bundle):

| OmniAvatar CSV | trace (s₀ → s_t) | regime | our cell |
|---|---|---|---|
| `14B_textaudio_euler_cfg45_cfg45.csv` | 4.5 → 4.5 | guided → guided | `euler_on_on` |
| `14B_textaudio_euler_nocfg_cfg45.csv` | 1.0 → 4.5 | unguided → guided | `euler_nocfg_on` |
| `14B_textaudio_euler_nocfg_nocfg.csv` | 1.0 → 1.0 | unguided → unguided | `euler_nocfg_nocfg` |
| `14B_textaudio_euler_cfg45_nocfg.csv` | 4.5 → 1.0 | guided → unguided | `euler_on_nocfg` |

InfiniteTalk's default `on = (t5, a4)` stands in for OmniAvatar's scalar `cfg4.5`, and
`nocfg = (t1, a1)` for `cfg1.0`. Metrics come from the same `eval_ode_perceptual_v2` pipeline, so
the outputs are directly comparable to those CSVs.

> **`fresh_noise` is NOT part of this.** `generate_single_step_predictions.py` has a second mode
> (`x_t = (1-t)·x0_gt + t·eps`) which produced a separate `fresh_noise/` dir and its own CSV. It is
> **not** one of the four above and is **excluded from the `paper_compat/` bundle**, so it is not
> required to replicate this result and has deliberately not been ported.

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

## Before you run — two pre-flight checks

**1. Verify the step-0 noise is shared across configs.** We read the step-0 leg from an
already-swept trajectory instead of recomputing it (the OmniAvatar original recomputed whenever
`cfg_step0 != 4.5`). The two are equivalent **only if `x_t_0` is identical across configs** — it
should be (same `seed=42`, same shape, no RNG consumed between `manual_seed` and `randn`), but this
has never been checked against real data:

```bash
S=$(ls ode_full_trajectories_infinitetalk/infinitetalk_t5.0_a4.0 | head -1)
python -c "
import torch
a=torch.load(f'ode_full_trajectories_infinitetalk/infinitetalk_t5.0_a4.0/$S/step_000_xt.pt')
b=torch.load(f'ode_full_trajectories_infinitetalk/infinitetalk_t1.0_a1.0/$S/step_000_xt.pt')
print('identical:', torch.equal(a,b), '| maxdiff:', (a.float()-b.float()).abs().max().item())"
```

If this is **not** identical, the step-0 legs aren't comparable and the driver must recompute step 0
with the teacher instead of loading it — do not run the sweep until this is resolved.

**2. Smoke-test `euler_on_on` on one GPU.**

```bash
bash scripts/run_stage2_euler_jump.sh euler_on_on 0
```

Its teacher CFG equals the source trajectory's, so its step-0 output should be **very close** to the
sequential trajectory's step 0 — not bit-identical, since the jump reconstructs `x_t` from `x0_0`
rather than reusing the saved tensor and re-pins the I2V anchor. The straightness `rel_l2` at step 0
should be near zero; a large value means the schedule or conditioning doesn't match.

This also exercises the `prepare_conditioning()` / `predict_noise()` refactor of
`generate_infinitetalk_ode_pairs_full.py`, which the Euler driver and the **already-validated
Stage-1 driver** now share. That refactor is pure code motion but has never been run against the
model, so this test protects both paths.

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
