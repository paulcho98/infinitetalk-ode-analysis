# InfiniteTalk ODE-Trajectory × CFG Analysis — Findings

Reproduction of the OmniAvatar ODE-trajectory study on the **InfiniteTalk** audio-avatar baseline
(Wan2.1-I2V-14B + audio cross-attention). For each denoising step of a 50-step flow-matching sampler
we saved the noisy state `x_t` and the derived denoised prediction `x0`, across a **2-D
classifier-free-guidance grid** (text × audio), then measured how ODE-trajectory quality and geometry
change with step and with CFG.

- **Samples:** 10 Hallo3 identities (same as the OmniAvatar runs), square 480p → latent `[16,21,80,80]`.
- **Grid (7 configs):** text∈{1,2.5,5,7.5}, audio∈{1,2,4,6}, sampled in two families —
  *text-fixed* `(5,1) (5,2) (5,4) (5,6)` and *text-scaled diagonal* `(1,1) (2.5,2) (5,4) (7.5,6)`.
  `(1,1)` is the no-CFG baseline (single forward pass).
- **Metrics (per ODE step):** pixel-MSE, SSIM, LPIPS, LMD (mouth landmark distance), mouth sharpness,
  and **SyncNet Sync-C / Sync-D** (lip-sync), over a dlib mouth bbox and the full 640² frame.
  GT baseline (real validation video) provided for the no-reference metrics.

> Note: everything below is a **standalone InfiniteTalk analysis** — the OmniAvatar results were not
> available on the machine the sweep ran on. They *are* available on the OmniAvatar machine
> (`/home/work/.local/ode_analysis`), and were verified sample- and schema-compatible with these CSVs
> on 2026-07-29, so the cross-model comparison is unblocked but **not yet done**; no number in this
> document has been compared against OmniAvatar. See `docs/cross-model-comparison.md`.

---

## Headline result — CFG trades pixel-fidelity for lip-sync

Terminal (final-step) mouth-region quality over the (text, audio) grid:

| config (t,a) | SSIM ↑ | LPIPS ↓ | Sync-C ↑ (GT 6.22) |
|---|---|---|---|
| (1,1) no-CFG | 0.513 | 0.337 | 5.42 |
| (2.5,2) | 0.503 | 0.331 | 7.04 |
| (5,1) | 0.497 | 0.344 | 5.81 |
| (5,2) | 0.501 | 0.328 | 7.14 |
| (5,4) default | 0.488 | 0.341 | 7.66 |
| (5,6) | 0.478 | 0.344 | 7.85 |
| (7.5,6) | 0.476 | 0.350 | 7.91 |

**Audio guidance drives lip-sync.** Sync-C climbs monotonically with the audio scale — from **5.42**
(no-CFG) to **7.85–7.91** at audio=6, *exceeding* the real-GT baseline (6.22). Text guidance alone
(compare `(5,1)`=5.81 to `(1,1)`=5.42) barely moves sync; audio is the lever. See
`figures/cfg_grid_heatmaps_mouth.png` (Sync-C panel).

**The cost is pixel-fidelity.** As audio guidance rises, terminal SSIM drops (0.513→0.476). Along the
trajectory, SSIM *decreases* from early to late steps at **every** config (e.g. 0.62→0.51 at `(1,1)`):
the early blurry `x0` is a low-variance estimate near the GT mean, and the ODE then adds detail that
pixel metrics penalize. This is the classic **perception–distortion tradeoff** — LPIPS improves over the
same steps (**0.53→0.34**) and lip-sync rises (Sync-C **~1→~7**), confirming the frames get genuinely
*better*, not worse.

**But the pixel-MSE direction flips with guidance** — it is *not* a universal "early steps are closer to
GT" story, and the earlier draft of this doc overstated it. Mouth pixel-MSE, first→last step:

| config | mouth pixel-MSE, step 0 → 49 | direction |
|---|---|---|
| (1,1) no-CFG | 0.0141 → 0.0237 | drifts **away** from GT |
| (2.5,2) | 0.0151 → 0.0254 | away |
| (5,2) | 0.0178 → 0.0259 | away |
| (5,4) default | 0.0256 → 0.0273 | ~flat |
| (5,6) | 0.0373 → 0.0280 | **toward** GT |
| (7.5,6) | 0.0423 → 0.0288 | **toward** GT |

Low guidance starts near the GT mean and drifts off it as detail is added; high guidance starts far
(it commits to generated content immediately) and *converges back*. All configs land in a narrow
terminal band (0.024–0.029) regardless of where they started. This is the pixel-space counterpart of
the cosine reconvergence in Stage 2b below — the same phenomenon measured two ways.

Figures:
- `figures/cfg_grid_heatmaps_mouth.png` — 2-D (text×audio) heatmaps of terminal SSIM / LPIPS / Sync-C.
- `figures/cfg_families_pareto_ssim_mouth.png` — family sweeps + the **perception–distortion frontier**
  (SSIM vs Sync-C, with the GT Sync-C line). Note `(5,1)` sits strictly *inside* the frontier: it is
  dominated by `(1,1)` on SSIM and by everything else on sync, i.e. text-only guidance is wasted budget.
- `figures/ode_curves_reference.png` — per-step MSE/SSIM/LPIPS/LMD (mouth+full), 7 configs overlaid.
- `figures/ode_curves_noref.png` — per-step sharpness / Sync-D / Sync-C with GT baselines.
- `figures/terminal_values.csv` — full terminal table (all metrics × regions × configs).

---

## Latent-trajectory geometry (Stage 2b)

Measuring how the denoised prediction `x0(t)` relates to the encoded GT latent along the trajectory
(cosine similarity + MSE, mouth and full frame), plus step-to-step velocity and inter-sample variance.

**CFG bends the early trajectory away from the GT-mean, then reconverges.** Initial (step-0) x0→GT
cosine (full frame) *decreases* with guidance strength:

| config | x0→GT cosine, step 0 → 49 (full) |
|---|---|
| (1,1) no-CFG | 0.909 → 0.849 |
| (2.5,2) | 0.896 → 0.848 |
| (5,1) | 0.886 → 0.836 |
| (5,2) | 0.878 → 0.840 |
| (5,4) default | 0.842 → 0.840 |
| (5,6) | 0.796 → 0.833 |
| (7.5,6) | 0.783 → 0.833 |

- **No/low CFG** starts *close* to the GT direction (0.91) — the early `x0` is essentially the blurry
  GT-mean — and drifts slightly down as detail is added.
- **High CFG** starts *far* (0.78): strong audio/text guidance immediately commits `x0` to
  generated, audio-driven content rather than the mean, then **reconverges** upward to a common
  endpoint (~0.83–0.85). The path is more curved (dip-then-recover), i.e. less "straight."
- All configs land at a **similar terminal cosine (~0.83–0.85)** — the guidance changes the *route*
  through latent space far more than the *destination's* GT-alignment; the perceptual/lip-sync gains
  from CFG (Stage 2a) come from *where on the GT-consistent manifold* the trajectory lands, not from
  getting closer to the GT latent overall.

This is the geometric counterpart of the perception–distortion tradeoff: guidance trades a
near-the-GT-mean early path for a more curved, detail-committing one.

Figures:
- `figures/trajectory/trajectory_geometry_overlay.png` — x0→GT cosine + per-step MSE improvement,
  7 configs. Regenerate with `scripts/infinitetalk/plot_trajectory_geometry_overlay.py`.
- per-config `gt_similarity.png` / `audio_ablation.png` / `summary.png` under `figures/per_config/`.

> **Corrected in this revision.** The overlay's second panel was previously labelled
> "trajectory velocity ‖x0(t)−x0(t−1)‖²" and drawn on a log axis. The quantity plotted was actually
> `delta_mse` = `MSE(t−1) − MSE(t)`, a **signed** per-step *improvement*, not a step size — so the log
> axis silently dropped every step where the prediction moved away from GT (most late steps at low
> CFG). It is now correctly labelled and drawn on a symlog axis with a zero line, so the negative
> excursions are visible. A genuine `{region}_velocity` = ‖x0(t)−x0(t−1)‖² has been added to
> `analyze_ode_trajectory_infinitetalk.py`; the JSONs in `data/` predate it, so re-running Stage-2b
> is needed to populate the true-velocity panel (the plotter adds it automatically when present).

---

## ODE straightness — the Euler-jump factorial

Stage 2b infers curvature indirectly (via distance to the GT latent). This measures it **directly**:
instead of denoising sequentially, take the step-0 prediction, extrapolate along a *single* Euler
jump to each landing noise level, and ask the teacher to re-predict `x0` there.

```
eps_euler = (x_t_0 − (1−σ₀)·x0_0) / σ₀        # σ₀ = 1.0 exactly, so this reduces to x_t_0
x_t       = (1−σ)·x0_0 + σ·eps_euler           # the jump
x0_pred   = x_t + σ·noise_pred(x_t; teacher CFG)
```

If the path were straight, one jump would reproduce the sequential result at every step. The gap
`rel_l2 = ‖x0_euler − x0_seq‖ / ‖x0_seq‖` **is** the curvature. Two independent CFG legs — the
guidance behind the step-0 prediction we jump *from*, and the guidance of the teacher's re-prediction
— give two overlapping 2×2s over `on=(t5,a4)`, `noaudio=(t5,a1)`, `nocfg=(t1,a1)`; 7 distinct cells,
since `on/on` is shared. Factorial B (`on`×`nocfg`) is the direct replication of OmniAvatar's four
`14B_textaudio_euler_*` CSVs. Each cell is compared against the sequential trajectory whose CFG
matches its **teacher** leg, so the only difference is *how the state was reached*.

Sanity floor: at step 0 the jump reconstructs `x_t_0` exactly, and every cell reports
`rel_l2 ≈ 0.002–0.003` there — bf16 round-off, confirming the schedule and conditioning match.

### Curvature is created by AUDIO guidance at step 0

Terminal `rel_l2`, averaged over the cells sharing each **step-0** leg:

| step-0 leg | text | audio | full | mouth |
|---|---|---|---|---|
| `noaudio` | 5 | 1 | 0.358 | 0.532 |
| `nocfg` | 1 | 1 | 0.361 | 0.551 |
| `on` | 5 | **4** | **0.524** | **0.754** |

`noaudio` (text=5) and `nocfg` (text=1) are indistinguishable despite a 5× difference in text
guidance, while `on` — differing only in audio=4 — is **~46% more curved**. **Text guidance
contributes essentially nothing to ODE curvature; audio guidance is the entire effect.** This is the
geometric echo of the Stage-2a headline, where audio drove lip-sync and text barely moved it.

The **matched diagonal** cells are the clean pure-curvature measurement (origin, teacher and
reference all at the same CFG, so nothing is confounded by a mismatched comparison target):

| cell | full | mouth |
|---|---|---|
| `nocfg → nocfg` | 0.355 | 0.541 |
| `noaudio → noaudio` | 0.361 | 0.528 |
| `on → on` (default) | **0.477** | **0.684** |

Default guidance is **+34%** more curved than unguided. Curvature is also consistently **~1.5× worse
in the mouth region than the frame as a whole** — the bending lives precisely in the audio-driven
content that matters.

### The teacher's audio guidance partially *corrects* the jump

The teacher leg acts in the opposite direction, and more weakly (terminal `rel_l2`, averaged over
cells sharing each teacher leg): `on` **0.400** < `noaudio` 0.445 < `nocfg` 0.461.

The mechanism is visible in *where* each curve peaks. Every cell whose teacher has audio guidance on
peaks mid-trajectory and then **reconverges** — `nocfg→on` 0.703 @ step 12 → 0.561; `on→on` 0.740 @
step 14 → 0.684; `noaudio→on` 0.663 @ step 15 → 0.536 (mouth). Cells with the audio-off teacher
instead grow **monotonically** to step 49. So audio CFG at the landing step actively pulls the
jumped state back toward the sequential manifold: **audio guidance in the origin creates curvature;
audio guidance in the teacher repairs it.**

### There is an optimal landing step — and it is not the end

Jumping *further* is not better. Sync-C over the landing step, mouth region:

| cell (step0 → teacher) | peak Sync-C | @ step | at step 49 | LPIPS @ peak | LPIPS @ 49 |
|---|---|---|---|---|---|
| `on → on` | **5.98** | 11 | 2.15 | 0.382 | 0.456 |
| `on → noaudio` | 5.41 | 13 | 2.18 | 0.375 | 0.456 |
| `noaudio → on` | 5.37 | 13 | 1.84 | 0.392 | 0.468 |
| `noaudio → noaudio` | 4.87 | 15 | 1.74 | 0.396 | 0.468 |
| `nocfg → on` | 4.29 | 12 | 1.19 | 0.408 | 0.528 |
| `on → nocfg` | 4.19 | 11 | 2.12 | 0.424 | 0.457 |
| `nocfg → nocfg` | 3.38 | 11 | 1.23 | 0.483 | 0.529 |
| *sequential* `(5,4)`, 50 steps | *7.89* | *28* | *7.66* | — | *0.341* |

Every cell peaks at **step 11–15** and then decays by ~3× toward step 49. The reason is structural:
as σ→0 the jump `x_t = (1−σ)·x0_0 + σ·eps` collapses onto `x0_0` itself — the blurry step-0
prediction — leaving the teacher no noise budget to synthesize detail, so the output falls back
toward the mean (LPIPS degrades 0.38 → 0.46 over the same range). At a mid noise level there is both
enough signal from step 0 and enough noise left to regenerate detail.

**The distillation-relevant number:** a single Euler jump plus one teacher forward — **2 model calls
instead of 150** — recovers **5.98 of the sequential path's 7.89 peak Sync-C (~76%)** at LPIPS 0.382
vs 0.341, provided you land at step ~11 rather than at the end. The trajectory is far from straight,
but it is *usefully* non-straight over the first quarter.

### Caveats

- **Only the diagonal cells measure pure curvature.** Off-diagonal cells (e.g. `on→nocfg` at 0.568
  full, the largest gap in the study) compare a guided-origin jump against an *unguided* sequential
  reference, so they conflate curvature with the fact that guidance lands somewhere different. They
  answer "does a guided origin make the ODE harder to shortcut" — not "how curved is this path".
- **Sharpness overshoots.** `on→nocfg` reaches terminal mouth sharpness 43.5 against a GT of 25.0;
  extrapolating linearly from a guidance-amplified prediction overshoots into artifacts. Read it as
  a failure mode, not quality.
- **Sync numbers on blurry jumps are noisy.** Per-cell `metrics.csv` row counts vary (4750–4963)
  because SyncNet fails to land a face track on some heavily-blurred jumped frames. This mostly
  affects late landing steps, where Sync-C is already near the floor.
- Stage 2b (GT-latent geometry) was **not** run for these cells — it measures distance-to-GT, which
  is secondary here to distance-to-sequential. Enable with `RUN_2B=1`.

Figures & data:
- `figures/euler_jump/euler_factorial_allcfg_mouth.png` — Factorial B (the OmniAvatar replication).
- `figures/euler_jump/euler_factorial_audio_mouth.png` — Factorial A (audio term isolated).
- `figures/euler_jump/euler_terminal_values.csv`, `data/straightness_*.json` (per-step curves).

---

## Method notes / reproducibility

- **x0 derivation:** InfiniteTalk has no x0 head (hand-written flow-matching Euler on velocity);
  `x0 = x_t + sigma·noise_pred`, `sigma = t/1000`. CFG-in-velocity ≡ CFG-in-x0 (affine), so the
  trajectories are comparable to OmniAvatar's x0-space dumps.
- **Force-square** 640² bucket for all references → uniform `[16,21,80,80]` latents.
- **Eager execution** (`TORCHDYNAMO_DISABLE=1`): the box lacks python3.10 dev headers, so InfiniteTalk's
  one `@torch.compile` helper can't JIT; eager is the reference semantics and cleaner for trajectories.
- **Compute:** Stage-1 sweep ~11 h on 8×A100 (job-sharded, 70 trajectories); Stage-2 decode ~55 min/config,
  metrics re-sharded 35-way across 8 GPUs (~90 min for all 7). Full pipeline scripts under `scripts/`.
- **Euler-jump sweep:** ~12.7 h on 7×A100 (one cell per GPU, 3,500 teacher forwards = 7 cells × 10
  samples × 50 landing steps). Measured ~29 s/forward, so the five 3-call cells dominate; the two
  1-call (`nocfg`-teacher) cells finish in ~⅓ the time. Stage-2 for these cells uses
  `run_stage2_euler_jump_sharded.sh`, which shards the SyncNet-bound metrics phase — the unsharded
  `run_stage2_euler_jump.sh` is correct but far slower.
