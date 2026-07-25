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

> Note: the OmniAvatar analysis results were not available on this machine, so this is a standalone
> InfiniteTalk analysis (the cross-model comparison is left for when those results are recovered).

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
  7 configs. Regenerate with `scripts/plot_trajectory_geometry_overlay.py`.
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

## Method notes / reproducibility

- **x0 derivation:** InfiniteTalk has no x0 head (hand-written flow-matching Euler on velocity);
  `x0 = x_t + sigma·noise_pred`, `sigma = t/1000`. CFG-in-velocity ≡ CFG-in-x0 (affine), so the
  trajectories are comparable to OmniAvatar's x0-space dumps.
- **Force-square** 640² bucket for all references → uniform `[16,21,80,80]` latents.
- **Eager execution** (`TORCHDYNAMO_DISABLE=1`): the box lacks python3.10 dev headers, so InfiniteTalk's
  one `@torch.compile` helper can't JIT; eager is the reference semantics and cleaner for trajectories.
- **Compute:** Stage-1 sweep ~11 h on 8×A100 (job-sharded, 70 trajectories); Stage-2 decode ~55 min/config,
  metrics re-sharded 35-way across 8 GPUs (~90 min for all 7). Full pipeline scripts under `scripts/`.
