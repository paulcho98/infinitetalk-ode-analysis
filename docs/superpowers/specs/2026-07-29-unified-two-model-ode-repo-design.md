# Unified Two-Model ODE Analysis Repo — Design

**Date:** 2026-07-29
**Status:** Approved by user (interactive design review)

## Motivation

The ODE-trajectory study currently spans three codebases and two machines:

- **InfiniteTalk**: this repo (`infinitetalk-ode-analysis`) — self-contained, current, correct.
- **OmniAvatar Stage 1** (trajectory generation): `FastGen/scripts/` — drivers importing
  `fastgen.networks.OmniAvatar.OmniAvatarWan`.
- **OmniAvatar Stage 2** (analysis/plots): `OmniAvatar/scripts/` — engines, euler-jump,
  plotters, launchers with hardcoded machine paths.

Three copies of the InfiniteTalk scripts have diverged (the copies in `OmniAvatar/scripts`
and `FastGen/scripts` are stale and lack the velocity/`delta_cosine[0]` fixes). The goal is
one repo that runs both models' full experiment suites and supports the cross-model
comparison (step 6 of `docs/status-and-todo.md`).

## Decisions made during design review

1. **Repro level:** parameterized + documented. No hardcoded machine paths; setup
   documented per model; a fresh machine needs manual setup following docs.
2. **Model scope:** OmniAvatar + InfiniteTalk only. latentsync/videopainter baseline
   experiments are out of scope; their results stay where they are.
3. **Re-run scope:** port only. The two OmniAvatar data gaps (cfg4.5 sequential `.pt`
   dump gone; euler runs never saved latents) are scripted-but-deferred, not executed.
4. **Approach:** expand this repo (not a new repo, not a thin comparison repo).
5. **Rename:** GitHub repo `infinitetalk-ode-analysis` → `talking-head-ode-analysis`
   (redirect preserved); local checkout renamed to match.
6. **Vendoring:** all code needed for reproduction lives in the repo. Only model weights,
   the public InfiniteTalk upstream repo, and the eval_metrics tooling stay external,
   each pinned and documented.

## Repo layout

```
talking-head-ode-analysis/
├── README.md                    # two-model scope, per-model quickstart, env matrix
├── configs/
│   ├── machine-omniavatar.env.example   # path roots for the OmniAvatar box
│   ├── machine-sweep.env.example        # path roots for the sweep machine
│   └── registry.yaml            # unified experiment registry
├── models/
│   ├── omniavatar_wan/          # vendored FastGen OmniAvatarWan package (see Vendoring)
│   └── wan_vae/                 # vendored Wan VAE decoder + minimal loader
├── scripts/
│   ├── common/                  # model-agnostic: measure_euler_straightness.py, shared helpers
│   ├── infinitetalk/            # existing scripts, git mv — no logic changes
│   ├── omniavatar/              # ported Stage-1 drivers/launchers, euler-jump, Stage-2 engines, plotters
│   └── comparison/              # step-6 cross-model CSV join + comparison plots (new)
├── data/                        # recon_sample_names.txt, recon_clips/, mask.png (1.9 KB)
├── reference/fastgen_generation/  # frozen snapshot kept for provenance
│                                  # reference/omniavatar_analysis/ deleted once ported
├── results/
│   ├── infinitetalk/{data,figures}     # existing results, git mv
│   ├── omniavatar/{data,figures}       # committed all_csvs (27 CSVs + README, ~13 MB)
│   └── comparison/
└── docs/                        # existing 8 docs updated + omniavatar-experiments.md
```

`.gitignore` continues to exclude `*.pt`, `*.npy`, trajectory/analysis roots. Only
CSVs/JSON/figures cross the machine boundary. Small whitelisted exceptions allowed via
`!path` rules if needed (none planned; `neg_text_emb.pt` is regenerated, not committed).

## Canonical sources

| Scripts | Canonical copy | Stale copies (delete in cleanup) |
|---|---|---|
| InfiniteTalk Stage 1 + 2 | **this repo** | `OmniAvatar/scripts/*_infinitetalk.py`, `FastGen/scripts/generate_infinitetalk_ode_pairs_full.py`, `FastGen/scripts/run_infinitetalk_ode_sweep.sh` |
| OmniAvatar Stage 1 | `FastGen/scripts/generate_omniavatar_ode_pairs_full.py` (byte-identical to `reference/`) | — |
| OmniAvatar Stage 2 | `OmniAvatar/scripts/` (byte-identical to `reference/`) | — |

The stale InfiniteTalk copies predate the `prepare_conditioning()` refactor and the
velocity/`delta_cosine[0]` fixes. Any port that picks them up reintroduces known bugs.

## Vendoring

**`models/omniavatar_wan/`** — promoted from `reference/fastgen_generation/`:
`network.py` (OmniAvatarWan), `wan_model.py` (DiT), `audio_pack.py`, `network_causal.py`,
plus `fastgen_core/network.py` (FastGenNetwork base) and `fastgen_core/noise_schedule.py`.
Internal imports rewritten from `fastgen.networks.*` to the local package (~6 lines).
`fastgen.utils.logging_utils` replaced with a small local shim. Result: OmniAvatar Stage-1
generation and euler-jump run with **no FastGen checkout**.

**`models/wan_vae/`** — `wan_video_vae.py` copied from the OmniAvatar package plus a
~20-line direct loader (instantiate `WanVideoVAE`, load `Wan2.1_VAE.pth`). Replaces the
`OmniAvatar.models.model_manager.ModelManager` import used only for VAE decode in
`eval_ode_perceptual_v2.py` (lazy, line 158) and `decode_ode_trajectory.py` (module level).
Result: Stage-2 decode runs with **no OmniAvatar checkout**.

Vendored code is a port, not a rewrite: file contents unchanged except import lines.

## Porting inventory (→ `scripts/omniavatar/`)

All ports replace hardcoded paths with env-with-default + CLI override (the existing
`INFINITETALK_ROOT` pattern). No logic changes.

**Stage 1:**
- `generate_omniavatar_ode_pairs_full.py` (from FastGen; imports vendored package)
- Launchers: `generate_ode_trajectories.sh`, `generate_ode_nocfg.sh`, `generate_ode_no_audio.sh`

**Euler-jump / fresh-noise (Stage-1-adjacent, mirroring the InfiniteTalk layout):**
- `generate_single_step_predictions.py` + `run_single_step_both.sh`
- New `--save_latents` flag (writes `step_NNN_x0.pt` per jump) so the deferred
  straightness gap-fill is ready to execute.

**Stage 2 engines:** `eval_ode_perceptual_v2.py`, `analyze_ode_trajectory.py`
**Stage 2 aux:** `decode_ode_trajectory.py`, `visualize_ode_stepwise.py`,
`simulate_euler_and_decode.py`, `spatial_cfg_probe.py`
**Stage 2 launchers:** `run_eval_ode_perceptual_v2.sh`, `run_all_metrics_sequential.sh`,
`run_mouthweight_generation.sh`, `run_mouthweight_evaluation.sh`
**Plotters (10):** `plot_combined_ode_comparison{,_audio_only,_latentsync}.py`,
`plot_trajectory_cfg_comparison.py`, `plot_cfg_mode_compare.py`, `plot_all_models_compare.py`,
`plot_exp1_schedule_compare.py`, `plot_mouthweight_ode_results.py`,
`plot_spatial_cfg_probe.py` + `plot_spatial_cfg_heatmaps.py`. Module-level absolute-path
dicts become CLI args or registry lookups.

**Excluded:** `fix_euler_nocfg_cfg15_step0.py` (LatentSync one-shot),
`eval_ode_perceptual.py` (superseded v1), `FastGen/scripts/verify_ode_trajectory.py`
(superseded by `decode_ode_trajectory.py` + vendored VAE), all latentsync/videopainter
generation/analysis scripts.

**LatentSync nuance:** the committed `all_csvs` includes 6 `latentsync_*` CSVs, and two
ported plotters (`plot_combined_ode_comparison_latentsync.py`, `plot_all_models_compare.py`)
read them. Committing those CSVs as read-only inputs is in scope; porting latentsync
generation/analysis is not. Registry entries for latentsync are marked `results-only`.

**Moves within this repo:** existing scripts → `scripts/infinitetalk/` via `git mv`
(no logic changes); `measure_euler_straightness.py` → `scripts/common/`;
`plotters_to_adapt/` deleted (superseded by the real ports);
`reference/omniavatar_analysis/` deleted once its scripts are ported
(git history preserves them).

## Parameterization pattern

Per-machine env files (`configs/machine-*.env`, actual files git-ignored, `.example`
committed) define: `INFINITETALK_ROOT`, `METRICS_ROOT`, `WEIGHTS_ROOT` (Wan shards, VAE
pth, teacher `step-10500.pt`, wav2vec2), `ODE_TRAJ_ROOT`, `ODE_ANALYSIS_ROOT`, and per-env
Python paths (`PY_OMNI`, `PY_FASTGEN`, `PY_IT`). Launchers `source` the machine file;
Python scripts read env vars with sane defaults and accept CLI overrides. This eliminates
all three classes of hardcoded paths found in the audit, including the dead
`/home/work/ode_*` references.

## Environments (documented matrix, no single requirements.txt)

| Env | Used for | Notes |
|---|---|---|
| `omniavatar` (conda) | All Stage-2 metrics (dlib + lpips + SyncNet), OmniAvatar decode | only env with the full metrics stack |
| `fastgen` (conda) | OmniAvatar Stage-1 generation | vendored model needs the FastGen-era torch stack |
| `infinitetalk` (conda/venv) | InfiniteTalk Stage-1 + decode | torch 2.4.1, transformers 4.49.0, diffusers 0.33.1, `TORCHDYNAMO_DISABLE=1` |

README records each env's creation steps (or points to existing docs) and which scripts
need which env.

## Results policy & experiment registry

- `results/omniavatar/data/` ← the 27 CSVs + README from
  `/home/work/.local/ode_analysis/all_csvs/` (~13 MB).
- `results/infinitetalk/` ← existing `results/data` + `results/figures` (git mv).
- `results/comparison/` ← outputs of the comparison stage.
- **Known gap A** (InfiniteTalk per-step euler CSVs, ~3.5 MB, gitignored on the sweep
  machine): documented TODO — requires one commit *from the sweep machine*.

**`configs/registry.yaml`** — one entry per experiment: model, text/audio guidance
scales, drop mode (text+audio vs audio-only), schedule, trajectory dir name, results CSV
path, and preprocessing provenance (512² pre-aligned + LatentSync mask with
mouth/upper_face/full vs 640² force-square + dlib bbox with mouth/full). Bridges
OmniAvatar's 1-D naming (`14B_cfg{1.0,3.0,6.0}`, `14B_nocfg`, …) and InfiniteTalk's 2-D
grid (`infinitetalk_t{T}_a{A}`). Plotters and comparison scripts iterate the registry
instead of hardcoded dicts.

## Comparison stage (`scripts/comparison/`)

Implements the plan in `docs/cross-model-comparison.md`: per-step, per-sample, per-metric
CSV join across models. Normalization rules baked in:

- Each model normalized to its own `step=-1` GT row or its no-CFG baseline.
- Only Sync-C/D compared at face value; sharpness and pixel-MSE flagged non-comparable
  (different crops/regions).
- `upper_face` exists only on the OmniAvatar side; joins are inner on region.
- The four euler replication targets are first-class:
  `14B_textaudio_euler_{cfg45_cfg45, nocfg_cfg45, nocfg_nocfg, cfg45_nocfg}` ↔
  `euler_{on_on, nocfg_on, nocfg_nocfg, on_nocfg}`.
- `on=(t5,a4) ↔ cfg4.5` is an assumption carried as a registry annotation, not an
  equivalence.

## Docs plan

- `README.md`: rewritten for two-model scope — repo map, env matrix, per-model
  quickstart (generate → analyze → plot), external deps with pinned commits/versions.
- Existing 8 docs kept with paths updated; `cross-model-comparison.md` remains the
  comparison spec; `status-and-todo.md` gains a deferred-re-runs section.
- New `docs/omniavatar-experiments.md`: experiment inventory adapted from
  `all_csvs/README.md` — every family, registry entry, regeneration command.

## Out of scope

- Executing the gap-fill re-runs (cfg4.5 sequential with latents; euler-jump with
  latents). Scripted, ready, deferred.
- The InfiniteTalk Stage-2b re-run (needs the 29 GB dump on the sweep machine).
- latentsync/videopainter baselines.
- New experiments, new metrics, changes to experimental logic. **Numerical behavior of
  every ported script must be unchanged.**

## Verification (reproduction against existing outputs)

- **Stage-2 engines:** run each ported engine on one sample/one config from the local
  OmniAvatar data (82 GB on this box) → rows must match committed CSVs (SyncNet
  tolerance noted if nondeterminism appears).
- **Plotters:** regenerate one figure each from committed CSVs → structurally equivalent
  to existing figures.
- **Vendored model package:** import test + state-dict load test (Wan shards + teacher
  ckpt; assert key/param counts), plus one 2-step single-sample GPU smoke run of the
  Stage-1 driver end-to-end.
- **Vendored VAE:** decode one saved `step_049_x0.pt` → frames match the
  ModelManager-loaded decode bit-for-bit (same weights, same code path).
- **InfiniteTalk scripts:** `git mv` only → import checks; `examples/` smoke assets
  already validate behavior.
- **Comparison stage:** joins committed CSVs; row counts must reconcile with the
  documented 6031 (OmniAvatar) vs 5027 (InfiniteTalk) accounting.

## Cleanup in other repos (after unified repo is verified)

- `OmniAvatar-Train`: delete `analyze_ode_trajectory_infinitetalk.py`,
  `eval_ode_perceptual_v2_infinitetalk.py`, ported ODE scripts/launchers; commit pointing
  to the unified repo.
- `FastGen`: delete the stale InfiniteTalk files (`generate_infinitetalk_ode_pairs_full.py`
  — untracked, `run_infinitetalk_ode_sweep.sh`) and the ODE-study driver
  `generate_omniavatar_ode_pairs_full.py` (now canonical in the unified repo). Keep
  `generate_omniavatar_ode_pairs.py` and `generate_ode_trajectories.py` — those are
  FastGen distillation assets, not part of this study. The `OmniAvatarWan` network
  wrapper stays in FastGen (used for distillation); the unified repo's vendored copy is
  the frozen study version.

## Implementation notes

Execution will be delegated to subagents by difficulty: mechanical path
parameterization of plotters/launchers → haiku/sonnet; engine ports and vendored-package
import surgery → sonnet; registry + comparison logic and final review → opus/fable.
Verification gates between phases (vendored package must pass its load test before
Stage-1 scripts port against it; engines must reproduce CSVs before plotters point at
them).
