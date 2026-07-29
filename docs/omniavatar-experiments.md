# OmniAvatar experiment inventory

Adapted from `results/omniavatar/data/README.md` — the committed copy of the original study's
`all_csvs/README.md`. That file has the full per-metric definitions, the mask/region breakdown,
and every generation command exactly as originally run; **this doc doesn't repeat that** — it adds
the unified `configs/registry.yaml` id for every experiment and points each family at the launcher
or command that regenerates it **in this repo's layout** (`scripts/omniavatar/...`, not the
original `FastGen/scripts/...` / `OmniAvatar/scripts/...` paths). When the two disagree on a path,
`configs/registry.yaml` and `scripts/omniavatar/` are canonical for this repo.

None of the 27 committed CSVs need regenerating to use them — they're already at
`results/omniavatar/data/*.csv`. The commands below are for reproducing or extending the study.

## Common setup (all experiments)

- **Teacher**: OmniAvatar 14B V2V (Wan2.1 T2V-14B base + LoRA + audio modules), `in_dim=65`.
  `$TEACHER_CKPT` (default `step-10500.pt`); base weights under `$WEIGHTS_ROOT/Wan2.1-T2V-14B/`.
- **Validation data**: 10 Hallo3 recon samples, `$RECON_DATA_DIR` — pre-aligned 512×512 face
  crops, 25 fps, 250 frames/sample. Sample IDs: `data/recon_sample_names.txt`.
- **ODE schedule**: 50 steps, shift=5.0, `t ∈ [0.999, 0]` with `t_shifted = shift·t / (1 + (shift-1)·t)`.
- **Metrics** (per-step, on VAE-decoded 512×512 video): pixel_mse, ssim, lpips, lmd (vs GT);
  sharpness, sync_c, sync_d (no-reference). Regions: mouth, upper_face, full (LatentSync mask,
  `data/mask.png`).
- **Registry preprocessing tag**: `configs/registry.yaml` → `models.omniavatar.preprocessing`:
  `resolution: 512, crop: pre-aligned latentsync crop, mask: latentsync_png, regions: [mouth, upper_face, full]`.

## CFG drop modes

| Mode | `--cfg_drop_text` | Unconditional branch |
|---|---|---|
| Text+audio (standard) | `true` | negative text + zero audio |
| Audio-only | `false` | positive text + zero audio |

noCFG (`guidance_scale=1.0`) runs only the conditional branch — drop mode is irrelevant, so there
is one `*_nocfg` CSV, not two.

## Experiment types → script & this-repo launcher

| Type | Script | Launcher in this repo |
|---|---|---|
| 50-step trajectory (fixed CFG) | `generate_omniavatar_ode_pairs_full.py` | `generate_ode_trajectories.sh` (cfg=4.5, hardcoded), `generate_ode_nocfg.sh` (cfg=1.0), `generate_ode_no_audio.sh` (audio ablation). All three omit `--cfg_drop_text`, so they run at the driver's default (`true` = text+audio). Other CFG values / audio-only mode / scheduled CFG have **no dedicated launcher** — call the driver directly. |
| Euler jump (2-step prediction) | `generate_single_step_predictions.py --mode euler_jump` | `run_single_step_both.sh` covers 3 of the 4 CFG-knob combinations (on/on, nocfg/on, nocfg/nocfg) for text+audio mode only — call the driver directly for the 4th cell (on/nocfg) and for every audio-only-mode cell. |
| Fresh noise | `generate_single_step_predictions.py --mode fresh_noise` | `run_single_step_both.sh` (GPU 0, text+audio mode only). |
| Scheduled CFG | `generate_omniavatar_ode_pairs_full.py --cfg_crossover <tau>` | No dedicated launcher — call the driver directly. |

`generate_single_step_predictions.py` also has a `--save_latents` flag (new in this repo, off by
default) that dumps `step_NNN_x0.pt` per landing step — needed to unblock the OmniAvatar-side
straightness re-run (see `docs/status-and-todo.md` § Deferred re-runs).

---

## Group 1 — 14B, text+audio CFG (main paper §5)

`cfg_drop_text=true`. All registry ids prefixed `omni_ta_*`. Folds in the original doc's
"Group 3: Fixed-CFG sweep — Text+audio" (same drop mode, same `trajectory` family, just a
different guidance scale) — the registry treats them as one family.

| Registry id | Shape | CSV | Regenerate (this repo) |
|---|---|---|---|
| `omni_ta_default` | trajectory, cfg=4.5 | `14B_textaudio_perceptual_v2.csv` | `generate_ode_trajectories.sh` → `run_eval_ode_perceptual_v2.sh` |
| `omni_ta_nocfg` | trajectory, cfg=1.0 | `14B_textaudio_trajectory_nocfg.csv` | `generate_ode_nocfg.sh` → `run_eval_ode_perceptual_v2.sh` (point `TRAJ_DIR` at the nocfg output dir) |
| `omni_ta_cfg1.0` | trajectory, cfg=1.0 (scalar sweep) | `14B_textaudio_cfg1.0_trajectory.csv` | direct: `generate_omniavatar_ode_pairs_full.py --guidance_scale 1.0 --cfg_drop_text true` |
| `omni_ta_cfg3.0` | trajectory, cfg=3.0 | `14B_textaudio_cfg3.0_trajectory.csv` | direct: `--guidance_scale 3.0 --cfg_drop_text true` |
| `omni_ta_cfg6.0` | trajectory, cfg=6.0 | `14B_textaudio_cfg6.0_trajectory.csv` | direct: `--guidance_scale 6.0 --cfg_drop_text true` |
| `omni_ta_euler_on_on` | euler, 4.5→4.5 | `14B_textaudio_euler_cfg45_cfg45.csv` | `run_single_step_both.sh` (GPU 1) |
| `omni_ta_euler_nocfg_on` | euler, 1.0→4.5 | `14B_textaudio_euler_nocfg_cfg45.csv` | `run_single_step_both.sh` (GPU 2) |
| `omni_ta_euler_nocfg_nocfg` | euler, 1.0→1.0 | `14B_textaudio_euler_nocfg_nocfg.csv` | `run_single_step_both.sh` (GPU 3) |
| `omni_ta_euler_on_nocfg` | euler, 4.5→1.0 | `14B_textaudio_euler_cfg45_nocfg.csv` | direct: `generate_single_step_predictions.py --mode euler_jump --cfg_step0 4.5 --cfg_teacher 1.0 --cfg_drop_text true` (not in `run_single_step_both.sh`) |
| `omni_ta_fresh_noise` | fresh_noise, cfg=4.5 | `14B_textaudio_fresh_noise.csv` | `run_single_step_both.sh` (GPU 0) |
| `omni_ta_schedule25` | scheduled_cfg, τ=25 | `14B_textaudio_schedule25.csv` | direct: `generate_omniavatar_ode_pairs_full.py --guidance_scale 4.5 --cfg_crossover 25 --cfg_drop_text true` |

## Group 2 — 14B, audio-only CFG (supplementary)

`cfg_drop_text=false`. Registry ids `omni_ao_*` — same 7 shapes as Group 1 (no separate nocfg CSV;
noCFG has no drop-mode branch, so `omni_ta_nocfg` is reused). None of the 3 fixed-CFG launchers or
`run_single_step_both.sh` pass `--cfg_drop_text false`, so **every** Group-2 cell needs a direct
driver call with that flag added; `run_all_metrics_sequential.sh` handles Group 2's *metrics* phase
(it's the "omni audio-only CFG trajectory" + 4-variant loop in that script).

| Registry id | CSV |
|---|---|
| `omni_ao_default` | `14B_audioonly_perceptual_v2.csv` |
| `omni_ao_euler_on_on` | `14B_audioonly_euler_cfg45_cfg45.csv` |
| `omni_ao_euler_nocfg_on` | `14B_audioonly_euler_nocfg_cfg45.csv` |
| `omni_ao_euler_nocfg_nocfg` | `14B_audioonly_euler_nocfg_nocfg.csv` |
| `omni_ao_euler_on_nocfg` | `14B_audioonly_euler_cfg45_nocfg.csv` |
| `omni_ao_fresh_noise` | `14B_audioonly_fresh_noise.csv` |
| `omni_ao_schedule25` | `14B_audioonly_schedule25.csv` |

## Groups 3/4 — Fixed-CFG sweep (scalar 1.0/3.0/4.5/6.0)

Text+audio (Group 3 in the original doc) is fully present and folded into Group 1's table above
(`omni_ta_cfg{1.0,3.0,6.0}` + the shared `omni_ta_default` at 4.5). **Audio-only (Group 4) is
incomplete**: only cfg=4.5 was ever finished (it's `omni_ao_default`, reused from Group 2); the
1.0/3.0/6.0 cells were queued and never completed in the original study, so **there is no CSV and
no registry entry** for them. To fill this in: direct driver calls with
`--guidance_scale {1.0,3.0,6.0} --cfg_drop_text false`.

## Group 5 — LatentSync 1.6 (512×512, supplementary, results-only)

A third model (UNet3D + DDIM + SD VAE + Whisper), **not ported into this repo** — no generation or
analysis script for it exists under `scripts/omniavatar/`. The 6 CSVs are committed read-only
inputs, registry ids `ls_*`, consumed by two ported plotters:
`plot_combined_ode_comparison_latentsync.py` and `plot_all_models_compare.py`.

| Registry id | CSV |
|---|---|
| `ls_default` | `latentsync_perceptual_v2.csv` |
| `ls_nocfg` | `latentsync_trajectory_nocfg.csv` |
| `ls_euler_on_on` | `latentsync_euler_cfg15_cfg15.csv` |
| `ls_euler_nocfg_on` | `latentsync_euler_nocfg_cfg15.csv` |
| `ls_euler_nocfg_nocfg` | `latentsync_euler_nocfg_nocfg.csv` |
| `ls_fresh_noise` | `latentsync_fresh_noise.csv` |

`run_all_metrics_sequential.sh`'s metrics phase covers the LatentSync trajectory + 4 variants too
(reading pre-existing trajectory `.pt` dumps at `$ODE_TRAJ_ROOT_OMNI/latentsync_1.6{,_nocfg}`) —
it does not generate LatentSync trajectories itself.

## Group 6 — Spatial CFG-Difference Probe (Exp 2, supplementary §S1.3)

Feeds the same `x_t` to the 14B teacher at CFG=4.5 and CFG=1.0 (audio-only), decodes both,
diffs per-pixel, partitions by mouth mask. Registry ids `omni_spatial_*`, `results_only: true`
(the metrics — `cfg_diff_raw`, `cfg_diff_relative`, `noise_floor` — have no InfiniteTalk
counterpart; marked `diagnostic_only` in `configs/registry.yaml`'s `metric_rules`).

| Registry id | Protocol | CSV | Regenerate |
|---|---|---|---|
| `omni_spatial_fresh` | fresh_noise | `spatial_probe_fresh_noise.csv` | `spatial_cfg_probe.py --protocol fresh_noise` |
| `omni_spatial_cfg` | trajectory_cfg | `spatial_probe_trajectory_cfg.csv` | `spatial_cfg_probe.py --protocol trajectory_cfg` |
| `omni_spatial_nocfg` | trajectory_nocfg | `spatial_probe_trajectory_nocfg.csv` | `spatial_cfg_probe.py --protocol trajectory_nocfg` |

Plots: `plot_spatial_cfg_probe.py` (line plots, mouth/upper ratio vs step) and
`plot_spatial_cfg_heatmaps.py` (representative-timestep Δ heatmaps).

## Other trajectories on disk (out of scope, no registry entry)

`1.3B/` (1.3B model), `14B_no_audio/` (audio-ablation trajectory dump — its perceptual CSV isn't
in the committed 27), `videopainter_*/` (a fourth model), and an earlier Euler analysis format
under `14B_cfg*.0/euler_perceptual*/` (different, superseded CSV structure). None of these have a
registry id or a committed CSV in this repo.

## Plot scripts (10, all in `scripts/omniavatar/`)

| Script | What it produces | Reads |
|---|---|---|
| `plot_combined_ode_comparison.py` | Group 1: all variants overlaid | `--analysis_root` (default `$ODE_ANALYSIS_ROOT_OMNI`) |
| `plot_combined_ode_comparison_audio_only.py` | Group 2: all variants overlaid | same |
| `plot_combined_ode_comparison_latentsync.py` | Group 5: all variants overlaid | same |
| `plot_trajectory_cfg_comparison.py` | CFG=4.5 vs noCFG trajectory only | same |
| `plot_cfg_mode_compare.py` | Group 1 vs Group 2 vs noCFG | same |
| `plot_all_models_compare.py` | Cross-model: OmniAvatar (both drop modes) vs LatentSync | same |
| `plot_exp1_schedule_compare.py` | Scheduled CFG vs fixed CFG vs Euler variants (paper Exp 1) | same |
| `plot_spatial_cfg_probe.py` | Group 6 line plots | `--probe_dir` (default `$ODE_ANALYSIS_ROOT_OMNI/spatial_cfg_probe`) |
| `plot_spatial_cfg_heatmaps.py` | Group 6 representative-timestep heatmaps | same |
| `plot_mouthweight_ode_results.py` | MouthWeight checkpoint: 1-D CFG frontier | `--analysis_root` (default `$ODE_ANALYSIS_ROOT_OMNI/14B_mouthweight`) |

All ten expect the nested `<config>/<variant>/metrics.csv` shape written by
`run_eval_ode_perceptual_v2.sh` / `run_all_metrics_sequential.sh` — a **live** analysis root, not
the flat merged CSVs at `results/omniavatar/data/`. To regenerate a figure from the committed data
you'd need to first lay the flat CSVs back out into that nested shape (not currently automated).

## Full metric/column definitions, mask math, evaluation-script phases

See `results/omniavatar/data/README.md` — the committed original inventory. It also has the exact
generation commands as originally run (pre-dating this repo's env/path parameterization) and the
per-experiment status table from the original study.
