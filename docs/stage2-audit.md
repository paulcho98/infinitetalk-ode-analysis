# Stage-2 script port audit

Classification of every OmniAvatar Stage-2 script by how much work it needs for InfiniteTalk. The two
metric/analysis engines (rows 1–2 below) are ALREADY ADAPTED in `scripts/` — this table is mainly a
guide for finishing the **plotters** (`scripts/plotters_to_adapt/`) and deciding what to skip.

InfiniteTalk deltas that drive every change:
- Latent `80×80` (not 64×64); decoded frames `640×640` (not 512×512).
- Config dirs `infinitetalk_t{T}_a{A}` for a **2-D** CFG grid (OmniAvatar used a **1-D** cfg sweep:
  `14B_cfg1.0/3.0/6.0`, `14B_nocfg`, `14B_audio_only_cfg*`, `14B_schedule25`).
- No LatentSync mouth mask → dlib mouth-bbox (points 48-67); regions = **mouth + full** (drop `upper_face`).
- Same Wan VAE → decoding compatible (but decode with InfiniteTalk's `WanVAE`).

| Script | Class | What it needs |
|---|---|---|
| `eval_ode_perceptual_v2.py` | PARAM → **DONE** (`..._infinitetalk.py`) | dims, mask→dlib mouth-bbox, InfiniteTalk WanVAE, Hallo3 GT, `METRICS_ROOT` |
| `analyze_ode_trajectory.py` | PARAM → **DONE** (`..._infinitetalk.py`) | latent 64→80, mouth region from down-projected dlib bbox, GT encode, tick guards |
| `decode_ode_trajectory.py` | TRIVIAL | pure VAE decode-to-video; mask overlay auto-skips at 640px; just CLI paths (in `reference/`) |
| `plot_combined_ode_comparison.py` | PARAM | repoint its 5-variant CSV-path dict + `TRAJ_DIR` → 7 `infinitetalk_t*_a*/perceptual_v2/metrics.csv`; drop `upper_face` panels |
| `plot_combined_ode_comparison_audio_only.py` | PARAM | same, was `14B_audio_only_cfg/*` |
| `plot_combined_ode_comparison_latentsync.py` | PARAM | same, was `latentsync_1.6/*` |
| `plot_trajectory_cfg_comparison.py` | PARAM | simplest: 2 hardcoded CSV paths (cfg/nocfg) + `TRAJ_DIR`; covers only 2 of 7 grid points |
| `plot_cfg_mode_compare.py` | PARAM | 3-variant path dict + `TRAJ_DIR` |
| `plot_all_models_compare.py` | PARAM | `MODELS` entries assume a `(cfg,nocfg)` pair; pick representative grid points (e.g. cfg=`t5_a2`, nocfg=`t1_a1`) |
| `plot_mouthweight_ode_results.py` | **REWORK** | its `CFG_FRONTIER` + Pareto plot are structurally a 1-D cfg sweep {1,3,4.5,6}; rebuild for the 2-D grid or skip |
| `simulate_euler_and_decode.py` | REWORK / N/A | RE-RUNS the OmniAvatar 14B teacher with OmniAvatar conditioning — model-specific; would need the InfiniteTalk model, likely not needed |
| `fix_euler_nocfg_cfg15_step0.py` | N/A | one-shot LatentSync bugfix; not reusable |

## Doing the 6 PARAM plotters
All read a `metrics.csv` with schema `step,t,sample,metric,region,value` (preserved by
`eval_ode_perceptual_v2_infinitetalk.py`) and overlay one curve per variant. To port each:
1. Replace the hardcoded OmniAvatar variant dict (paths like `.../14B_cfg3.0/perceptual_v2/metrics.csv`)
   with the 7 `.../infinitetalk_t{T}_a{A}/perceptual_v2/metrics.csv`.
2. Replace `TRAJ_DIR` (used only for `load_schedule`/`ode_schedule.json`) with any one config dir.
3. Remove/empty the `upper_face` panels (region set is now `mouth`, `full`).
4. Update legend labels/colors for the `t{T}_a{A}` variants.

## The one NEW artifact to build (not a port)
A **2-D `text × audio` heatmap / Pareto view**: the grid is 2-D (7 points over text∈{1,2.5,5,7.5},
audio∈{1,2,4,6}), so the natural read is a heatmap of a final-quality metric (e.g. terminal-step
mouth-SSIM or SyncNet) over the (T,A) plane, plus a quality-vs-guidance Pareto scatter. Build this
fresh — none of the 1-D OmniAvatar plotters express it. This is the headline figure for the CFG study.
