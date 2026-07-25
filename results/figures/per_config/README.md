# Per-config plots

The native per-config figures emitted by the Stage-2 scripts, one folder per CFG config
(`t{text}_a{audio}`). These complement the cross-config overlays in `results/figures/`.

Each config folder holds 5 plots:

| file | stage | what it shows |
|---|---|---|
| `reference_metrics.png` | 2a | per-step reference metrics (pixel-MSE, SSIM, LPIPS, LMD) for mouth + full frame |
| `noref_metrics.png` | 2a | per-step no-reference metrics (sharpness, Sync-D, Sync-C) with GT baselines |
| `gt_similarity.png` | 2b | x0 → GT latent similarity (cosine / MSE) vs step, mouth + full |
| `inter_sample_variance.png` | 2b | inter-sample variance of the x0 prediction vs step |
| `summary.png` | 2b | combined trajectory-geometry summary |

Configs: `t5.0_a1.0`, `t5.0_a2.0`, `t5.0_a4.0` (default), `t5.0_a6.0` (text-fixed family);
`t1.0_a1.0` (no-CFG), `t2.5_a2.0`, `t7.5_a6.0` (text-scaled diagonal).
