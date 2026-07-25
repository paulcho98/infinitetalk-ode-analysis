#!/usr/bin/env python
"""Stage-2c HEADLINE figure: the 2-D text x audio CFG grid for InfiniteTalk.

The sweep is a 2-D guidance grid (text in {1,2.5,5,7.5}, audio in {1,2,4,6}) sampled at 7 points
in two families. None of the OmniAvatar 1-D plotters express this, so this is built fresh.

Reads each config's perceptual metrics:
    <analysis_root>/infinitetalk_t{T}_a{A}/perceptual_v2/metrics.csv   (schema: step,t,sample,metric,region,value)
and produces, for a chosen terminal-quality metric:
  1. a (text x audio) heatmap of the final-step value (unsampled cells = blank),
  2. the two family line-sweeps (text=5 vs the scaled diagonal) vs audio,
  3. a quality-vs-guidance Pareto scatter (final quality vs total guidance magnitude),
all for the mouth region by default. Terminal value = mean over samples of the last ODE step.

Usage:
    python scripts/plot_cfg_grid_infinitetalk.py \
        --analysis_root <ode_analysis_infinitetalk> --output_dir <out> [--region mouth]
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# the 7 sampled configs (text, audio) — matches run_infinitetalk_ode_sweep_8gpu.sh
CONFIGS = [(5.0, 4.0), (5.0, 1.0), (5.0, 2.0), (5.0, 6.0), (1.0, 1.0), (2.5, 2.0), (7.5, 6.0)]
TEXT_AXIS = [1.0, 2.5, 5.0, 7.5]
AUDIO_AXIS = [1.0, 2.0, 4.0, 6.0]
FAMILY1 = [(5.0, 1.0), (5.0, 2.0), (5.0, 4.0), (5.0, 6.0)]          # text fixed = 5
FAMILY2 = [(1.0, 1.0), (2.5, 2.0), (5.0, 4.0), (7.5, 6.0)]          # text scaled ~1.25x audio

# metric -> (pretty, lower_is_better)
METRIC_INFO = {
    "ssim": ("SSIM (mouth)", False),
    "lpips": ("LPIPS (mouth)", True),
    "pixel_mse": ("Pixel MSE (mouth)", True),
    "lmd": ("LMD (mouth)", True),
    "sync_c": ("Sync-C", False),
    "sync_d": ("Sync-D", True),
    "sharpness": ("Sharpness", False),
}


def cfg_dir(analysis_root, T, A):
    return os.path.join(analysis_root, f"infinitetalk_t{T}_a{A}", "perceptual_v2", "metrics.csv")


def terminal_value(csv_path, metric, region):
    """Mean-over-samples value at the final ODE step for (metric, region). None if unavailable."""
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    df = df[pd.to_numeric(df["step"], errors="coerce") >= 0]
    sub = df[(df["metric"] == metric) & (df["region"] == region)].copy()
    if sub.empty:
        return None
    sub["step"] = sub["step"].astype(int)
    last = sub["step"].max()
    return sub[sub["step"] == last]["value"].astype(float).mean()


def collect(analysis_root, metric, region):
    vals = {}
    for (T, A) in CONFIGS:
        vals[(T, A)] = terminal_value(cfg_dir(analysis_root, T, A), metric, region)
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis_root", required=True,
                    help="dir holding infinitetalk_t{T}_a{A}/perceptual_v2/metrics.csv")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--region", default="mouth", choices=["mouth", "full"])
    ap.add_argument("--metrics", default="ssim,lpips,sync_c",
                    help="comma list of metrics to render heatmaps for")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    # ── 1) heatmaps over the (text x audio) plane ──
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.6), squeeze=False)
    for j, metric in enumerate(metrics):
        pretty, lower = METRIC_INFO.get(metric, (metric, False))
        vals = collect(args.analysis_root, metric, args.region)
        grid = np.full((len(AUDIO_AXIS), len(TEXT_AXIS)), np.nan)
        for (T, A), v in vals.items():
            if v is None or T not in TEXT_AXIS or A not in AUDIO_AXIS:
                continue
            grid[AUDIO_AXIS.index(A), TEXT_AXIS.index(T)] = v
        ax = axes[0, j]
        cmap = "viridis_r" if lower else "viridis"
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(TEXT_AXIS))); ax.set_xticklabels([f"{t:g}" for t in TEXT_AXIS])
        ax.set_yticks(range(len(AUDIO_AXIS))); ax.set_yticklabels([f"{a:g}" for a in AUDIO_AXIS])
        ax.set_xlabel("text guidance"); ax.set_ylabel("audio guidance")
        ax.set_title(f"{pretty}\n(final step, {args.region})")
        for a_i in range(len(AUDIO_AXIS)):
            for t_i in range(len(TEXT_AXIS)):
                if not np.isnan(grid[a_i, t_i]):
                    ax.text(t_i, a_i, f"{grid[a_i, t_i]:.3g}", ha="center", va="center",
                            color="white", fontsize=9, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("InfiniteTalk CFG grid — terminal quality over (text x audio)", fontweight="bold")
    plt.tight_layout()
    p1 = os.path.join(args.output_dir, f"cfg_grid_heatmaps_{args.region}.png")
    plt.savefig(p1, dpi=150); plt.close()
    print("saved", p1)

    # ── 2) family sweeps + 3) Pareto, for the first metric ──
    metric = metrics[0]
    pretty, lower = METRIC_INFO.get(metric, (metric, False))
    vals = collect(args.analysis_root, metric, args.region)
    fig2, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

    for fam, lab, col in [(FAMILY1, "text=5 (audio sweep)", "tab:blue"),
                          (FAMILY2, "text scaled (diagonal)", "tab:red")]:
        xs = [A for (_, A) in fam]
        ys = [vals.get((T, A)) for (T, A) in fam]
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if pts:
            axL.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=col, label=lab)
    axL.set_xlabel("audio guidance"); axL.set_ylabel(pretty)
    axL.set_title(f"{pretty} vs audio guidance"); axL.grid(True, alpha=0.3); axL.legend()

    # Pareto: quality vs total guidance magnitude sqrt(T^2+A^2)
    for (T, A), v in vals.items():
        if v is None:
            continue
        mag = float(np.hypot(T, A))
        axR.scatter(mag, v, s=60)
        axR.annotate(f"t{T:g}a{A:g}", (mag, v), fontsize=8,
                     textcoords="offset points", xytext=(4, 4))
    axR.set_xlabel("guidance magnitude  sqrt(text^2+audio^2)")
    axR.set_ylabel(pretty + ("  (lower=better)" if lower else "  (higher=better)"))
    axR.set_title("Quality vs guidance (Pareto)"); axR.grid(True, alpha=0.3)
    plt.tight_layout()
    p2 = os.path.join(args.output_dir, f"cfg_families_pareto_{metric}_{args.region}.png")
    plt.savefig(p2, dpi=150); plt.close()
    print("saved", p2)

    # dump the collected terminal table for all metrics/regions
    rows = []
    for m in METRIC_INFO:
        for region in ("mouth", "full"):
            for (T, A) in CONFIGS:
                v = terminal_value(cfg_dir(args.analysis_root, T, A), m, region)
                if v is not None:
                    rows.append(dict(text=T, audio=A, metric=m, region=region, terminal=v))
    if rows:
        outcsv = os.path.join(args.output_dir, "terminal_values.csv")
        pd.DataFrame(rows).to_csv(outcsv, index=False)
        print("saved", outcsv)


if __name__ == "__main__":
    main()
