#!/usr/bin/env python
"""Stage-2c: overlay per-step ODE metric curves for all 7 InfiniteTalk CFG configs.

Analog of the OmniAvatar plot_combined_ode_comparison, ported to the 2-D 7-config grid:
regions are mouth + full (NO upper_face). One curve per config, colored by family
(text=5 audio-sweep in blues; text-scaled diagonal in reds), value vs ODE step.

Reads <analysis_root>/infinitetalk_t{T}_a{A}/perceptual_v2/metrics.csv
      (schema: step,t,sample,metric,region,value ; step=-1 rows are GT baselines).

Usage:
    python scripts/infinitetalk/plot_ode_curves_infinitetalk.py --analysis_root <root> --output_dir <out>
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONFIGS = [(5.0, 1.0), (5.0, 2.0), (5.0, 4.0), (5.0, 6.0), (1.0, 1.0), (2.5, 2.0), (7.5, 6.0)]
FAMILY1 = {(5.0, 1.0), (5.0, 2.0), (5.0, 4.0), (5.0, 6.0)}   # blues (by audio)
BLUES = {1.0: "#9ecae1", 2.0: "#6baed6", 4.0: "#3182bd", 6.0: "#08519c"}
REDS = {(1.0, 1.0): "#fcae91", (2.5, 2.0): "#fb6a4a", (5.0, 4.0): "#de2d26", (7.5, 6.0): "#a50f15"}

# (metric, [regions], lower_is_better)
REF_METRICS = [("pixel_mse", ["mouth", "full"], True), ("ssim", ["mouth", "full"], False),
               ("lpips", ["mouth", "full"], True), ("lmd", ["mouth"], True)]
NOREF_METRICS = [("sharpness", "Sharpness (mouth)"), ("sync_d", "Sync-D (mouth, lower=better)"),
                 ("sync_c", "Sync-C (mouth, higher=better)")]


def style(T, A):
    if (T, A) in FAMILY1:
        return BLUES.get(A, "tab:blue"), "-", f"t{T:g} a{A:g}"
    return REDS.get((T, A), "tab:red"), "--", f"t{T:g} a{A:g}"


def load_cfg(analysis_root, T, A):
    p = os.path.join(analysis_root, f"infinitetalk_t{T}_a{A}", "perceptual_v2", "metrics.csv")
    if not os.path.exists(p):
        return None, None
    df = pd.read_csv(p)
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    gt = df[df["step"] == -1].groupby(["metric", "region"])["value"].mean()
    steps = df[df["step"] >= 0].groupby(["step", "metric", "region"])["value"].mean().reset_index()
    return steps, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis_root", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data = {}
    for (T, A) in CONFIGS:
        steps, gt = load_cfg(args.analysis_root, T, A)
        if steps is not None:
            data[(T, A)] = (steps, gt)
    if not data:
        print("no metrics.csv found under", args.analysis_root); return
    print(f"loaded {len(data)}/{len(CONFIGS)} configs")

    # ── reference metrics: 4 metrics x (mouth,full) ──
    fig, axes = plt.subplots(2, 4, figsize=(26, 11))
    fig.suptitle("InfiniteTalk ODE per-step reference metrics — 7-config CFG sweep",
                 fontsize=15, fontweight="bold")
    for col, (metric, regions, lower) in enumerate(REF_METRICS):
        for row, region in enumerate(["mouth", "full"]):
            ax = axes[row, col]
            if region not in regions:
                ax.axis("off"); continue
            for (T, A), (steps, _) in data.items():
                d = steps[(steps["metric"] == metric) & (steps["region"] == region)].sort_values("step")
                if d.empty:
                    continue
                c, ls, lab = style(T, A)
                ax.plot(d["step"], d["value"], color=c, linestyle=ls, lw=1.6, marker=".", ms=2, label=lab)
            ax.set_title(f"{metric} ({region})"); ax.set_xlabel("ODE step"); ax.grid(True, alpha=0.3)
            if lower and metric == "pixel_mse":
                ax.set_yscale("log")
            if col == 0 and row == 0:
                ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    p1 = os.path.join(args.output_dir, "ode_curves_reference.png")
    plt.savefig(p1, dpi=150); plt.close(); print("saved", p1)

    # ── no-reference metrics (mouth): sharpness, sync_d, sync_c, with GT baselines ──
    fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6))
    fig2.suptitle("InfiniteTalk ODE per-step no-reference metrics (mouth)", fontsize=14, fontweight="bold")
    for col, (metric, title) in enumerate(NOREF_METRICS):
        ax = axes2[col]
        gtvals = []
        for (T, A), (steps, gt) in data.items():
            d = steps[(steps["metric"] == metric) & (steps["region"] == "mouth")].sort_values("step")
            if d.empty:
                continue
            c, ls, lab = style(T, A)
            ax.plot(d["step"], d["value"], color=c, linestyle=ls, lw=1.6, marker=".", ms=2, label=lab)
            if gt is not None and (metric, "mouth") in gt.index:
                gtvals.append(gt[(metric, "mouth")])
        if gtvals:
            ax.axhline(np.mean(gtvals), color="green", ls="--", lw=2, label=f"GT ({np.mean(gtvals):.2f})")
        ax.set_title(title); ax.set_xlabel("ODE step"); ax.grid(True, alpha=0.3); ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    p2 = os.path.join(args.output_dir, "ode_curves_noref.png")
    plt.savefig(p2, dpi=150); plt.close(); print("saved", p2)


if __name__ == "__main__":
    main()
