#!/usr/bin/env python
"""Two-line comparisons of the DEFAULT InfiniteTalk config against a CFG baseline (mouth region).

Two variants, each contrasting the default (text=5, audio=4) with one ablation:
  vs_t1a1  — all guidance off (text=1, audio=1): isolates the effect of CFG as a whole.
  vs_t5a1  — audio guidance off, text held at 5 (text=5, audio=1): isolates the AUDIO term.

For each variant it regenerates the per-config figure set restricted to the mouth region,
so every panel carries exactly two curves.

Reads the committed results/ layout:
    results/data/perceptual_t{T}_a{A}.csv   (step,t,sample,metric,region,value; step=-1 = GT)
    results/data/geometry_t{T}_a{A}.json

Usage:
    python scripts/plot_default_vs_baseline.py --data_dir results/data \
        --output_dir results/figures/compare_default
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT = (5.0, 4.0)
VARIANTS = [
    ((1.0, 1.0), "vs_t1a1", "no guidance (t=1, a=1)"),
    ((5.0, 1.0), "vs_t5a1", "no audio guidance (t=5, a=1)"),
]
REGION = "mouth"

# default = solid dark blue; each baseline gets its own dashed contrast colour
STYLE_DEFAULT = ("#08519c", "-", "default  t=5, a=4")
STYLE_BASE = {"vs_t1a1": ("#e6550d", "--"), "vs_t5a1": ("#31a354", "--")}

REF_METRICS = [("pixel_mse", "Pixel MSE vs GT", True),
               ("ssim", "SSIM vs GT", False),
               ("lpips", "LPIPS vs GT", True),
               ("lmd", "LMD (lip landmarks) vs GT", True)]
NOREF_METRICS = [("sharpness", "Mouth sharpness (Laplacian var)"),
                 ("sync_d", "Sync-D (lower = better)"),
                 ("sync_c", "Sync-C (higher = better)")]


def load_perceptual(data_dir, T, A):
    p = os.path.join(data_dir, f"perceptual_t{T}_a{A}.csv")
    if not os.path.exists(p):
        return None, None
    df = pd.read_csv(p)
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    gt = df[df["step"] == -1].groupby(["metric", "region"])["value"].mean()
    steps = (df[df["step"] >= 0]
             .groupby(["step", "metric", "region"])["value"].mean().reset_index())
    return steps, gt


def load_geometry(data_dir, T, A):
    p = os.path.join(data_dir, f"geometry_t{T}_a{A}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def series(steps, metric, region=REGION):
    d = steps[(steps["metric"] == metric) & (steps["region"] == region)].sort_values("step")
    return d["step"].to_numpy(), d["value"].to_numpy()


def curves(default_pack, base_pack, tag, base_label):
    """Yield (data_pack, colour, linestyle, label) for the two lines, default first."""
    bc, bls = STYLE_BASE[tag]
    yield default_pack, STYLE_DEFAULT[0], STYLE_DEFAULT[1], STYLE_DEFAULT[2]
    yield base_pack, bc, bls, base_label


def fig_reference(dp, bp, tag, base_label, outdir):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(f"InfiniteTalk — reference metrics ({REGION}):  default vs {base_label}",
                 fontsize=14, fontweight="bold")
    for ax, (metric, title, lower) in zip(axes, REF_METRICS):
        for (steps, _), c, ls, lab in curves(dp, bp, tag, base_label):
            x, y = series(steps, metric)
            ax.plot(x, y, color=c, linestyle=ls, lw=2, label=lab)
        ax.set_title(f"{title}\n({'lower' if lower else 'higher'} = better)")
        ax.set_xlabel("ODE step"); ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    plt.tight_layout()
    p = os.path.join(outdir, "reference_metrics_mouth.png")
    plt.savefig(p, dpi=150); plt.close(); return p


def fig_noref(dp, bp, tag, base_label, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"InfiniteTalk — no-reference metrics ({REGION}):  default vs {base_label}",
                 fontsize=14, fontweight="bold")
    for ax, (metric, title) in zip(axes, NOREF_METRICS):
        gtval = None
        for (steps, gt), c, ls, lab in curves(dp, bp, tag, base_label):
            x, y = series(steps, metric)
            ax.plot(x, y, color=c, linestyle=ls, lw=2, label=lab)
            if gt is not None and (metric, REGION) in gt.index:
                gtval = gt[(metric, REGION)]
        if gtval is not None:
            ax.axhline(gtval, color="k", ls=":", lw=2, label=f"GT ({gtval:.2f})")
        ax.set_title(title); ax.set_xlabel("ODE step"); ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    plt.tight_layout()
    p = os.path.join(outdir, "noref_metrics_mouth.png")
    plt.savefig(p, dpi=150); plt.close(); return p


def fig_gt_similarity(dg, bg, tag, base_label, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"InfiniteTalk — x0 vs GT latent ({REGION}):  default vs {base_label}",
                 fontsize=14, fontweight="bold")
    bc, bls = STYLE_BASE[tag]
    lines = [(dg, STYLE_DEFAULT[0], STYLE_DEFAULT[1], STYLE_DEFAULT[2]),
             (bg, bc, bls, base_label)]
    for ax, (key, title, ylab) in zip(axes, [
            (f"{REGION}_cosine", "x0 → GT cosine similarity", "cosine(x0, GT latent)"),
            (f"{REGION}_mse", "x0 → GT MSE", "MSE to GT latent"),
            (f"{REGION}_delta_mse", "Per-step improvement in x0→GT MSE\n(>0 = moved closer to GT)",
             "MSE(t−1) − MSE(t)")]):
        for g, c, ls, lab in lines:
            if g is None or key not in g:
                continue
            y = g[key]
            if key.endswith("delta_mse"):
                ax.plot(range(1, len(y)), y[1:], color=c, linestyle=ls, lw=1.6, label=lab)
            else:
                ax.plot(range(len(y)), y, color=c, linestyle=ls, lw=2, label=lab)
        if key.endswith("delta_mse"):
            ax.axhline(0, color="k", lw=1, alpha=0.6)
            ax.set_yscale("symlog", linthresh=1e-4)
        ax.set_title(title); ax.set_xlabel("ODE step"); ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    plt.tight_layout()
    p = os.path.join(outdir, "gt_similarity_mouth.png")
    plt.savefig(p, dpi=150); plt.close(); return p


def fig_variance(dg, bg, tag, base_label, outdir):
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.5))
    fig.suptitle(f"InfiniteTalk — inter-sample variance ({REGION}):  default vs {base_label}",
                 fontsize=13, fontweight="bold")
    bc, bls = STYLE_BASE[tag]
    for g, c, ls, lab in [(dg, STYLE_DEFAULT[0], STYLE_DEFAULT[1], STYLE_DEFAULT[2]),
                          (bg, bc, bls, base_label)]:
        if g is None or f"{REGION}_x0_variance" not in g:
            continue
        y = g[f"{REGION}_x0_variance"]
        ax.plot(range(len(y)), y, color=c, linestyle=ls, lw=2, label=lab)
    ax.set_title("Variance of per-sample mouth-region mean")
    ax.set_xlabel("ODE step"); ax.set_ylabel("variance across samples")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    plt.tight_layout()
    p = os.path.join(outdir, "inter_sample_variance_mouth.png")
    plt.savefig(p, dpi=150); plt.close(); return p


def fig_summary(dp, bp, dg, bg, tag, base_label, outdir):
    """Condensed 2-panel read: where sync is won, and where GT-alignment settles."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"InfiniteTalk — summary ({REGION}):  default vs {base_label}",
                 fontsize=13, fontweight="bold")
    bc, bls = STYLE_BASE[tag]

    ax = axes[0]
    gtval = None
    for (steps, gt), c, ls, lab in curves(dp, bp, tag, base_label):
        x, y = series(steps, "sync_c")
        ax.plot(x, y, color=c, linestyle=ls, lw=2, label=lab)
        if gt is not None and ("sync_c", REGION) in gt.index:
            gtval = gt[("sync_c", REGION)]
    if gtval is not None:
        ax.axhline(gtval, color="k", ls=":", lw=2, label=f"GT ({gtval:.2f})")
    ax.set_title("Lip-sync (Sync-C, higher = better)")
    ax.set_xlabel("ODE step"); ax.set_ylabel("Sync-C")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    ax = axes[1]
    for g, c, ls, lab in [(dg, STYLE_DEFAULT[0], STYLE_DEFAULT[1], STYLE_DEFAULT[2]),
                          (bg, bc, bls, base_label)]:
        if g is None:
            continue
        y = g[f"{REGION}_cosine"]
        ax.plot(range(len(y)), y, color=c, linestyle=ls, lw=2, label=lab)
    ax.set_title("Convergence to GT (x0 → GT cosine)")
    ax.set_xlabel("ODE step"); ax.set_ylabel("cosine(x0, GT latent)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    plt.tight_layout()
    p = os.path.join(outdir, "summary_mouth.png")
    plt.savefig(p, dpi=150); plt.close(); return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="results/data")
    ap.add_argument("--output_dir", default="results/figures/compare_default")
    args = ap.parse_args()

    dT, dA = DEFAULT
    dp = load_perceptual(args.data_dir, dT, dA)
    dg = load_geometry(args.data_dir, dT, dA)
    if dp[0] is None:
        print("default config data not found in", args.data_dir); return

    for (bT, bA), tag, base_label in VARIANTS:
        bp = load_perceptual(args.data_dir, bT, bA)
        bg = load_geometry(args.data_dir, bT, bA)
        if bp[0] is None:
            print(f"skip {tag}: no data for t{bT}_a{bA}"); continue
        outdir = os.path.join(args.output_dir, tag)
        os.makedirs(outdir, exist_ok=True)
        made = [fig_reference(dp, bp, tag, base_label, outdir),
                fig_noref(dp, bp, tag, base_label, outdir),
                fig_gt_similarity(dg, bg, tag, base_label, outdir),
                fig_variance(dg, bg, tag, base_label, outdir),
                fig_summary(dp, bp, dg, bg, tag, base_label, outdir)]
        print(f"[{tag}] default vs {base_label}")
        for p in made:
            print("   saved", p)


if __name__ == "__main__":
    main()
