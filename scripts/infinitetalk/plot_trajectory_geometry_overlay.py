#!/usr/bin/env python
"""Stage-2c: overlay latent-trajectory geometry (Stage-2b) for all 7 InfiniteTalk CFG configs.

Reads the per-config geometry JSONs emitted by analyze_ode_trajectory_infinitetalk.py and
overlays them. Produces `trajectory_geometry_overlay.png`.

Panels:
  1. x0 -> GT cosine similarity (how aligned the denoised prediction is with the GT latent)
  2. Per-step improvement in x0 -> GT MSE  (`delta_mse`, SIGNED: >0 = moved closer to GT)
  3. Trajectory velocity ||x0(t) - x0(t-1)||^2  (`velocity`, step size) -- only if present

NOTE ON PANELS 2/3: these are DIFFERENT quantities and were previously conflated.
`delta_mse` is signed and MUST NOT go on a log axis (negative steps would be silently
dropped). `velocity` is a squared norm, is non-negative, and is the true step size.
Geometry JSONs produced before the `velocity` metric was added simply omit panel 3.

Usage:
    # from committed results/
    python scripts/infinitetalk/plot_trajectory_geometry_overlay.py \
        --geometry_dir results/infinitetalk/data --output_dir results/infinitetalk/figures/trajectory

    # from a live analysis root (<root>/infinitetalk_t{T}_a{A}/trajectory/*.json)
    python scripts/infinitetalk/plot_trajectory_geometry_overlay.py \
        --analysis_root ode_analysis_infinitetalk --output_dir results/infinitetalk/figures/trajectory
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONFIGS = [(5.0, 1.0), (5.0, 2.0), (5.0, 4.0), (5.0, 6.0), (1.0, 1.0), (2.5, 2.0), (7.5, 6.0)]
FAMILY1 = {(5.0, 1.0), (5.0, 2.0), (5.0, 4.0), (5.0, 6.0)}   # blues (by audio)
BLUES = {1.0: "#9ecae1", 2.0: "#6baed6", 4.0: "#3182bd", 6.0: "#08519c"}
REDS = {(1.0, 1.0): "#fcae91", (2.5, 2.0): "#fb6a4a", (5.0, 4.0): "#de2d26", (7.5, 6.0): "#a50f15"}


def style(T, A):
    if (T, A) in FAMILY1:
        return BLUES.get(A, "tab:blue"), "-", f"t{T:g} a{A:g}"
    return REDS.get((T, A), "tab:red"), "--", f"t{T:g} a{A:g}"


def load_geometry(args, T, A):
    """Locate a config's geometry JSON in either the committed or the live layout."""
    cands = []
    if args.geometry_dir:
        cands.append(os.path.join(args.geometry_dir, f"geometry_t{T}_a{A}.json"))
    if args.analysis_root:
        cands += sorted(glob.glob(os.path.join(
            args.analysis_root, f"infinitetalk_t{T}_a{A}", "trajectory", "*.json")))
    for p in cands:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry_dir", default=None,
                    help="dir of committed geometry_t{T}_a{A}.json (e.g. results/infinitetalk/data)")
    ap.add_argument("--analysis_root", default=None,
                    help="live Stage-2b root (<root>/infinitetalk_t{T}_a{A}/trajectory/)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--region", default="full", choices=["full", "mouth"])
    args = ap.parse_args()
    if not args.geometry_dir and not args.analysis_root:
        ap.error("pass --geometry_dir and/or --analysis_root")
    os.makedirs(args.output_dir, exist_ok=True)

    data = {}
    for T, A in CONFIGS:
        d = load_geometry(args, T, A)
        if d is not None:
            data[(T, A)] = d
    if not data:
        print("no geometry JSONs found"); return
    print(f"loaded {len(data)}/{len(CONFIGS)} configs")

    R = args.region
    has_velocity = any(f"{R}_velocity" in d for d in data.values())
    npanel = 3 if has_velocity else 2
    fig, axes = plt.subplots(1, npanel, figsize=(7.5 * npanel, 5.5))
    fig.suptitle(f"InfiniteTalk latent-trajectory geometry ({R} region) — 7-config CFG sweep",
                 fontsize=14, fontweight="bold")

    # ── Panel 1: x0 → GT cosine ──
    ax = axes[0]
    for (T, A), d in data.items():
        c, ls, lab = style(T, A)
        y = d[f"{R}_cosine"]
        ax.plot(range(len(y)), y, color=c, linestyle=ls, lw=1.8, label=lab)
    ax.set_title(f"x0 → GT cosine similarity ({R})")
    ax.set_xlabel("ODE step"); ax.set_ylabel("cosine(x0, GT latent)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8, ncol=2)

    # ── Panel 2: signed per-step MSE improvement (LINEAR axis — values go negative) ──
    ax = axes[1]
    for (T, A), d in data.items():
        c, ls, lab = style(T, A)
        y = d[f"{R}_delta_mse"][1:]           # step 0 is a definitional 0
        ax.plot(range(1, len(y) + 1), y, color=c, linestyle=ls, lw=1.5, label=lab)
    ax.axhline(0, color="k", lw=1, alpha=0.6)
    ax.set_title(f"Per-step improvement in x0→GT MSE ({R})\n(>0 = moved closer to GT)")
    ax.set_xlabel("ODE step"); ax.set_ylabel("MSE(t−1) − MSE(t)")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: true trajectory velocity (non-negative → log axis is safe) ──
    if has_velocity:
        ax = axes[2]
        for (T, A), d in data.items():
            if f"{R}_velocity" not in d:
                continue
            c, ls, lab = style(T, A)
            y = d[f"{R}_velocity"][1:]        # step 0 has no predecessor
            ax.plot(range(1, len(y) + 1), y, color=c, linestyle=ls, lw=1.5, label=lab)
        ax.set_yscale("log")
        ax.set_title(f"Trajectory velocity  ‖x0(t) − x0(t−1)‖²  ({R})")
        ax.set_xlabel("ODE step"); ax.set_ylabel("step size (MSE)")
        ax.grid(True, alpha=0.3)
    else:
        print("NOTE: no `{R}_velocity` in these JSONs (produced before the metric was added); "
              "re-run Stage-2b to get the true-velocity panel.".replace("{R}", R))

    plt.tight_layout()
    out = os.path.join(args.output_dir, "trajectory_geometry_overlay.png")
    plt.savefig(out, dpi=150); plt.close()
    print("saved", out)


if __name__ == "__main__":
    main()
