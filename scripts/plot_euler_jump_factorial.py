#!/usr/bin/env python
"""Stage-2c: figures for the Euler-jump (ODE-straightness) factorial.

Renders the two overlapping 2x2 factorials over (step-0 CFG) x (teacher CFG), mouth region:

  A) audio-off : on=(t5,a4) vs noaudio=(t5,a1)  -> isolates the AUDIO guidance term
  B) all-off   : on=(t5,a4) vs nocfg=(t1,a1)    -> isolates guidance as a whole

Each factorial gets 4 Euler cells overlaid, plus the SEQUENTIAL trajectory at the matching
teacher CFG as a dashed reference. The Euler-vs-sequential gap IS the curvature measurement:
if the ODE path were straight, a single jump from step 0 would reproduce the sequential result.

Usage:
    python scripts/plot_euler_jump_factorial.py \
        --euler_analysis_root ode_analysis_euler_jump \
        --sequential_analysis_root ode_analysis_infinitetalk \
        --output_dir results/figures/euler_jump
"""
import argparse
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REGION = "mouth"

# alias -> (text, audio) used for that leg
CFG = {"on": (5.0, 4.0), "noaudio": (5.0, 1.0), "nocfg": (1.0, 1.0)}
PRETTY = {"on": "on (t5,a4)", "noaudio": "audio-off (t5,a1)", "nocfg": "all-off (t1,a1)"}

FACTORIALS = [
    ("audio", "noaudio", "Factorial A — audio guidance on/off"),
    ("allcfg", "nocfg", "Factorial B — all guidance on/off"),
]

# (step0, teacher) -> colour/style within a factorial
CELL_STYLE = {
    ("on", "on"): ("#08519c", "-", "step0 on -> teacher on"),
    ("on", "off"): ("#e6550d", "-", "step0 on -> teacher off"),
    ("off", "on"): ("#31a354", "-", "step0 off -> teacher on"),
    ("off", "off"): ("#999999", "-", "step0 off -> teacher off"),
}

METRICS = [("pixel_mse", "Pixel MSE vs GT", True), ("ssim", "SSIM vs GT", False),
           ("lpips", "LPIPS vs GT", True), ("sync_c", "Sync-C", False)]


def load_metrics(path):
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    gt = df[df["step"] == -1].groupby(["metric", "region"])["value"].mean()
    steps = (df[df["step"] >= 0]
             .groupby(["step", "metric", "region"])["value"].mean().reset_index())
    return steps, gt


def euler_csv(root, s0, tch):
    return os.path.join(root, f"euler_{s0}_{tch}", "perceptual_v2", "metrics.csv")


def sequential_csv(root, alias):
    T, A = CFG[alias]
    return os.path.join(root, f"infinitetalk_t{T}_a{A}", "perceptual_v2", "metrics.csv")


def series(steps, metric):
    d = steps[(steps["metric"] == metric) & (steps["region"] == REGION)].sort_values("step")
    return d["step"].to_numpy(), d["value"].to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--euler_analysis_root", required=True)
    ap.add_argument("--sequential_analysis_root", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for tag, off_alias, title in FACTORIALS:
        # 4 Euler cells: (step0, teacher) over {on, off}
        cells = {}
        for s0_key, s0 in [("on", "on"), ("off", off_alias)]:
            for t_key, tch in [("on", "on"), ("off", off_alias)]:
                steps, _ = load_metrics(euler_csv(args.euler_analysis_root, s0, tch))
                if steps is not None:
                    cells[(s0_key, t_key)] = steps
        if not cells:
            print(f"[{tag}] no Euler cells found, skipping")
            continue

        # sequential references at each teacher CFG
        seq = {}
        for t_key, alias in [("on", "on"), ("off", off_alias)]:
            steps, gt = load_metrics(sequential_csv(args.sequential_analysis_root, alias))
            if steps is not None:
                seq[t_key] = (steps, gt)

        fig, axes = plt.subplots(1, len(METRICS), figsize=(5.6 * len(METRICS), 5))
        fig.suptitle(f"InfiniteTalk Euler-jump — {title}  ({REGION} region)\n"
                     f"'off' = {PRETTY[off_alias]};  dashed = sequential 50-step reference",
                     fontsize=13, fontweight="bold")

        for ax, (metric, mtitle, lower) in zip(axes, METRICS):
            for key, steps in sorted(cells.items()):
                c, ls, lab = CELL_STYLE[key]
                x, y = series(steps, metric)
                if len(x):
                    ax.plot(x, y, color=c, linestyle=ls, lw=1.9, label=lab)
            # sequential references
            for t_key, (steps, _) in seq.items():
                x, y = series(steps, metric)
                if len(x):
                    ax.plot(x, y, color="k", lw=1.3,
                            linestyle="--" if t_key == "on" else ":",
                            alpha=0.75, label=f"sequential (teacher {t_key})")
            # GT baseline where meaningful
            for t_key, (_, gt) in seq.items():
                if gt is not None and (metric, REGION) in gt.index:
                    ax.axhline(gt[(metric, REGION)], color="tab:red", ls=":", lw=1.2, alpha=0.6)
                    break
            ax.set_title(f"{mtitle}\n({'lower' if lower else 'higher'} = better)")
            ax.set_xlabel("ODE step"); ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3); ax.legend(fontsize=7)

        plt.tight_layout()
        p = os.path.join(args.output_dir, f"euler_factorial_{tag}_{REGION}.png")
        plt.savefig(p, dpi=150); plt.close()
        print("saved", p)

    # ── terminal table across every cell that exists ──
    rows = []
    for s0 in CFG:
        for tch in CFG:
            steps, _ = load_metrics(euler_csv(args.euler_analysis_root, s0, tch))
            if steps is None:
                continue
            last = steps["step"].max()
            for metric, _, _ in METRICS:
                d = steps[(steps["metric"] == metric) & (steps["region"] == REGION)
                          & (steps["step"] == last)]
                if not d.empty:
                    rows.append(dict(step0=s0, teacher=tch, metric=metric,
                                     terminal=float(d["value"].iloc[0])))
    if rows:
        out = os.path.join(args.output_dir, "euler_terminal_values.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print("saved", out)


if __name__ == "__main__":
    main()
