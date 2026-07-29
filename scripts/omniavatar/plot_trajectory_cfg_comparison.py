"""Plot CFG vs noCFG 50-step trajectory comparison.
Same layout as plot_combined_ode_comparison.py.

Usage:
    python scripts/plot_trajectory_cfg_comparison.py \
        --output_dir /home/work/.local/ode_analysis/14B/combined
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANALYSIS_ROOT = os.environ.get("ODE_ANALYSIS_ROOT_OMNI", "/home/work/.local/ode_analysis")

VARIANTS = {
    "cfg": {
        "path": "14B/perceptual_v2/metrics.csv",
        "label": "50-step CFG=4.5",
        "color": "black",
        "linestyle": "-",
        "linewidth": 2.5,
    },
    "nocfg": {
        "path": "14B/trajectory_nocfg/metrics.csv",
        "label": "50-step noCFG",
        "color": "black",
        "linestyle": "--",
        "linewidth": 2.5,
    },
}

TRAJ_DIR = os.environ.get("ODE_TRAJ_ROOT_OMNI", "/home/work/.local/ode_full_trajectories") + "/14B"


def load_schedule():
    samples = sorted([
        d for d in os.listdir(TRAJ_DIR)
        if os.path.isdir(os.path.join(TRAJ_DIR, d))
        and os.path.isfile(os.path.join(TRAJ_DIR, d, "ode_schedule.json"))
    ])
    with open(os.path.join(TRAJ_DIR, samples[0], "ode_schedule.json")) as f:
        schedule = json.load(f)
    return schedule["t_list"], schedule["num_steps"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--analysis_root", type=str, default=ANALYSIS_ROOT)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve variant CSV paths against --analysis_root (single join point)
    for cfg in VARIANTS.values():
        cfg["path"] = os.path.join(args.analysis_root, cfg["path"])

    t_list, num_steps = load_schedule()
    t_values = np.array(t_list[:num_steps])

    # Load data
    dfs = {}
    for name, cfg in VARIANTS.items():
        df = pd.read_csv(cfg["path"])
        df["step"] = pd.to_numeric(df["step"], errors="coerce")
        df_steps = df[df["step"] >= 0]
        dfs[name] = df_steps.groupby(["step", "metric", "region"])["value"].mean().reset_index()
        print(f"Loaded {name}: {len(df_steps)} rows")

    # GT baselines (average over samples)
    gt_baselines = {}
    gt_df = pd.read_csv(VARIANTS["cfg"]["path"])
    gt_df["step"] = pd.to_numeric(gt_df["step"], errors="coerce")
    gt_agg = gt_df[gt_df["step"] == -1].groupby(["metric", "region"])["value"].mean()
    for (metric, region), val in gt_agg.items():
        gt_baselines[(metric, region)] = val

    # ── Reference metrics: same layout as combined plot ──
    ref_metrics = [
        ("pixel_mse", "Pixel MSE (mouth)", True),
        ("ssim", "SSIM (mouth)", False),
        ("lpips", "LPIPS (mouth)", False),
        ("lmd", "LMD (mouth)", False),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(28, 10))
    fig.suptitle("50-Step Trajectory: CFG=4.5 vs noCFG",
                 fontsize=16, fontweight="bold")

    for col, (metric_name, title, use_log) in enumerate(ref_metrics):
        # Top row: mouth
        ax = axes[0, col]
        for name, cfg in VARIANTS.items():
            sub = dfs[name]
            data = sub[(sub["metric"] == metric_name) & (sub["region"] == "mouth")].sort_values("step")
            if len(data) == 0:
                continue
            ax.plot(data["step"], data["value"],
                    color=cfg["color"], linestyle=cfg["linestyle"],
                    linewidth=cfg["linewidth"], label=cfg["label"],
                    marker=".", markersize=2)

        ax.set_xlabel("ODE Step")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if use_log:
            ax.set_yscale("log")

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        tick_pos = [i for i in [0, 10, 20, 30, 40, 49] if i < len(t_values)]
        ax2.set_xticks(tick_pos)
        ax2.set_xticklabels([f"t={t_values[i]:.2f}" for i in tick_pos])
        ax2.set_xlabel("Timestep t", fontsize=8)

        # Bottom row: upper face (or LMD delta for last column)
        ax_uf = axes[1, col]
        if metric_name != "lmd":
            for name, cfg in VARIANTS.items():
                sub = dfs[name]
                data = sub[(sub["metric"] == metric_name) & (sub["region"] == "upper_face")].sort_values("step")
                if len(data) == 0:
                    continue
                ax_uf.plot(data["step"], data["value"],
                           color=cfg["color"], linestyle=cfg["linestyle"],
                           linewidth=cfg["linewidth"], label=cfg["label"],
                           marker=".", markersize=2)
            ax_uf.set_ylabel(f"{title.split('(')[0]}(Upper Face)")
            ax_uf.set_title(f"{title.split('(')[0]}(Upper Face)")
        else:
            for name, cfg in VARIANTS.items():
                sub = dfs[name]
                data = sub[(sub["metric"] == "lmd") & (sub["region"] == "mouth")].sort_values("step")
                if len(data) == 0:
                    continue
                vals = data["value"].values
                delta = np.zeros(len(vals))
                delta[1:] = vals[:-1] - vals[1:]
                ax_uf.plot(data["step"].values, delta,
                           color=cfg["color"], linestyle=cfg["linestyle"],
                           linewidth=cfg["linewidth"], label=cfg["label"],
                           marker=".", markersize=2)
            ax_uf.axhline(y=0, color="gray", linewidth=0.5)
            ax_uf.set_ylabel("Δ LMD (improvement)")
            ax_uf.set_title("Per-Step Δ LMD")

        ax_uf.set_xlabel("ODE Step")
        ax_uf.legend(fontsize=9)
        ax_uf.grid(True, alpha=0.3)
        if use_log and metric_name != "lmd":
            ax_uf.set_yscale("log")

    plt.tight_layout()
    path1 = os.path.join(args.output_dir, "trajectory_cfg_vs_nocfg_reference.png")
    plt.savefig(path1, dpi=150)
    plt.close()
    print(f"Saved {path1}")

    # ── No-reference metrics: same layout as combined plot ──
    noref_metrics = [
        ("sharpness", "Mouth Sharpness (Laplacian var)"),
        ("sync_d", "Sync-D (lower=better)"),
        ("sync_c", "Sync-C (higher=better)"),
    ]

    fig2, axes2 = plt.subplots(1, 3, figsize=(21, 6))
    fig2.suptitle("50-Step Trajectory: CFG=4.5 vs noCFG — No-Reference Metrics",
                  fontsize=14, fontweight="bold")

    for col, (metric_name, title) in enumerate(noref_metrics):
        ax = axes2[col]

        for name, cfg in VARIANTS.items():
            sub = dfs[name]
            data = sub[(sub["metric"] == metric_name) & (sub["region"] == "mouth")].sort_values("step")
            if len(data) == 0:
                continue
            ax.plot(data["step"], data["value"],
                    color=cfg["color"], linestyle=cfg["linestyle"],
                    linewidth=cfg["linewidth"], label=cfg["label"],
                    marker=".", markersize=2)

        gt_key = (metric_name, "mouth")
        if gt_key in gt_baselines:
            gt_val = gt_baselines[gt_key]
            ax.axhline(y=gt_val, color="green", linestyle="--",
                       linewidth=2, label=f"GT ({gt_val:.2f})")

        ax.set_xlabel("ODE Step")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        tick_pos = [i for i in [0, 10, 20, 30, 40, 49] if i < len(t_values)]
        ax2.set_xticks(tick_pos)
        ax2.set_xticklabels([f"t={t_values[i]:.2f}" for i in tick_pos])
        ax2.set_xlabel("Timestep t", fontsize=8)

    plt.tight_layout()
    path2 = os.path.join(args.output_dir, "trajectory_cfg_vs_nocfg_noref.png")
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"Saved {path2}")


if __name__ == "__main__":
    main()
