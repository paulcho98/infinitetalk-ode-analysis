"""Plot MouthWeight ODE evaluation results returned from the eval machine.

Outputs:
  - mouthweight_combined_metrics.{png,pdf}
  - mouthweight_reference_regions.{png,pdf}
  - mouthweight_frontier.{png,pdf}
  - mouthweight_step_aggregates.csv
  - mouthweight_frontier_fixedXXX_scheduleYYY.csv

Usage:
    python scripts/plot_mouthweight_ode_results.py \
        --output_dir /home/work/.local/ode_analysis/14B_mouthweight/combined
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANALYSIS_ROOT = Path("/home/work/.local/ode_analysis/14B_mouthweight")
TRAJ_DIR = Path("/home/work/.local/ode_full_trajectories/14B_mouthweight/cfg4.5")

VARIANTS = {
    "trajectory_cfg45": {
        "path": ANALYSIS_ROOT / "perceptual_v2" / "metrics.csv",
        "label": "50-step CFG=4.5",
        "color": "black",
        "linestyle": "-",
        "linewidth": 2.5,
    },
    "trajectory_nocfg": {
        "path": ANALYSIS_ROOT / "trajectory_nocfg" / "metrics.csv",
        "label": "50-step noCFG",
        "color": "black",
        "linestyle": "--",
        "linewidth": 2.5,
    },
    "euler_cfg45_cfg45": {
        "path": ANALYSIS_ROOT / "euler_cfg45_cfg45" / "metrics.csv",
        "label": "Euler CFG->CFG",
        "color": "tab:blue",
        "linestyle": "-",
        "linewidth": 1.6,
    },
    "euler_nocfg_cfg45": {
        "path": ANALYSIS_ROOT / "euler_nocfg_cfg45" / "metrics.csv",
        "label": "Euler noCFG->CFG",
        "color": "tab:orange",
        "linestyle": "-",
        "linewidth": 1.8,
    },
    "euler_nocfg_nocfg": {
        "path": ANALYSIS_ROOT / "euler_nocfg_nocfg" / "metrics.csv",
        "label": "Euler noCFG->noCFG",
        "color": "tab:red",
        "linestyle": "--",
        "linewidth": 1.6,
    },
    "euler_cfg45_nocfg": {
        "path": ANALYSIS_ROOT / "euler_cfg45_nocfg" / "metrics.csv",
        "label": "Euler CFG->noCFG",
        "color": "tab:green",
        "linestyle": "--",
        "linewidth": 1.6,
    },
}

CFG_FRONTIER = {
    "CFG=1.0": {
        "path": ANALYSIS_ROOT / "trajectory_nocfg" / "metrics.csv",
        "cfg": 1.0,
        "color": "tab:gray",
    },
    "CFG=3.0": {
        "path": ANALYSIS_ROOT / "trajectory_cfg3.0" / "metrics.csv",
        "cfg": 3.0,
        "color": "tab:purple",
    },
    "CFG=4.5": {
        "path": ANALYSIS_ROOT / "perceptual_v2" / "metrics.csv",
        "cfg": 4.5,
        "color": "black",
    },
    "CFG=6.0": {
        "path": ANALYSIS_ROOT / "trajectory_cfg6.0" / "metrics.csv",
        "cfg": 6.0,
        "color": "tab:brown",
    },
}


def load_schedule() -> np.ndarray | None:
    schedules = sorted(TRAJ_DIR.glob("*/ode_schedule.json"))
    if not schedules:
        return None
    with schedules[0].open() as f:
        schedule = json.load(f)
    return np.array(schedule["t_list"][: schedule["num_steps"]])


def read_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["step", "value"])


def aggregate_steps(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[df["step"] >= 0]
        .groupby(["step", "metric", "region"], as_index=False)["value"]
        .mean()
        .sort_values(["metric", "region", "step"])
    )


def add_t_axis(ax: plt.Axes, t_values: np.ndarray | None) -> None:
    if t_values is None:
        return
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    tick_pos = [i for i in [0, 10, 20, 30, 40, 49] if i < len(t_values)]
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels([f"t={t_values[i]:.2f}" for i in tick_pos])
    ax2.set_xlabel("Timestep t", fontsize=8)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png}")
    print(f"Saved {pdf}")


def step_values(df: pd.DataFrame, label: str, step: int) -> dict[str, float | str | int]:
    rows = {"variant": label}
    endpoint = df[df["step"] == step]
    for metric in ["ssim", "lpips", "sync_d", "sync_c"]:
        sub = endpoint[(endpoint["metric"] == metric) & (endpoint["region"] == "mouth")]
        rows[metric] = float(sub["value"].mean()) if len(sub) else np.nan
    rows["frontier_step"] = step
    return rows


def plot_combined(dfs: dict[str, pd.DataFrame], output_dir: Path, t_values: np.ndarray | None) -> None:
    panels = [
        ("ssim", "mouth", "SSIM mouth", "higher"),
        ("lpips", "mouth", "LPIPS mouth", "lower"),
        ("sync_d", "mouth", "Sync-D", "lower"),
        ("sync_c", "mouth", "Sync-C", "higher"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("MouthWeight ODE Analysis: Stepwise Metrics", fontsize=15, fontweight="bold")

    for ax, (metric, region, title, direction) in zip(axes.flat, panels):
        for name, cfg in VARIANTS.items():
            sub = dfs[name]
            data = sub[(sub["metric"] == metric) & (sub["region"] == region)].sort_values("step")
            if data.empty:
                continue
            ax.plot(
                data["step"],
                data["value"],
                color=cfg["color"],
                linestyle=cfg["linestyle"],
                linewidth=cfg["linewidth"],
                marker=".",
                markersize=2,
                label=cfg["label"],
            )
        ax.set_title(f"{title} ({direction}=better)")
        ax.set_xlabel("ODE step")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        add_t_axis(ax, t_values)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    save_figure(fig, output_dir, "mouthweight_combined_metrics")


def plot_reference_regions(
    dfs: dict[str, pd.DataFrame], output_dir: Path, t_values: np.ndarray | None
) -> None:
    panels = [
        ("ssim", "mouth", "SSIM mouth"),
        ("ssim", "upper_face", "SSIM upper face"),
        ("lpips", "mouth", "LPIPS mouth"),
        ("lpips", "full", "LPIPS full frame"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("MouthWeight Reference Metrics by Region", fontsize=15, fontweight="bold")

    for ax, (metric, region, title) in zip(axes.flat, panels):
        for name, cfg in VARIANTS.items():
            sub = dfs[name]
            data = sub[(sub["metric"] == metric) & (sub["region"] == region)].sort_values("step")
            if data.empty:
                continue
            ax.plot(
                data["step"],
                data["value"],
                color=cfg["color"],
                linestyle=cfg["linestyle"],
                linewidth=cfg["linewidth"],
                marker=".",
                markersize=2,
                label=cfg["label"],
            )
        ax.set_title(title)
        ax.set_xlabel("ODE step")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        add_t_axis(ax, t_values)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    save_figure(fig, output_dir, "mouthweight_reference_regions")


def plot_frontier(
    endpoint_df: pd.DataFrame,
    output_dir: Path,
    fixed_step: int,
    schedule_step: int,
) -> None:
    fixed = endpoint_df[endpoint_df["group"] == "fixed_cfg"].sort_values("cfg")
    schedule = endpoint_df[endpoint_df["variant"] == "Euler noCFG->CFG"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"MouthWeight Frontier (fixed CFG step {fixed_step}, schedule step {schedule_step})",
        fontsize=14,
        fontweight="bold",
    )

    axes[0].plot(fixed["lpips"], fixed["sync_c"], color="black", linewidth=1.5, alpha=0.5)
    axes[1].plot(fixed["ssim"], fixed["sync_c"], color="black", linewidth=1.5, alpha=0.5)

    for _, row in fixed.iterrows():
        axes[0].scatter(row["lpips"], row["sync_c"], s=70, label=row["variant"])
        axes[0].annotate(row["variant"], (row["lpips"], row["sync_c"]), xytext=(5, 5), textcoords="offset points")
        axes[1].scatter(row["ssim"], row["sync_c"], s=70, label=row["variant"])
        axes[1].annotate(row["variant"], (row["ssim"], row["sync_c"]), xytext=(5, 5), textcoords="offset points")

    if not schedule.empty:
        row = schedule.iloc[0]
        axes[0].scatter(row["lpips"], row["sync_c"], s=120, marker="D", color="tab:orange", label=row["variant"])
        axes[0].annotate(row["variant"], (row["lpips"], row["sync_c"]), xytext=(5, -12), textcoords="offset points")
        axes[1].scatter(row["ssim"], row["sync_c"], s=120, marker="D", color="tab:orange", label=row["variant"])
        axes[1].annotate(row["variant"], (row["ssim"], row["sync_c"]), xytext=(5, -12), textcoords="offset points")

    axes[0].set_xlabel("LPIPS mouth (lower=better)")
    axes[0].set_ylabel("Sync-C (higher=better)")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("SSIM mouth (higher=better)")
    axes[1].set_ylabel("Sync-C (higher=better)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, output_dir, "mouthweight_frontier")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--fixed_frontier_step",
        type=int,
        default=49,
        help="ODE step used for fixed-CFG trajectory points. The paper figure uses the final 50-step output.",
    )
    parser.add_argument(
        "--schedule_frontier_step",
        type=int,
        default=30,
        help="ODE step used for the schedule/Euler point. The paper figure uses step 30 for ours.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    t_values = load_schedule()

    raw = {}
    dfs = {}
    aggregate_rows = []
    endpoint_rows = []

    for name, cfg in VARIANTS.items():
        if not cfg["path"].exists():
            raise FileNotFoundError(cfg["path"])
        raw_df = read_metrics(cfg["path"])
        agg = aggregate_steps(raw_df)
        raw[name] = raw_df
        dfs[name] = agg
        agg_out = agg.copy()
        agg_out.insert(0, "variant", name)
        aggregate_rows.append(agg_out)
        endpoint_rows.append(step_values(raw_df, cfg["label"], args.fixed_frontier_step))
        print(f"Loaded {name}: {len(raw_df)} rows")

    step_aggregates = pd.concat(aggregate_rows, ignore_index=True)
    step_path = args.output_dir / "mouthweight_step_aggregates.csv"
    step_aggregates.to_csv(step_path, index=False)
    print(f"Saved {step_path}")

    frontier_rows = []
    for label, cfg in CFG_FRONTIER.items():
        df = read_metrics(cfg["path"])
        row = step_values(df, label, args.fixed_frontier_step)
        row["group"] = "fixed_cfg"
        row["cfg"] = cfg["cfg"]
        frontier_rows.append(row)
    schedule_row = step_values(
        raw["euler_nocfg_cfg45"], "Euler noCFG->CFG", args.schedule_frontier_step
    )
    schedule_row["group"] = "schedule"
    schedule_row["cfg"] = np.nan
    frontier_rows.append(schedule_row)

    endpoint_df = pd.DataFrame(frontier_rows)
    endpoint_path = (
        args.output_dir
        / (
            "mouthweight_frontier_"
            f"fixed{args.fixed_frontier_step:03d}_schedule{args.schedule_frontier_step:03d}.csv"
        )
    )
    endpoint_df.to_csv(endpoint_path, index=False)
    print(f"Saved {endpoint_path}")

    plot_combined(dfs, args.output_dir, t_values)
    plot_reference_regions(dfs, args.output_dir, t_values)
    plot_frontier(
        endpoint_df,
        args.output_dir,
        args.fixed_frontier_step,
        args.schedule_frontier_step,
    )


if __name__ == "__main__":
    main()
