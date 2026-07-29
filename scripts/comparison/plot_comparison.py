"""Cross-model comparison plots: one figure per (pair, metric, region).

Reads every `comparison_<pair>.csv` produced by `compare_models.py` (schema:
`pair,model,experiment_id,step,sample,metric,region,value,value_norm,comparability`)
and, for each pair, plots one line per model (mean over samples, per step) for every
`(metric, region)` group present for BOTH models (inner-join semantics — a region only
one model reports, e.g. OmniAvatar's `upper_face`, drops out naturally).

Y-axis selection per group:
  1. `comparability == "face_value"` -> raw `value`, annotated "(face value)" (these
     metrics, e.g. Sync-C/Sync-D, are already on a comparable absolute scale).
  2. otherwise `value_norm`, annotated "(normalized)".
  3. EXCEPT: if `value_norm` is entirely NaN for the group (the no-CFG baseline pair's
     GT-less metrics -- pixel_mse/ssim/lpips/lmd -- are the normalization baseline
     themselves, so self-normalizing would be a constant 1.0 and was never computed),
     fall back to raw `value`, annotated "(raw — baseline pair)" instead of emitting
     a blank figure.

`comparison_long.csv` (the concatenation of all pairs) is skipped -- only per-pair files
are plotted.

Usage:
    python scripts/comparison/plot_comparison.py \\
        --comparison_dir results/comparison --out results/comparison/figures
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MODEL_COLORS = {"omniavatar": "tab:blue", "infinitetalk": "tab:orange"}


def _common_metric_regions(df):
    """(metric, region) pairs present for every model in df, sorted."""
    per_model = [set(zip(g["metric"], g["region"])) for _, g in df.groupby("model")]
    if len(per_model) < 2:
        return []
    return sorted(set.intersection(*per_model))


def _model_color(model, fallback_cycle):
    if model in MODEL_COLORS:
        return MODEL_COLORS[model]
    return next(fallback_cycle)


def plot_pair(df, pair_name, out_dir):
    """Write one PNG per (metric, region) group; return list of (metric, region, annotation)."""
    written = []
    models = sorted(df["model"].unique())
    for metric, region in _common_metric_regions(df):
        group = df[(df["metric"] == metric) & (df["region"] == region)]
        comparability = group["comparability"].iloc[0]

        if comparability == "face_value":
            y_col, annotation = "value", "(face value)"
        elif group["value_norm"].isna().all():
            y_col, annotation = "value", "(raw — baseline pair)"
        else:
            y_col, annotation = "value_norm", "(normalized)"

        fig, ax = plt.subplots(figsize=(8, 5))
        fallback_cycle = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        for model in models:
            per_step = (
                group[group["model"] == model]
                .groupby("step")[y_col]
                .mean()
                .sort_index()
            )
            ax.plot(per_step.index, per_step.values, marker=".",
                    label=model, color=_model_color(model, fallback_cycle))

        ax.set_xlabel("step")
        ax.set_ylabel(f"{metric} {annotation}")
        ax.set_title(f"{pair_name} — {metric} ({region}) {annotation}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out_path = os.path.join(out_dir, f"{pair_name}_{metric}_{region}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        written.append((metric, region, annotation))
    return written


def main():
    ap = argparse.ArgumentParser(description="Plot per-(pair,metric,region) cross-model comparisons.")
    ap.add_argument("--comparison_dir", default="results/comparison")
    ap.add_argument("--out", default="results/comparison/figures")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    csv_paths = sorted(glob.glob(os.path.join(args.comparison_dir, "comparison_*.csv")))
    csv_paths = [p for p in csv_paths if os.path.basename(p) != "comparison_long.csv"]

    for path in csv_paths:
        pair_name = os.path.basename(path)[len("comparison_"):-len(".csv")]
        df = pd.read_csv(path)
        written = plot_pair(df, pair_name, args.out)
        print(f"[plot] {pair_name}: {len(written)} figures")
        for metric, region, annotation in written:
            print(f"  {pair_name}_{metric}_{region}.png {annotation}")


if __name__ == "__main__":
    main()
