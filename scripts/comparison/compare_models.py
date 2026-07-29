"""Cross-model comparison join: OmniAvatar vs InfiniteTalk per-step metrics.

For each pair in the registry's `comparisons` block, reads both models' per-step metric
CSVs (schema `step,t,sample,metric,region,value`, ground truth at `step == -1`), normalizes
each value against that model's own scale (`value_norm`), tags each row with the metric's
cross-model comparability rule from `metric_rules`, and concatenates both models.

Normalizer, per the design spec ("its own step=-1 GT row *or* its no-CFG baseline"):
  1. the matching GT row at `(sample, metric, region)`, when the metric has one
     (only sharpness/sync_c/sync_d do — the rest are distances *to* GT);
  2. otherwise, if the experiment declares `baseline: <experiment_id>`, that experiment's
     value at the same `(step, sample, metric, region)`.
A missing or zero normalizer yields NaN (never inf).

Regions are NOT inner-joined: rows a model does not have (e.g. InfiniteTalk has no
`upper_face`) simply do not appear for that model. Region filtering is the plotter's job.

Usage:
    python scripts/comparison/compare_models.py --registry configs/registry.yaml --out results/comparison/
"""

import argparse
import os

import numpy as np
import pandas as pd
import yaml

MODEL_KEYS = ("omniavatar", "infinitetalk")
DEFAULT_RULE = "normalized_only"
OUT_COLUMNS = ["pair", "model", "experiment_id", "step", "sample", "metric",
               "region", "value", "value_norm", "comparability"]


def load_registry(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _read_csv(path):
    df = pd.read_csv(path)
    df["step"] = df["step"].astype(int)   # t may be the string "gt" on GT rows; never parsed
    return df


def _divide_by(body, normalizer, key_cols):
    """value / normalizer looked up on key_cols; missing or zero normalizer -> NaN, never inf."""
    denom = normalizer.reindex(pd.MultiIndex.from_frame(body[key_cols])).values
    return body["value"].values / np.where(denom == 0, np.nan, denom)


def _baseline_values(exp, experiments, root):
    """Per-step normalizer from the experiment's no-CFG baseline, or None if unavailable."""
    baseline_id = exp.get("baseline")
    if baseline_id is None:
        return None
    baseline = experiments.get(baseline_id)
    path = os.path.join(root, baseline["csv"]) if baseline else None
    if baseline is None or baseline.get("status") or not os.path.exists(path):
        print(f"[warn] experiment '{exp['id']}': baseline '{baseline_id}' unavailable "
              f"-> {path}; GT-less metrics stay unnormalized")
        return None
    df = _read_csv(path)
    key_cols = ["step", "sample", "metric", "region"]
    df = df[df["step"] != -1].drop_duplicates(subset=key_cols)
    return df.set_index(key_cols)["value"]


def _load_one(exp, pair_name, metric_rules, root, experiments):
    """Read one experiment CSV -> tagged, normalized long frame."""
    df = _read_csv(os.path.join(root, exp["csv"]))

    gt = df[df["step"] == -1].drop_duplicates(subset=["sample", "metric", "region"])
    gt_value = gt.set_index(["sample", "metric", "region"])["value"]

    body = df[df["step"] != -1].copy()
    body["value_norm"] = _divide_by(body, gt_value, ["sample", "metric", "region"])

    # Fall back to the no-CFG baseline for metrics that cannot have a GT row of their own.
    baseline_value = _baseline_values(exp, experiments, root)
    if baseline_value is not None and body["value_norm"].isna().any():
        fallback = _divide_by(body, baseline_value, ["step", "sample", "metric", "region"])
        body["value_norm"] = body["value_norm"].fillna(pd.Series(fallback, index=body.index))

    body["pair"] = pair_name
    body["model"] = exp["model"]
    body["experiment_id"] = exp["id"]
    body["comparability"] = body["metric"].map(metric_rules).fillna(DEFAULT_RULE)
    return body[OUT_COLUMNS]


def build_comparison(registry, pair_name, root=""):
    """Joined long frame for one comparison pair, or None if either side is unavailable."""
    pair = {c["name"]: c for c in registry.get("comparisons", [])}[pair_name]
    experiments = {e["id"]: e for e in registry.get("experiments", [])}
    metric_rules = registry.get("metric_rules") or {}

    frames = []
    for model_key in MODEL_KEYS:
        exp = experiments[pair[model_key]]
        csv_path = os.path.join(root, exp["csv"])
        if exp.get("status") or not os.path.exists(csv_path):
            reason = exp.get("status") or "csv_not_found"
            print(f"[skip] pair '{pair_name}': experiment '{exp['id']}' unavailable ({reason}) -> {csv_path}")
            return None
        frames.append(_load_one(exp, pair_name, metric_rules, root, experiments))

    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description="Join OmniAvatar/InfiniteTalk metric CSVs per comparison pair.")
    ap.add_argument("--registry", default="configs/registry.yaml")
    ap.add_argument("--out", default="results/comparison")
    args = ap.parse_args()

    registry = load_registry(args.registry)
    # CSV paths in the registry are repo-root relative; the registry lives in <root>/configs/.
    root = os.path.dirname(os.path.dirname(os.path.abspath(args.registry)))
    os.makedirs(args.out, exist_ok=True)

    frames = []
    for comp in registry.get("comparisons", []):
        name = comp["name"]
        df = build_comparison(registry, name, root=root)
        if df is None:
            continue
        path = os.path.join(args.out, f"comparison_{name}.csv")
        df.to_csv(path, index=False)
        print(f"[write] {path}  ({len(df)} rows, models={sorted(set(df['model']))}, "
              f"regions={sorted(set(df['region']))})")
        frames.append(df)

    if not frames:
        print("[warn] no comparison pairs produced output")
        return
    long_path = os.path.join(args.out, "comparison_long.csv")
    long_df = pd.concat(frames, ignore_index=True)
    long_df.to_csv(long_path, index=False)
    print(f"[write] {long_path}  ({len(long_df)} rows, {long_df['pair'].nunique()} pairs)")


if __name__ == "__main__":
    main()
