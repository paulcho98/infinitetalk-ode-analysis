"""Cross-model comparison join: OmniAvatar vs InfiniteTalk per-step metrics.

For each pair in the registry's `comparisons` block, reads both models' per-step metric
CSVs (schema `step,t,sample,metric,region,value`, ground truth at `step == -1`), divides
each value by the matching GT value (`value_norm`), tags each row with the metric's
cross-model comparability rule from `metric_rules`, and concatenates both models.

Regions are NOT inner-joined: rows a model does not have (e.g. InfiniteTalk has no
`upper_face`) simply do not appear for that model. Region filtering is the plotter's job.

Usage:
    python scripts/comparison/compare_models.py --registry configs/registry.yaml --out results/comparison/
"""

import argparse
import os

import pandas as pd
import yaml

MODEL_KEYS = ("omniavatar", "infinitetalk")
DEFAULT_RULE = "normalized_only"
OUT_COLUMNS = ["pair", "model", "experiment_id", "step", "sample", "metric",
               "region", "value", "value_norm", "comparability"]


def load_registry(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _load_one(exp, pair_name, metric_rules, root):
    """Read one experiment CSV -> tagged, GT-normalized long frame."""
    df = pd.read_csv(os.path.join(root, exp["csv"]))
    df["step"] = df["step"].astype(int)   # t may be the string "gt" on GT rows; never parsed

    gt = df[df["step"] == -1].drop_duplicates(subset=["sample", "metric", "region"])
    gt_value = gt.set_index(["sample", "metric", "region"])["value"]

    body = df[df["step"] != -1].copy()
    keys = pd.MultiIndex.from_frame(body[["sample", "metric", "region"]])
    body["value_norm"] = body["value"].values / gt_value.reindex(keys).values

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
        frames.append(_load_one(exp, pair_name, metric_rules, root))

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
