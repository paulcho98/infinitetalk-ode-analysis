"""Tests for the cross-model comparison join (scripts/comparison/compare_models.py)."""

import pandas as pd

from scripts.comparison.compare_models import load_registry, build_comparison


def _write_csv(path, model_tag):
    rows = ["step,t,sample,metric,region,value"]
    for sample in ("s1", "s2"):
        rows.append(f"-1,gt,{sample},mse,mouth,2.0")        # GT row (real CSVs put "gt" in t)
        rows.append(f"-1,gt,{sample},sync_c,mouth,8.0")
        for step, v in ((0, 4.0), (49, 1.0)):
            rows.append(f"{step},0.5,{sample},mse,mouth,{v}")
            rows.append(f"{step},0.5,{sample},sync_c,mouth,{v + (1 if model_tag == 'b' else 0)}")
    path.write_text("\n".join(rows))


def test_build_comparison_joins_and_normalizes(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _write_csv(a, "a"); _write_csv(b, "b")
    registry = {
        "metric_rules": {"sync_c": "face_value", "mse": "normalized_only"},
        "experiments": [
            {"id": "ea", "model": "omniavatar", "csv": str(a)},
            {"id": "eb", "model": "infinitetalk", "csv": str(b)},
        ],
        "comparisons": [{"name": "p", "omniavatar": "ea", "infinitetalk": "eb"}],
    }
    df = build_comparison(registry, "p")
    assert set(df["model"]) == {"omniavatar", "infinitetalk"}
    assert len(df) == 16                       # 2 models x 2 samples x 2 steps x 2 metrics (GT rows consumed, not emitted)
    mse0 = df[(df.model == "omniavatar") & (df.step == 0) & (df.metric == "mse")]
    assert (mse0["value_norm"] == 2.0).all()   # 4.0 / GT 2.0
    assert (df[df.metric == "sync_c"]["comparability"] == "face_value").all()
    assert (df[df.metric == "mse"]["comparability"] == "normalized_only").all()
    assert list(df.columns) == ["pair", "model", "experiment_id", "step", "sample",
                                "metric", "region", "value", "value_norm", "comparability"]
    assert (df["pair"] == "p").all()


def test_missing_csv_is_skipped(tmp_path, capsys):
    registry = {
        "metric_rules": {}, "comparisons": [{"name": "p", "omniavatar": "ea", "infinitetalk": "eb"}],
        "experiments": [
            {"id": "ea", "model": "omniavatar", "csv": str(tmp_path / "nope.csv"), "status": "missing_sweep_machine"},
            {"id": "eb", "model": "infinitetalk", "csv": str(tmp_path / "also_nope.csv")},
        ],
    }
    assert build_comparison(registry, "p") is None
    assert "skip" in capsys.readouterr().out.lower()


def test_unknown_rules_pass_through_and_missing_gt_is_nan(tmp_path):
    """Real registry uses rule values beyond the fixture's two: anything that is not
    `face_value` flows through verbatim (e.g. diagnostic_only); unlisted metrics default
    to normalized_only. Metrics without a GT row (e.g. pixel_mse) get value_norm = NaN."""
    csv = tmp_path / "a.csv"
    csv.write_text("\n".join([
        "step,t,sample,metric,region,value",
        "-1,gt,s1,sharpness,mouth,10.0",
        "0,0.5,s1,sharpness,mouth,5.0",
        "0,0.5,s1,pixel_mse,upper_face,0.25",   # no GT row for this metric
        "0,0.5,s1,cfg_diff_raw,full,3.0",
    ]))
    registry = {
        "metric_rules": {"cfg_diff_raw": "diagnostic_only"},
        "experiments": [
            {"id": "ea", "model": "omniavatar", "csv": str(csv)},
            {"id": "eb", "model": "infinitetalk", "csv": str(csv)},
        ],
        "comparisons": [{"name": "p", "omniavatar": "ea", "infinitetalk": "eb"}],
    }
    df = build_comparison(registry, "p")
    assert (df[df.metric == "cfg_diff_raw"]["comparability"] == "diagnostic_only").all()
    assert (df[df.metric == "pixel_mse"]["comparability"] == "normalized_only").all()
    assert df[df.metric == "pixel_mse"]["value_norm"].isna().all()
    assert (df[df.metric == "sharpness"]["value_norm"] == 0.5).all()
    # regions are not inner-joined away: upper_face survives even though only one model has it
    assert set(df["region"]) == {"mouth", "upper_face", "full"}


def test_load_registry_reads_yaml(tmp_path):
    p = tmp_path / "reg.yaml"
    p.write_text("schema_version: 1\nmetric_rules: {sync_c: face_value}\n")
    reg = load_registry(str(p))
    assert reg["metric_rules"]["sync_c"] == "face_value"
