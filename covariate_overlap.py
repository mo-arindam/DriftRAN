#!/usr/bin/env python
"""
Covariate-overlap diagnostics for the Stage-2 covariate-vs-concept control.

Reports:
  - domain-classifier AUC for reference-vs-test covariates
  - density-ratio effective sample size before/after clipping
  - weight quantiles
  - fraction of held-out 2025 runs inside reference-window covariate support

Outputs: results/covariate_overlap.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "..", "RANalyzer-Dataset", "processed-dataset.csv")
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

COVARS = ["target_rate_mbps", "avg_throughput_mbps", "avg_nprb", "avg_snr",
          "avg_rsrp", "total_packets", "num_intervals"]


def make_Z(df, med=None):
    Z = df[COVARS].copy()
    if med is None:
        med = Z.median(numeric_only=True)
    return Z.fillna(med), med


def ess(w):
    w = np.asarray(w, float)
    return float((w.sum() ** 2) / np.sum(w ** 2))


def main():
    df = pd.read_csv(DS)
    df["date"] = pd.to_datetime(df["log_id"].str[:8], format="%Y%m%d")
    df["ym"] = df["date"].dt.to_period("M")
    ref = df[df["ym"] <= pd.Period("2024-06", "M")].copy()
    test = df[df["ym"] >= pd.Period("2025-01", "M")].copy()

    Zref, med = make_Z(ref)
    Ztest, _ = make_Z(test, med)
    Z = pd.concat([Zref, Ztest], ignore_index=True)
    y = np.r_[np.zeros(len(Zref)), np.ones(len(Ztest))]

    Ztr, Zte, ytr, yte = train_test_split(Z, y, test_size=0.3, random_state=42,
                                          stratify=y)
    sc = StandardScaler().fit(Ztr)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
        sc.transform(Ztr), ytr)
    auc = roc_auc_score(yte, clf.predict_proba(sc.transform(Zte))[:, 1])

    # Density ratio for weighting reference samples toward the held-out test block.
    p_ref_as_test = clf.predict_proba(sc.transform(Zref))[:, 1]
    prior_ref = len(Zref) / (len(Zref) + len(Ztest))
    prior_test = len(Ztest) / (len(Zref) + len(Ztest))
    raw_w = (p_ref_as_test / np.maximum(1 - p_ref_as_test, 1e-8)) * (prior_ref / prior_test)
    clip_w = np.clip(raw_w, 0.1, 10.0)

    # Support coverage: fraction of held-out test runs whose covariates are inside
    # the reference 1st--99th percentile band for every available covariate.
    lo = Zref.quantile(0.01)
    hi = Zref.quantile(0.99)
    inside_all = ((Ztest >= lo) & (Ztest <= hi)).all(axis=1)
    inside_core = ((Ztest[["avg_snr", "avg_rsrp", "avg_nprb"]] >= lo[["avg_snr", "avg_rsrp", "avg_nprb"]]) &
                   (Ztest[["avg_snr", "avg_rsrp", "avg_nprb"]] <= hi[["avg_snr", "avg_rsrp", "avg_nprb"]])).all(axis=1)

    row = {
        "ref_runs": len(ref),
        "test_runs_2025": len(test),
        "domain_auc": auc,
        "ess_raw": ess(raw_w),
        "ess_raw_frac": ess(raw_w) / len(raw_w),
        "ess_clipped": ess(clip_w),
        "ess_clipped_frac": ess(clip_w) / len(clip_w),
        "w_raw_p50": float(np.percentile(raw_w, 50)),
        "w_raw_p95": float(np.percentile(raw_w, 95)),
        "w_raw_max": float(np.max(raw_w)),
        "w_clip_p50": float(np.percentile(clip_w, 50)),
        "w_clip_p95": float(np.percentile(clip_w, 95)),
        "w_clip_max": float(np.max(clip_w)),
        "support_all_covariates": float(inside_all.mean()),
        "support_radio_core": float(inside_core.mean()),
    }
    out = pd.DataFrame([row])
    out.to_csv(os.path.join(OUT, "covariate_overlap.csv"), index=False)
    print(out.round(4).to_string(index=False))
    print("wrote results/covariate_overlap.csv")


if __name__ == "__main__":
    main()
