#!/usr/bin/env python
"""
Robustness checks for the BLER transfer-gap result.

Reports whether the across-release predictability collapse persists after removing
the main f7d3b72 regression and after removing the before/regression/fix triple.

Output: results/robustness_exclusions.csv
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "..", "RANalyzer-Dataset", "processed-dataset.csv")
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

TARGET = "avg_dl_bler"
FEATS = ["target_rate_mbps", "avg_snr", "min_snr", "max_snr", "avg_rsrp", "avg_nprb"]
TRIPLE = {"f9bff3d", "f7d3b72", "10e07bc"}


def make_x(df, med=None):
    x = df[FEATS].copy()
    if med is None:
        med = x.median(numeric_only=True)
    return pd.concat([x.fillna(med), x.isna().astype(int).add_suffix("_na")], axis=1), med


def eta_share(df, by):
    d = df.dropna(subset=[TARGET]).copy()
    g = d.groupby(by, observed=True)[TARGET]
    return float(((g.transform("mean") - d[TARGET].mean()) ** 2).sum() /
                 ((d[TARGET] - d[TARGET].mean()) ** 2).sum())


def score(df, label):
    d = df.dropna(subset=[TARGET]).copy()
    x, _ = make_x(d)
    shuffled = cross_val_score(
        RandomForestRegressor(200, max_depth=10, random_state=42, n_jobs=-1),
        x,
        d[TARGET],
        cv=KFold(5, shuffle=True, random_state=0),
        scoring="r2",
    ).mean()

    train = d[d["ym"] <= pd.Period("2024-12")]
    test = d[d["ym"] >= pd.Period("2025-01")]
    xtr, med = make_x(train)
    xte, _ = make_x(test, med)
    temporal = r2_score(
        test[TARGET],
        RandomForestRegressor(200, max_depth=10, random_state=42, n_jobs=-1)
        .fit(xtr, train[TARGET])
        .predict(xte),
    )

    return {
        "scenario": label,
        "runs": len(d),
        "runs_2025": len(test),
        "r2_shuffled": shuffled,
        "r2_temporal": temporal,
        "transfer_gap": shuffled - temporal,
        "commit_eta2": eta_share(d, "git_hash"),
    }


def main():
    df = pd.read_csv(DS)
    df["date"] = pd.to_datetime(df["log_id"].str[:8], format="%Y%m%d")
    df["ym"] = df["date"].dt.to_period("M")

    rows = [
        score(df, "all_runs"),
        score(df[df["git_hash"] != "f7d3b72"], "exclude_f7d3b72"),
        score(df[~df["git_hash"].isin(TRIPLE)], "exclude_regression_fix_triple"),
    ]
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "robustness_exclusions.csv"), index=False)
    print(out.round(3).to_string(index=False))
    print("wrote results/robustness_exclusions.csv")


if __name__ == "__main__":
    main()
