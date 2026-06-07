#!/usr/bin/env python
"""
Negative controls for reviewer-facing attribution checks.

1. Detector placebo: keep the residual stream and validation-frozen threshold fixed,
   but score random stable-tail windows with the same length as the f7d3b72 episode.
2. Commit-label placebo: keep BLER values fixed, but permute commit labels and
   recompute eta^2. The observed commit eta^2 should sit far above this null.

Outputs: results/placebo_controls.csv
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu
from sklearn.ensemble import RandomForestRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "..", "RANalyzer-Dataset", "processed-dataset.csv")
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

FEATS = ["target_rate_mbps", "avg_throughput_mbps", "avg_nprb", "avg_snr",
         "avg_rsrp", "total_packets", "num_intervals"]
TARGET = "avg_dl_bler"
W, ALPHA = 120, 1e-4
P_REF = ("2023-11", "2024-06")
P_VAL = ("2024-07", "2024-12")
EPISODE = ("2025-01-10", "2025-02-05")
STABLE = "2025-09-01"


def make_X(df):
    X = df[FEATS].copy()
    return pd.concat([X.fillna(X.median()), X.isna().astype(int).add_suffix("_na")], axis=1)


def eta_squared(y, labels):
    y = np.asarray(y, float)
    labels = pd.Series(labels).reset_index(drop=True)
    overall = float(np.nanmean(y))
    denom = np.nansum((y - overall) ** 2)
    if denom == 0:
        return np.nan
    means = pd.Series(y).groupby(labels, observed=True).transform("mean").to_numpy()
    return float(np.nansum((means - overall) ** 2) / denom)


def build_detector(df):
    df = df.copy()
    df["ym"] = df["date"].dt.to_period("M")
    m_ref = (df["ym"] >= pd.Period(P_REF[0], "M")) & (df["ym"] <= pd.Period(P_REF[1], "M"))
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42,
                                  n_jobs=-1)
    X = make_X(df)
    model.fit(X[m_ref.values], df.loc[m_ref, TARGET])
    resid = df[TARGET].values - model.predict(X)

    rows = []
    for t in range(2 * W, len(resid)):
        a = resid[t - 2 * W:t - W]
        b = resid[t - W:t]
        try:
            p = mannwhitneyu(a, b, alternative="two-sided")[1]
        except ValueError:
            p = 1.0
        rows.append((df["date"].iloc[t], ks_2samp(a, b)[0], p, df["git_hash"].iloc[t]))
    det = pd.DataFrame(rows, columns=["date", "D", "p", "commit"])
    in_val = det["date"].between(P_VAL[0], pd.Period(P_VAL[1], "M").end_time)
    tau_D = float(np.nanpercentile(det.loc[in_val, "D"], 95))
    det["fire"] = (det["D"] >= tau_D) & (det["p"] < ALPHA)
    return det, tau_D


def detector_placebo(det, rng, n_placebo=2000):
    epi = det["date"].between(EPISODE[0], EPISODE[1])
    real = float(det.loc[epi, "fire"].mean())
    win_len = int(epi.sum())
    stable_idx = np.flatnonzero(det["date"].ge(pd.Timestamp(STABLE)).to_numpy())
    starts = stable_idx[stable_idx <= stable_idx[-1] - win_len + 1]
    rates = []
    for _ in range(n_placebo):
        s = int(rng.choice(starts))
        rates.append(float(det["fire"].iloc[s:s + win_len].mean()))
    rates = np.asarray(rates)
    return {
        "detector_real_episode_rate": real,
        "detector_placebo_median": float(np.median(rates)),
        "detector_placebo_p95": float(np.percentile(rates, 95)),
        "detector_placebo_max": float(np.max(rates)),
        "detector_placebo_n": n_placebo,
    }


def commit_label_placebo(df, rng, n_perm=1000):
    dd = df.dropna(subset=[TARGET, "avg_snr"]).copy().reset_index(drop=True)
    observed = eta_squared(dd[TARGET], dd["git_hash"])
    labels = dd["git_hash"].to_numpy()
    null = []
    for _ in range(n_perm):
        null.append(eta_squared(dd[TARGET], rng.permutation(labels)))
    null = np.asarray(null)
    return {
        "commit_eta_observed": float(observed),
        "commit_eta_shuffle_median": float(np.median(null)),
        "commit_eta_shuffle_p95": float(np.percentile(null, 95)),
        "commit_eta_shuffle_max": float(np.max(null)),
        "commit_eta_shuffle_n": n_perm,
    }


def main():
    rng = np.random.default_rng(20260607)
    df = pd.read_csv(DS)
    df["date"] = pd.to_datetime(df["log_id"].str[:8], format="%Y%m%d")
    df = df.dropna(subset=[TARGET]).sort_values("date").reset_index(drop=True)

    det, tau_D = build_detector(df)
    results = {"tau_D": tau_D}
    results.update(detector_placebo(det, rng))
    results.update(commit_label_placebo(df, rng))

    out = pd.DataFrame([results])
    out.to_csv(os.path.join(OUT, "placebo_controls.csv"), index=False)
    print(out.round(4).to_string(index=False))
    print("wrote results/placebo_controls.csv")


if __name__ == "__main__":
    main()
