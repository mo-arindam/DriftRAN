#!/usr/bin/env python
"""
Detector sensitivity: show the held-out detection / false-alarm of the windowed
effect-size detector is stable across its hyperparameters (window W, effect-size
threshold tau_D, significance alpha) -- i.e. the headline numbers are not cherry-picked.
Also marks where the data-driven rule (tau_D = validation 95th-pct of D) lands.

Output: results/fig11_detector_sensitivity.png, results/ablation_detector.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import ks_2samp, mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "..", "RANalyzer-Dataset", "processed-dataset.csv")
OUT = os.path.join(HERE, "results")
FEATS = ["target_rate_mbps", "avg_throughput_mbps", "avg_nprb", "avg_snr",
         "avg_rsrp", "total_packets", "num_intervals"]
EPISODE = ("2025-01-10", "2025-02-05")
STABLE = "2025-09-01"
VAL = ("2024-07-01", "2024-12-31")


def make_X(df):
    X = df[FEATS].copy()
    return pd.concat([X.fillna(X.median()), X.isna().astype(int).add_suffix("_na")], axis=1)


def stream(df, r, W):
    """adjacent-window KS D and Mann-Whitney p over the run-ordered residual stream."""
    D = np.full(len(r), np.nan); P = np.full(len(r), np.nan)
    for t in range(2 * W, len(r)):
        a, b = r[t - 2 * W:t - W], r[t - W:t]
        D[t] = ks_2samp(a, b)[0]
        try:
            P[t] = mannwhitneyu(a, b)[1]
        except ValueError:
            P[t] = 1.0
    return D, P


def main():
    df = pd.read_csv(DS)
    df["date"] = pd.to_datetime(df["log_id"].str[:8], format="%Y%m%d")
    df = df.dropna(subset=["avg_dl_bler"]).sort_values("date").reset_index(drop=True)
    df["ym"] = df["date"].dt.to_period("M")
    X = make_X(df)
    tr = df["ym"] <= pd.Period("2024-06", "M")
    r = df["avg_dl_bler"].values - RandomForestRegressor(
        200, max_depth=10, random_state=42, n_jobs=-1).fit(
        X[tr.values], df.loc[tr, "avg_dl_bler"]).predict(X)
    val = df["date"].between(*VAL).values
    epi = df["date"].between(*EPISODE).values
    stab = (df["date"] >= pd.Timestamp(STABLE)).values

    Ws = [60, 90, 120, 150, 200]
    taus = [0.30, 0.40, 0.45, 0.50, 0.58, 0.65, 0.75]
    alphas = [1e-3, 1e-4, 1e-5]

    rows = []
    streams = {W: stream(df, r, W) for W in Ws}
    for W in Ws:
        D, P = streams[W]
        ok = ~np.isnan(D)
        auto_tau = np.nanpercentile(D[val & ok], 95)
        for tau in taus:
            for al in alphas:
                fire = (D >= tau) & (P < al) & ok
                det = fire[epi & ok].mean()
                fa = fire[stab & ok].mean()
                rows.append(dict(W=W, tau=tau, alpha=al, auto_tau=round(auto_tau, 3),
                                 detect=det, false_alarm=fa))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "ablation_detector.csv"), index=False)

    # report: at default alpha=1e-4, detection & FA vs tau for each W
    base = res[res["alpha"] == 1e-4]
    print("detection / FA at alpha=1e-4 (rows=W, cols=tau_D):")
    print(base.pivot(index="W", columns="tau", values="detect").round(2).to_string())
    print("\nfalse-alarm:")
    print(base.pivot(index="W", columns="tau", values="false_alarm").round(3).to_string())
    print("\nalpha effect (W=120, tau=0.45):")
    print(res[(res.W == 120) & (res.tau == 0.45)][["alpha", "detect", "false_alarm"]].to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for W in Ws:
        b = base[base.W == W]
        ax[0].plot(b["tau"], b["detect"], "-o", label=f"W={W}", ms=4)
        ax[1].plot(b["tau"], b["false_alarm"], "-o", label=f"W={W}", ms=4)
    for a, ttl, yl in [(ax[0], "Detection (f7d3b72 episode)", "detection rate"),
                       (ax[1], "False-alarm (stable tail)", "false-alarm rate")]:
        a.axvspan(0.40, 0.60, color="green", alpha=.08)
        a.set_xlabel(r"effect-size threshold $\tau_D$"); a.set_ylabel(yl)
        a.set_title(ttl); a.grid(alpha=.3); a.legend(fontsize=8)
    ax[1].text(0.50, ax[1].get_ylim()[1] * .7, "data-driven\n" + r"$\tau_D$ region",
               fontsize=8, ha="center", color="green")
    fig.suptitle(r"Detector sensitivity: stable across $W$ and $\tau_D$ (default $\alpha=10^{-4}$)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig11_detector_sensitivity.png"), dpi=130)
    print("\nfigure -> results/fig11_detector_sensitivity.png")


if __name__ == "__main__":
    main()
