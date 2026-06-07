#!/usr/bin/env python
"""
Centerpiece evidence for the concept-drift thesis (corrects the earlier mis-read of
temporal R^2 as 'no signal'):

  (A) The env->KPI relationship is strongly predictive IN-DISTRIBUTION but does NOT
      transfer across the release timeline:
        BLER  shuffled-CV R^2 = 0.77   ->  temporal R^2 = 0.04
        RACH  shuffled-CV AUC = 0.86   ->  temporal AUC = 0.70
      i.e. P(y|x) is non-stationary  ==  concept drift.

  (B) Software, not channel, drives the KPI:
        commit identity explains ~64% of BLER variance; SNR decile ~5%; rate ~0%.

Output: results/fig7_predictability.png + results/predictability.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "..", "RANalyzer-Dataset", "processed-dataset.csv")
OUT = os.path.join(HERE, "results")
F = ["target_rate_mbps", "avg_snr", "min_snr", "max_snr", "avg_rsrp", "avg_nprb"]


def Xof(d):
    X = d[F].copy()
    return pd.concat([X.fillna(X.median()), X.isna().astype(int).add_suffix("_na")], axis=1)


def main():
    df = pd.read_csv(DS)
    df["date"] = pd.to_datetime(df["log_id"].str[:8], format="%Y%m%d")
    df["ym"] = df["date"].dt.to_period("M")
    df["ra"] = (df["failed_msg2_ra_window"].fillna(0) > 0).astype(int)

    # (A) shuffled vs temporal -- BLER
    d = df.dropna(subset=["avg_dl_bler"])
    bler_shuf = cross_val_score(
        RandomForestRegressor(200, max_depth=10, random_state=42, n_jobs=-1),
        Xof(d), d["avg_dl_bler"], cv=KFold(5, shuffle=True, random_state=0),
        scoring="r2").mean()
    tr, te = d[d.ym <= pd.Period("2024-12")], d[d.ym >= pd.Period("2025-01")]
    bler_temp = r2_score(te["avg_dl_bler"],
                         RandomForestRegressor(200, max_depth=10, random_state=42, n_jobs=-1)
                         .fit(Xof(tr), tr["avg_dl_bler"]).predict(Xof(te)))

    # (A) shuffled vs temporal -- RACH
    dr = df[df.ym >= pd.Period("2024-05")]
    ra_shuf = cross_val_score(
        RandomForestClassifier(300, max_depth=8, random_state=42, n_jobs=-1,
                               class_weight="balanced"),
        Xof(dr), dr["ra"], cv=KFold(5, shuffle=True, random_state=0),
        scoring="roc_auc").mean()
    trr, ter = dr[dr.ym <= pd.Period("2024-12")], dr[dr.ym >= pd.Period("2025-01")]
    ra_temp = roc_auc_score(ter["ra"],
                            RandomForestClassifier(300, max_depth=8, random_state=42, n_jobs=-1,
                                                   class_weight="balanced")
                            .fit(Xof(trr), trr["ra"]).predict_proba(Xof(ter))[:, 1])

    # (B) variance decomposition for BLER
    dd = df.dropna(subset=["avg_dl_bler", "avg_snr"]).copy()
    dd["snrb"] = pd.qcut(dd["avg_snr"], 10, duplicates="drop")
    dd["rateb"] = pd.qcut(dd["target_rate_mbps"], 8, duplicates="drop")

    def eta(by):
        g = dd.groupby(by, observed=True)["avg_dl_bler"]
        return ((g.transform("mean") - dd["avg_dl_bler"].mean()) ** 2).sum() / \
               ((dd["avg_dl_bler"] - dd["avg_dl_bler"].mean()) ** 2).sum()
    eta_commit, eta_snr, eta_rate = eta("git_hash"), eta("snrb"), eta("rateb")

    rep = pd.DataFrame([
        ("BLER R2 in-distribution (shuffled)", bler_shuf),
        ("BLER R2 across releases (temporal)", bler_temp),
        ("RACH AUC in-distribution (shuffled)", ra_shuf),
        ("RACH AUC across releases (temporal)", ra_temp),
        ("BLER var share: commit", eta_commit),
        ("BLER var share: SNR decile", eta_snr),
        ("BLER var share: target-rate octile", eta_rate),
    ], columns=["metric", "value"])
    rep.to_csv(os.path.join(OUT, "predictability.csv"), index=False)
    print(rep.round(3).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    g = np.arange(2)
    ax[0].bar(g - 0.18, [bler_shuf, ra_shuf], 0.36, label="in-distribution (shuffled CV)", color="C0")
    ax[0].bar(g + 0.18, [bler_temp, ra_temp], 0.36, label="across releases (temporal)", color="C3")
    ax[0].set_xticks(g); ax[0].set_xticklabels(["BLER (R$^2$)", "RACH (AUC)"])
    ax[0].axhline(0, color="k", lw=.8)
    ax[0].set_title("env$\\to$KPI predictability collapses across releases\n(= concept drift)")
    ax[0].set_ylabel("score"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, axis="y")

    shares = [eta_commit, eta_snr, eta_rate, max(0, 1 - eta_commit - eta_snr - eta_rate)]
    ax[1].bar(range(4), shares,
              color=["C3", "C0", "C2", "lightgray"])
    ax[1].set_xticks(range(4))
    ax[1].set_xticklabels(["commit\nidentity", "SNR\ndecile", "target\nrate", "residual"], fontsize=9)
    ax[1].set_ylabel("share of DL-BLER variance ($\\eta^2$)")
    ax[1].set_title("Software dominates BLER, not channel")
    ax[1].grid(alpha=.3, axis="y")
    for i, v in enumerate(shares):
        ax[1].text(i, v + .01, f"{v:.0%}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig7_predictability.png"), dpi=130)
    print("figure -> results/fig7_predictability.png")


if __name__ == "__main__":
    main()
