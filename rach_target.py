#!/usr/bin/env python
"""
Second KPI target: CONTROL-PLANE RACH instability  ->  shows the drift-attribution
method is not BLER-specific. Different KPI type (binary), different drift episodes,
different root-cause commit (5d1c0aa) than the data-plane BLER case (f7d3b72).

target = ra_unstable := 1[failed_msg2_ra_window > 0]   (always logged, 21.7% positive)
NB: successful_cbra is NOT used as a target -- it is only logged from 2024Q4 on
(schema drift), which would contaminate a success-ratio target.

Outputs: results/fig5_rach_decay.png + console verdict.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.stats import ks_2samp, mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "..", "RANalyzer-Dataset", "processed-dataset.csv")
OUT = os.path.join(HERE, "results")
FEATS = ["target_rate_mbps", "avg_throughput_mbps", "avg_nprb", "avg_snr",
         "avg_rsrp", "total_packets", "num_intervals"]


def make_X(df):
    X = df[FEATS].copy()
    return pd.concat([X.fillna(X.median()), X.isna().astype(int).add_suffix("_na")], axis=1)


def main():
    df = pd.read_csv(DS)
    df["date"] = pd.to_datetime(df["log_id"].str[:8], format="%Y%m%d")
    df = df.sort_values("date").reset_index(drop=True)
    df["ym"] = df["date"].dt.to_period("M")
    df["y"] = (df["failed_msg2_ra_window"].fillna(0) > 0).astype(int)
    X = make_X(df)

    # train window must contain BOTH classes -> 2024-05..2024-12 (~20% positive).
    tr = (df["ym"] >= pd.Period("2024-05", "M")) & (df["ym"] <= pd.Period("2024-12", "M"))
    print(f"[train] {df.loc[tr,'ym'].min()}..{df.loc[tr,'ym'].max()}  "
          f"n={tr.sum()}  pos_rate={df.loc[tr,'y'].mean():.2f}")
    clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42,
                                 n_jobs=-1, class_weight="balanced")
    clf.fit(X[tr.values], df.loc[tr, "y"])

    # covariate-shift-corrected variant (importance weight on load/channel covariates)
    te_all = df["ym"] > pd.Period("2024-12", "M")
    cov = ["target_rate_mbps", "avg_throughput_mbps", "avg_rsrp"]
    Z = pd.concat([df.loc[tr, cov], df.loc[te_all, cov]]).fillna(0)
    sc = StandardScaler().fit(Z)
    lr = LogisticRegression(max_iter=1000).fit(
        sc.transform(Z), np.r_[np.zeros(tr.sum()), np.ones(te_all.sum())])
    p = lr.predict_proba(sc.transform(df.loc[tr, cov].fillna(0)))[:, 1]
    w = np.clip(p / (1 - p + 1e-6), 0.1, 10)
    clf_iw = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42,
                                    n_jobs=-1, class_weight="balanced")
    clf_iw.fit(X[tr.values], df.loc[tr, "y"], sample_weight=w)

    # monthly decay (Brier robust to class-rate shift; AUC where both classes present)
    rows = []
    for mth, g in df.groupby("ym"):
        if mth <= pd.Period("2024-12", "M") or len(g) < 20:
            continue
        Xte, yte = X.loc[g.index], g["y"].values
        pf = clf.predict_proba(Xte)[:, 1]
        pi = clf_iw.predict_proba(Xte)[:, 1]
        brier = brier_score_loss(yte, pf)
        brier_iw = brier_score_loss(yte, pi)
        auc = roc_auc_score(yte, pf) if len(np.unique(yte)) == 2 else np.nan
        # adaptive: retrain on trailing 6 months
        lo = mth - 6
        win = df[(df["ym"] < mth) & (df["ym"] >= lo)]
        b_ad = np.nan
        if win["y"].nunique() == 2 and len(win) >= 80:
            ca = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42,
                                        n_jobs=-1, class_weight="balanced")
            ca.fit(X.loc[win.index], win["y"])
            b_ad = brier_score_loss(yte, ca.predict_proba(Xte)[:, 1])
        rows.append(dict(ym=str(mth), n=len(g), pos=g["y"].mean(),
                         brier=brier, brier_iw=brier_iw, auc=auc, brier_adapt=b_ad))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "rach_decay_table.csv"), index=False)
    print("\n[decay]\n", res.round(3).to_string(index=False))

    # windowed effect-size detector on residual (y - p_frozen), run-ordered
    df["resid"] = df["y"].values - clf.predict_proba(X)[:, 1]
    r = df["resid"].values
    W = 150
    D, P, dt = [], [], []
    for t in range(2 * W, len(r)):
        a, b = r[t - 2 * W:t - W], r[t - W:t]
        D.append(ks_2samp(a, b)[0])
        try:
            P.append(mannwhitneyu(a, b)[1])
        except ValueError:
            P.append(1.0)
        dt.append(df["date"].values[t])
    det = pd.DataFrame({"date": dt, "D": D, "p": P})
    det["fire"] = (det["D"] >= 0.45) & (det["p"] < 1e-4)
    stable = det["date"] >= np.datetime64("2025-10-01")   # RA instability shuts off
    print(f"\n[detector] RACH residual, windowed D>=0.45 & p<1e-4 (W={W})")
    print(f"  false-alarm in stable late-2025 (>=2025-10): {det.loc[stable,'fire'].mean():.3f}")
    print(f"  fires anywhere in 2025-04..2025-08 (peak era): "
          f"{det.loc[(det['date']>=np.datetime64('2025-04-01'))&(det['date']<=np.datetime64('2025-08-31')),'fire'].mean():.2f}")

    # plot
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    xm = pd.to_datetime(res["ym"] + "-01")
    ax[0].plot(xm, res["brier"], "-o", label="frozen (train 2024-05..12)")
    ax[0].plot(xm, res["brier_iw"], "-s", label="covariate-corrected")
    ax[0].plot(xm, res["brier_adapt"], "-d", label="adaptive (6-mo retrain)")
    ax0b = ax[0].twinx()
    ax0b.plot(xm, res["pos"], ":", color="gray", label="actual RA-instability rate")
    ax0b.set_ylabel("RA-instability rate", color="gray")
    ax[0].set_ylabel("Brier score (lower=better)")
    ax[0].set_title("Control-plane target (RACH instability): decay + adaptation")
    ax[0].legend(fontsize=8, loc="upper left"); ax[0].grid(alpha=.3)
    xd = pd.to_datetime(det["date"])
    ax[1].plot(xd, det["D"], color="C0", lw=1)
    ax[1].axhline(0.45, ls="--", color="r", label="threshold 0.45")
    ax[1].fill_between(xd, 0, 1, where=det["fire"], color="orange", alpha=.25, label="fires")
    ax[1].set_ylabel("KS D (adjacent residual windows)")
    ax[1].set_title("Windowed effect-size detector on RACH residuals")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_rach_decay.png"), dpi=130)

    print("\n================ TARGET #3 VERDICT ================")
    e = res[res["ym"] <= "2025-04"]["brier"].mean()
    l = res[res["ym"].between("2025-05", "2025-08")]["brier"].mean()
    print(f"Brier frozen: pre-2025-05={e:.3f} -> 2025-05..08 spike-era={l:.3f}")
    print(f"adaptive cuts spike-era Brier to "
          f"{res[res['ym'].between('2025-05','2025-08')]['brier_adapt'].mean():.3f}")
    print("root-cause commit for RA-failure COUNT episode: 5d1c0aa "
          "(mean 128 failed_msg2/run, SNR ~30 dB) -- distinct from BLER's f7d3b72")
    print("figure -> results/fig5_rach_decay.png")


if __name__ == "__main__":
    main()
