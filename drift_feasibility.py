#!/usr/bin/env python
"""
Feasibility study: Continual learning / concept-drift attribution on the RANalyzer
dataset (8.6k OTA tests, 65 OAI releases, Nov-2023 -> Dec-2025).

Goal = de-risk the proposed paper by answering three questions on real data:
  (E1) Does a frozen environment/load -> KPI model decay when tested on newer releases?
  (E2) How much of that "decay" is just covariate shift (target-rate workload drifting
       upward over time) vs genuine code-induced concept drift in P(y|X)?
  (E3) Does a simple residual drift detector fire on the documented 2025Q1 regression
       episode (BLER + RACH-failure spike)?

Target KPI = avg_dl_bler (downlink BLER): real variance + the 2025Q1 episode lives here,
unlike throughput efficiency which is near-degenerate (92.6% of runs in [95,101]%).

Outputs: figures + a printed summary table under results/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy.stats import ks_2samp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "RANalyzer-Dataset", "processed-dataset.csv")
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

TARGET = "avg_dl_bler"
# environment + load predictors (channel/resource/load). NOT other BLER columns.
FEATS = ["target_rate_mbps", "avg_throughput_mbps", "avg_nprb", "avg_snr",
         "avg_rsrp", "total_packets", "num_intervals"]

RNG = 42


def load():
    df = pd.read_csv(DATA)
    df["date"] = pd.to_datetime(df["log_id"].str[:8], format="%Y%m%d")
    df = df.dropna(subset=[TARGET]).sort_values("date").reset_index(drop=True)
    df["ym"] = df["date"].dt.to_period("M")
    return df


def make_X(df):
    """Impute missing (schema drift) with train-agnostic median + add missingness flags."""
    X = df[FEATS].copy()
    flags = X.isna().astype(int).add_suffix("_isna")
    X = X.fillna(X.median(numeric_only=True))
    return pd.concat([X, flags], axis=1)


def fit_model(Xtr, ytr, weights=None):
    m = RandomForestRegressor(n_estimators=200, max_depth=10,
                              random_state=RNG, n_jobs=-1)
    m.fit(Xtr, ytr, sample_weight=weights)
    return m


# --------------------------------------------------------------------------- #
# E1 + E2: per-release decay, with and without covariate-shift handling
# --------------------------------------------------------------------------- #
def experiment_decay(df):
    # Reference training window = first 8 months (2023-11 .. 2024-06).
    cut = pd.Period("2024-06", "M")
    tr = df[df["ym"] <= cut]
    print(f"[E1] train window: {tr['ym'].min()}..{tr['ym'].max()}  n={len(tr)}")
    Xall = make_X(df)
    Xtr, ytr = Xall.loc[tr.index], tr[TARGET].values

    model = fit_model(Xtr, ytr)

    # --- covariate-shift-corrected model: importance weight training by
    #     w(x)=p_test(x)/p_train(x) estimated with a train-vs-test classifier on
    #     the load/channel covariates (the things that drift environmentally).
    te_future = df[df["ym"] > cut]
    cov = ["target_rate_mbps", "avg_throughput_mbps", "avg_rsrp"]
    Z = pd.concat([df.loc[tr.index, cov], df.loc[te_future.index, cov]])
    Z = Z.fillna(Z.median())
    lbl = np.r_[np.zeros(len(tr)), np.ones(len(te_future))]
    sc = StandardScaler().fit(Z)
    clf = LogisticRegression(max_iter=1000).fit(sc.transform(Z), lbl)
    p_te = clf.predict_proba(sc.transform(sc.inverse_transform(sc.transform(Z[:len(tr)]))))[:, 1]
    p_te = clf.predict_proba(sc.transform(df.loc[tr.index, cov].fillna(Z.median())))[:, 1]
    w = np.clip(p_te / (1 - p_te + 1e-6), 0.1, 10.0)  # density ratio, clipped
    model_iw = fit_model(Xtr, ytr, weights=w)

    # --- adaptive (sliding-window retrain) baseline: motivates continual learning.
    months = sorted(df["ym"].unique())
    rows = []
    for mth in months:
        if mth <= cut:
            continue
        te = df[df["ym"] == mth]
        if len(te) < 15:
            continue
        Xte, yte = Xall.loc[te.index], te[TARGET].values

        mae_static = mean_absolute_error(yte, model.predict(Xte))
        mae_iw = mean_absolute_error(yte, model_iw.predict(Xte))

        # matched-support eval: restrict to target-rate band present in train (<=30 Mbps)
        band = te["target_rate_mbps"] <= 30
        mae_matched = (mean_absolute_error(yte[band.values], model.predict(Xte[band.values]))
                       if band.sum() >= 10 else np.nan)

        # adaptive: retrain on trailing 6 months
        lo = mth - 6
        win = df[(df["ym"] < mth) & (df["ym"] >= lo)]
        mae_adapt = np.nan
        if len(win) >= 50:
            ma = fit_model(Xall.loc[win.index], win[TARGET].values)
            mae_adapt = mean_absolute_error(yte, ma.predict(Xte))

        rows.append(dict(ym=str(mth), n=len(te),
                         mean_rate=te["target_rate_mbps"].mean(),
                         mae_static=mae_static, mae_iw=mae_iw,
                         mae_matched=mae_matched, mae_adapt=mae_adapt))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "decay_table.csv"), index=False)

    # plot
    fig, ax = plt.subplots(figsize=(11, 5))
    x = range(len(res))
    ax.plot(x, res["mae_static"], "-o", label="Static model (frozen on 2023-11..2024-06)")
    ax.plot(x, res["mae_iw"], "-s", label="Covariate-shift corrected (importance-weighted)")
    ax.plot(x, res["mae_matched"], "-^", label="Matched support (target-rate <=30 Mbps)")
    ax.plot(x, res["mae_adapt"], "-d", label="Adaptive (6-month sliding retrain)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(res["ym"], rotation=90, fontsize=7)
    ax.set_ylabel(f"Test MAE on {TARGET}")
    ax.set_title("E1/E2: per-release model decay and how much is covariate shift")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_decay.png"), dpi=130)
    print("\n[E1/E2] decay table:\n", res.round(4).to_string(index=False))
    return res, model


# --------------------------------------------------------------------------- #
# E3: residual drift detector vs the 2025Q1 episode
# --------------------------------------------------------------------------- #
def experiment_drift_detector(df, model):
    Xall = make_X(df)
    resid = df[TARGET].values - model.predict(Xall)
    df = df.assign(resid=resid)

    # Page-Hinkley test on the residual stream (ordered by time).
    delta, lam = 0.005, 0.15
    mean = 0.0
    mT = 0.0
    PH = []
    flags = []
    for i, r in enumerate(np.abs(resid)):
        mean = (mean * i + r) / (i + 1)
        mT += r - mean - delta
        PH.append(mT)
        flags.append(mT > lam)
    df = df.assign(ph=PH, ph_flag=flags)

    first_flag = df[df["ph_flag"]]["date"].min()
    print(f"\n[E3] Page-Hinkley first sustained alarm: {first_flag}")

    # monthly KS test of residuals vs the reference window
    ref = df[df["ym"] <= pd.Period("2024-06", "M")]["resid"].values
    ks_rows = []
    for mth, g in df.groupby("ym"):
        if len(g) < 15:
            continue
        st, p = ks_2samp(ref, g["resid"].values)
        ks_rows.append(dict(ym=str(mth), ks=st, p=p, drift=p < 0.01,
                            mean_bler=g[TARGET].mean(),
                            ra_fail=g["failed_msg2_ra_window"].mean()))
    ks = pd.DataFrame(ks_rows)
    ks.to_csv(os.path.join(OUT, "drift_table.csv"), index=False)

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    monthly = df.groupby("ym").agg(bler=(TARGET, "mean"),
                                   ra=("failed_msg2_ra_window", "mean")).reset_index()
    xm = range(len(monthly))
    ax[0].plot(xm, monthly["bler"], "-o", color="C3", label="mean DL BLER")
    ax0b = ax[0].twinx()
    ax0b.plot(xm, monthly["ra"], "-s", color="C0", alpha=.6, label="mean failed_msg2_ra_window")
    ax[0].set_ylabel("DL BLER", color="C3")
    ax0b.set_ylabel("RA msg2 failures", color="C0")
    ax[0].set_title("E3: real concept-drift episode (2025Q1) in KPI space")
    flagged = ks[ks["drift"]]["ym"].tolist()
    for j, m in enumerate(monthly["ym"].astype(str)):
        if m in flagged:
            ax[0].axvspan(j - .4, j + .4, color="orange", alpha=.18)
    ax[1].plot(xm, [df[df["ym"] == m]["ph"].iloc[-1] if (df["ym"] == m).any() else np.nan
                    for m in monthly["ym"]], "-d", color="k")
    ax[1].axhline(lam, ls="--", color="r", label="PH threshold")
    ax[1].set_ylabel("Page-Hinkley stat")
    ax[1].set_xticks(list(xm))
    ax[1].set_xticklabels(monthly["ym"].astype(str), rotation=90, fontsize=7)
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_drift_detection.png"), dpi=130)
    print("\n[E3] monthly KS drift table:\n", ks.round(4).to_string(index=False))
    return ks


def main():
    df = load()
    print(f"loaded {len(df)} runs, {df['git_hash'].nunique()} releases, "
          f"{df['date'].min().date()}..{df['date'].max().date()}")
    res, model = experiment_decay(df)
    ks = experiment_drift_detector(df, model)

    # headline numbers
    print("\n================ FEASIBILITY VERDICT ================")
    early = res[res["ym"] <= "2024-12"]["mae_static"].mean()
    late = res[res["ym"] >= "2025-01"]["mae_static"].mean()
    late_iw = res[res["ym"] >= "2025-01"]["mae_iw"].mean()
    late_match = res[res["ym"] >= "2025-01"]["mae_matched"].mean()
    print(f"static MAE  2024(H2): {early:.4f}  -> 2025: {late:.4f}  "
          f"(decay x{late/early:.2f})")
    print(f"2025 MAE  static={late:.4f}  importance-weighted={late_iw:.4f}  "
          f"matched-support={late_match:.4f}")
    print(f"drift months flagged (KS p<0.01): "
          f"{ks[ks['drift']]['ym'].tolist()}")
    print("figures -> results/fig1_decay.png, results/fig2_drift_detection.png")


if __name__ == "__main__":
    main()
