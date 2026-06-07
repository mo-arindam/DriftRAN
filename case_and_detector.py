#!/usr/bin/env python
"""
(1) Before/after per-frame case study for the f7d3b72 PUCCH/HARQ regression vs the
    10e07bc fix  ->  results/fig3_case_f7d3b72.png
(2) A proper windowed effect-size drift detector to replace the over-powered fixed-
    reference KS p-value test  ->  results/fig4_detector.png + console false-alarm report.
"""
import os, re, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import ks_2samp, mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "..", "RANalyzer-Dataset")
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

PERFRAME = re.compile(
    r"dlsch_rounds (\d+)/(\d+)/(\d+)/(\d+), dlsch_errors (\d+), "
    r"pucch0_DTX (\d+), BLER ([\d.]+)")


def parse_perframe(logpath):
    rows = []
    with open(logpath, errors="ignore") as fh:
        for ln in fh:
            m = PERFRAME.search(ln)
            if m:
                r0, r1, r2, r3, err, dtx, bler = m.groups()
                rows.append((int(r0), int(r1), int(r2), int(r3),
                             int(err), int(dtx), float(bler)))
    df = pd.DataFrame(rows, columns=["r0", "r1", "r2", "r3", "err", "dtx", "bler"])
    # cumulative -> per-snapshot increments for an instantaneous view
    inc = df[["r0", "r3", "dtx"]].diff().clip(lower=0)
    df["dtx_inc"] = inc["dtx"]
    df["round3_ratio"] = (inc["r3"] / inc["r0"].replace(0, np.nan)).clip(0, 1.2)
    return df


def find_log(commit, n=1):
    out = []
    for f in glob.glob(os.path.join(DS, "cicd-dataset", "2025*", "*", "nr-gnb.log")):
        try:
            with open(f, errors="ignore") as fh:
                head = "".join([next(fh) for _ in range(6)])
        except StopIteration:
            head = ""
        if f"Hash: {commit}" in head:
            out.append(f)
            if len(out) >= n:
                break
    return out


def case_study():
    bad = os.path.join(DS, "cicd-dataset", "20250122", "060303", "nr-gnb.log")
    good = find_log("10e07bc", 1)
    good = good[0] if good else None
    db = parse_perframe(bad)
    dg = parse_perframe(good) if good else pd.DataFrame()
    print(f"[case] regression log {os.path.relpath(bad, DS)}: {len(db)} frames")
    print(f"[case] fix log        {os.path.relpath(good, DS) if good else None}: {len(dg)} frames")

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].plot(db["bler"].values, color="C3", label="f7d3b72 (regression)")
    if len(dg):
        ax[0].plot(dg["bler"].values, color="C2", label="10e07bc (fix)")
    ax[0].set_title("Instantaneous DL BLER"); ax[0].set_xlabel("per-frame snapshot")
    ax[0].set_ylabel("BLER"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

    ax[1].plot(db["dtx_inc"].values, color="C3")
    if len(dg):
        ax[1].plot(dg["dtx_inc"].values, color="C2")
    ax[1].set_title("PUCCH DTX increment / snapshot\n(lost HARQ-ACK feedback)")
    ax[1].set_xlabel("per-frame snapshot"); ax[1].set_yscale("symlog"); ax[1].grid(alpha=.3)

    ax[2].plot(db["round3_ratio"].values, color="C3")
    if len(dg):
        ax[2].plot(dg["round3_ratio"].values, color="C2")
    ax[2].set_title("HARQ round-3 retention\n(1.0 = retx never succeeds)")
    ax[2].set_xlabel("per-frame snapshot"); ax[2].set_ylim(-.05, 1.2); ax[2].grid(alpha=.3)
    fig.suptitle("Case study: PUCCH/HARQ-feedback regression f7d3b72 -> fix 10e07bc "
                 "(SNR ~30 dB in both)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_case_f7d3b72.png"), dpi=130)
    print(f"[case] BLER mean  regression={db['bler'].mean():.3f}  "
          f"fix={dg['bler'].mean():.3f}")


# --------------------------------------------------------------------------- #
def windowed_detector():
    df = pd.read_csv(os.path.join(DS, "processed-dataset.csv"))
    df["date"] = pd.to_datetime(df["log_id"].str[:8], format="%Y%m%d")
    df = df.dropna(subset=["avg_dl_bler"]).sort_values("date").reset_index(drop=True)
    df["ym"] = df["date"].dt.to_period("M")

    feats = ["target_rate_mbps", "avg_throughput_mbps", "avg_nprb", "avg_snr",
             "avg_rsrp", "total_packets", "num_intervals"]
    X = df[feats].copy()
    X = pd.concat([X.fillna(X.median()), X.isna().astype(int).add_suffix("_na")], axis=1)
    tr = df["ym"] <= pd.Period("2024-06", "M")
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42,
                                  n_jobs=-1).fit(X[tr.values], df.loc[tr, "avg_dl_bler"])
    df["resid"] = df["avg_dl_bler"].values - model.predict(X)

    # run-ordered stream; sliding adjacent windows of W runs.
    W = 120
    r = df["resid"].values
    ks_D, mw_p, idx = [], [], []
    for t in range(2 * W, len(r)):
        ref = r[t - 2 * W:t - W]
        cur = r[t - W:t]
        D, _ = ks_2samp(ref, cur)
        try:
            _, p = mannwhitneyu(ref, cur, alternative="two-sided")
        except ValueError:
            p = 1.0
        ks_D.append(D); mw_p.append(p); idx.append(t)
    det = pd.DataFrame({"i": idx, "D": ks_D, "p": mw_p})
    det["date"] = df["date"].values[det["i"].values]
    det["commit"] = df["git_hash"].values[det["i"].values]

    # DECISION RULE: effect-size gate (D>=0.45) AND significance (p<1e-4).
    det["fire"] = (det["D"] >= 0.45) & (det["p"] < 1e-4)

    # known regression window (f7d3b72 active) and a known-healthy window (>=2025-05)
    reg = (det["date"] >= "2025-01-10") & (det["date"] <= "2025-02-05")
    healthy = det["date"] >= "2025-05-01"
    det_recall = det.loc[reg, "fire"].mean()
    fa = det.loc[healthy, "fire"].mean()
    print(f"\n[detector] windowed effect-size rule (D>=0.45 & p<1e-4), W={W}")
    print(f"  detection rate inside f7d3b72 episode window : {det_recall:.2f}")
    print(f"  false-alarm rate in healthy period (>=2025-05): {fa:.3f}")
    fire_commits = det.loc[det["fire"], "commit"].value_counts().head(8)
    print("  commits inside fired windows (top):\n",
          fire_commits.to_string())

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = pd.to_datetime(det["date"])
    ax.plot(x, det["D"], color="C0", lw=1, label="KS effect size D (adjacent windows)")
    ax.axhline(0.45, ls="--", color="r", label="effect-size threshold 0.45")
    ax.fill_between(x, 0, 1, where=det["fire"], color="orange", alpha=.25,
                    label="detector fires")
    ax.axvspan(pd.Timestamp("2025-01-15"), pd.Timestamp("2025-01-22"),
               color="red", alpha=.12, label="f7d3b72 ground truth")
    ax.set_ylabel("KS D between adjacent residual windows")
    ax.set_title("Windowed effect-size drift detector (replaces over-powered fixed-ref KS p-test)")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_detector.png"), dpi=130)
    det.to_csv(os.path.join(OUT, "detector_table.csv"), index=False)


if __name__ == "__main__":
    case_study()
    windowed_detector()
    print("\nfigures -> results/fig3_case_f7d3b72.png, results/fig4_detector.png")
