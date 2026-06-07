#!/usr/bin/env python
"""
Baseline comparison for the held-out regression-detection task. All methods are scored
on the SAME run-ordered stream and thresholded to an EQUAL ~5% validation false-alarm
budget (threshold = 95th pct of the score on the 2024-07..2024-12 validation block,
no test peeking). We then measure, on the held-out 2025 block:
  - detection rate inside the f7d3b72 episode window
  - false-alarm rate in the stable tail (>=2025-09)
  - detection latency (runs from episode start to first alarm)

Methods:
  naive_kpi      : score = DL BLER                     (no environment model)
  static_resid   : score = |y - f(x)|                  (RANalyzer-style static residual)
  fixedref_ks    : KS D(reference resid, trailing win) (over-powered fixed reference)
  page_hinkley   : Page-Hinkley statistic on |resid|
  windowed_ks    : KS D(adjacent resid windows)        ** ours **
  world_model    : self-supervised anomaly (from cache, label-free)

Honest note: on this stable-channel testbed a naive KPI threshold is a strong baseline
(degradations are severe BLER spikes); the residual/world-model methods match it while
being robust to covariate shift and able to flag drift a KPI threshold cannot see.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import ks_2samp

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "..", "RANalyzer-Dataset", "processed-dataset.csv")
CACHE = os.path.join(HERE, "cache")
OUT = os.path.join(HERE, "results")
W = 120
FEATS = ["target_rate_mbps", "avg_throughput_mbps", "avg_nprb", "avg_snr",
         "avg_rsrp", "total_packets", "num_intervals"]
EPISODE = ("2025-01-10", "2025-02-05")
STABLE = "2025-09-01"
VAL = ("2024-07-01", "2024-12-31")


def make_X(df):
    X = df[FEATS].copy()
    return pd.concat([X.fillna(X.median()), X.isna().astype(int).add_suffix("_na")], axis=1)


def page_hinkley_alarms(x, lam, delta=0.005):
    """Proper streaming Page-Hinkley with reset-on-alarm. Returns binary fire array."""
    fire = np.zeros(len(x), bool)
    mean = cum = run_min = 0.0
    n = 0
    for i, v in enumerate(x):
        n += 1
        mean = (mean * (n - 1) + v) / n
        cum += v - mean - delta
        run_min = min(run_min, cum)
        if cum - run_min > lam:
            fire[i] = True
            mean = cum = run_min = 0.0
            n = 0                          # reset segment after detection
    return fire


def tune_ph(x, val_mask, target=0.05):
    """Pick lambda so the validation alarm rate ~= target."""
    best, best_gap = 0.1, 1e9
    for lam in np.linspace(0.005, 0.5, 60):
        rate = page_hinkley_alarms(x, lam)[val_mask].mean()
        if abs(rate - target) < best_gap:
            best, best_gap = lam, abs(rate - target)
    return best


def main():
    df = pd.read_csv(DS)
    df["date"] = pd.to_datetime(df["log_id"].str[:8], format="%Y%m%d")
    df = df.dropna(subset=["avg_dl_bler"]).sort_values("date").reset_index(drop=True)
    df["ym"] = df["date"].dt.to_period("M")
    X = make_X(df)
    tr = df["ym"] <= pd.Period("2024-06", "M")
    f = RandomForestRegressor(200, max_depth=10, random_state=42, n_jobs=-1).fit(
        X[tr.values], df.loc[tr, "avg_dl_bler"])
    df["resid"] = df["avg_dl_bler"].values - f.predict(X)
    r = df["resid"].values

    # --- per-run scores for each method (aligned to df index) ---
    scores = {}
    scores["naive_kpi"] = df["avg_dl_bler"].values.copy()
    scores["static_resid"] = np.abs(r)
    # windowed / fixed-reference KS need a window history -> NaN until enough runs
    ref = r[tr.values]
    fixed = np.full(len(r), np.nan)
    wind = np.full(len(r), np.nan)
    for t in range(2 * W, len(r)):
        fixed[t] = ks_2samp(ref, r[t - W:t])[0]
        wind[t] = ks_2samp(r[t - 2 * W:t - W], r[t - W:t])[0]
    scores["fixedref_ks"] = fixed
    scores["windowed_ks"] = wind
    # world-model anomaly (label-free) from cache
    try:
        a = pd.read_parquet(os.path.join(CACHE, "anomaly_scores.parquet"))[["log_id", "anom"]]
        df = df.merge(a, on="log_id", how="left")
        scores["world_model"] = df["anom"].values
    except Exception as e:
        print("world_model anomaly unavailable:", e)

    val = df["date"].between(*VAL).values
    epi = df["date"].between(*EPISODE).values
    stab = (df["date"] >= pd.Timestamp(STABLE)).values

    # build a binary fire[] for every method at the equal ~5% validation budget
    fires = {}
    okmask = {}
    for name, s in scores.items():
        s = np.asarray(s, float); ok = ~np.isnan(s)
        tau = np.percentile(s[val & ok], 95)
        fires[name] = (s >= tau) & ok; okmask[name] = ok
    # Page-Hinkley as a proper streaming detector (reset), tuned to ~5% val rate
    lam = tune_ph(np.abs(r), val)
    fires["page_hinkley"] = page_hinkley_alarms(np.abs(r), lam)
    okmask["page_hinkley"] = np.ones(len(r), bool)

    rows = []
    for name in fires:
        fire, ok = fires[name], okmask[name]
        vmask = val & ok
        val_fa = fire[vmask].mean()
        ce, cs = epi & ok, stab & ok
        det = fire[ce].mean() if ce.sum() else np.nan
        fa = fire[cs].mean() if cs.sum() else np.nan
        caught = bool(fire[ce].any())
        epi_idx = np.where(ce)[0]
        fired = epi_idx[fire[epi_idx]]
        lat = int(fired[0] - epi_idx[0]) if len(fired) else np.nan
        rows.append(dict(method=name, val_fa=round(val_fa, 3), caught=caught,
                         detect=round(det, 3), false_alarm=round(fa, 3),
                         latency_runs=lat))
    rep = pd.DataFrame(rows)
    order = ["naive_kpi", "static_resid", "fixedref_ks", "page_hinkley",
             "world_model", "windowed_ks"]
    rep["o"] = rep["method"].map({m: i for i, m in enumerate(order)})
    rep = rep.sort_values("o").drop(columns="o").reset_index(drop=True)
    rep.to_csv(os.path.join(OUT, "baselines.csv"), index=False)
    pd.set_option("display.width", 160)
    print("\n==== held-out detection at equal ~5% validation FA budget ====")
    print("(detect = fraction of f7d3b72-episode runs flagged; "
          "false_alarm = fraction of stable-tail runs flagged)\n")
    print(rep.to_string(index=False))

    # figure: detection vs false-alarm
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for _, row in rep.iterrows():
        star = row["method"] == "windowed_ks"
        ax.scatter(row["false_alarm"], row["detect"], s=160 if star else 90,
                   marker="*" if star else "o",
                   color="C3" if star else "C0", zorder=3)
        ax.annotate(row["method"], (row["false_alarm"], row["detect"]),
                    textcoords="offset points", xytext=(7, 4), fontsize=8)
    ax.axvline(0.05, ls="--", color="gray", lw=1, label="5% FA budget")
    ax.set_xlabel("held-out false-alarm rate (stable tail)")
    ax.set_ylabel("detection rate (f7d3b72 episode)")
    ax.set_title("Baselines at equal validation FA budget (top-left = best)")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig10_baselines.png"), dpi=130)
    print("\nfigure -> results/fig10_baselines.png")


if __name__ == "__main__":
    main()
