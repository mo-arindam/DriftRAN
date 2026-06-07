#!/usr/bin/env python
"""
Reviewer-grade evaluation: (1) a strict 3-way TEMPORAL split so the detector
threshold is tuned on a validation block and evaluated ONCE on a held-out test
block containing the f7d3b72 regression; (2) temporal moving-block bootstrap 95%
CIs on every headline number (so nothing is a point estimate).

Split (forward in time, no leakage):
  reference/train  : 2023-11 .. 2024-06   (fit env->BLER model)
  validation       : 2024-07 .. 2024-12   (set detector threshold tau_D)
  HELD-OUT test    : 2025-01 .. 2025-12   (f7d3b72 episode + recovery + stable tail)

Threshold rule (no test peeking): tau_D := 95th percentile of the adjacent-window
KS effect size D over the validation block  => validation false-alarm ~5% by
construction; we then report test detection and test FA at that frozen tau_D.

Outputs: results/holdout_metrics.csv, results/fig6_holdout_ci.png + console report.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from scipy.stats import ks_2samp, mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "..", "RANalyzer-Dataset", "processed-dataset.csv")
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(42)

FEATS = ["target_rate_mbps", "avg_throughput_mbps", "avg_nprb", "avg_snr",
         "avg_rsrp", "total_packets", "num_intervals"]
TARGET = "avg_dl_bler"
W, ALPHA = 120, 1e-4
P_REF = ("2023-11", "2024-06")
P_VAL = ("2024-07", "2024-12")
P_TEST = ("2025-01", "2025-12")
EPISODE = ("2025-01-10", "2025-02-05")   # f7d3b72 ground-truth window
STABLE = "2025-09-01"                     # recovered/stable tail start (within test)


def make_X(df):
    X = df[FEATS].copy()
    return pd.concat([X.fillna(X.median()), X.isna().astype(int).add_suffix("_na")], axis=1)


def moving_block_ci(values, stat=np.mean, block=8, B=3000, alpha=0.05):
    """Moving-block bootstrap CI for autocorrelated series."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    block = max(1, min(block, n))
    nblocks = int(np.ceil(n / block))
    starts_max = max(1, n - block + 1)
    est = []
    for _ in range(B):
        idx = []
        for _ in range(nblocks):
            s = RNG.integers(0, starts_max)
            idx.extend(range(s, s + block))
        est.append(stat(v[np.array(idx[:n])]))
    lo, hi = np.percentile(est, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(stat(v)), float(lo), float(hi))


def main():
    df = pd.read_csv(DS)
    df["date"] = pd.to_datetime(df["log_id"].str[:8], format="%Y%m%d")
    df = df.dropna(subset=[TARGET]).sort_values("date").reset_index(drop=True)
    df["ym"] = df["date"].dt.to_period("M")
    X = make_X(df)

    def mask(p):
        return (df["ym"] >= pd.Period(p[0], "M")) & (df["ym"] <= pd.Period(p[1], "M"))
    m_ref = mask(P_REF)

    # --- env model + in-distribution confounding score (CV R^2 on reference) ---
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42,
                                  n_jobs=-1).fit(X[m_ref.values], df.loc[m_ref, TARGET])
    cv_r2 = cross_val_score(
        RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        X[m_ref.values], df.loc[m_ref, TARGET], cv=5, scoring="r2").mean()
    df["resid"] = df[TARGET].values - model.predict(X)

    # --- adjacent-window KS effect-size stream over the whole timeline ---
    r = df["resid"].values
    rows = []
    for t in range(2 * W, len(r)):
        a, b = r[t - 2 * W:t - W], r[t - W:t]
        try:
            p = mannwhitneyu(a, b)[1]
        except ValueError:
            p = 1.0
        rows.append((df["date"].values[t], ks_2samp(a, b)[0], p,
                     df["git_hash"].values[t]))
    det = pd.DataFrame(rows, columns=["date", "D", "p", "commit"])
    det["date"] = pd.to_datetime(det["date"])

    in_val = det["date"].between(P_VAL[0], pd.Period(P_VAL[1], "M").end_time)
    in_test = det["date"] >= pd.Timestamp(P_TEST[0])
    # frozen threshold from validation only
    tau_D = float(np.nanpercentile(det.loc[in_val, "D"], 95))
    det["fire"] = (det["D"] >= tau_D) & (det["p"] < ALPHA)

    val_fa = det.loc[in_val, "fire"].mean()
    epi = det["date"].between(EPISODE[0], EPISODE[1])
    stab = det["date"] >= pd.Timestamp(STABLE)
    test_det = det.loc[in_test & epi, "fire"]
    test_fa = det.loc[in_test & stab, "fire"]

    # --- bootstrap CIs ---
    det_pt = moving_block_ci(test_det.values, block=8)
    fa_pt = moving_block_ci(test_fa.values, block=16)
    # decay MAE CIs (block-bootstrap residual abs-errors by period)
    ae = np.abs(df["resid"].values)
    mae_2024 = moving_block_ci(ae[mask(("2024-07", "2024-12")).values], block=20,
                               stat=np.mean)
    mae_2025 = moving_block_ci(ae[mask(("2025-01", "2025-12")).values], block=20,
                               stat=np.mean)

    rep = pd.DataFrame([
        dict(metric="env CV R^2 (reference, in-dist)", value=cv_r2, lo=np.nan, hi=np.nan),
        dict(metric="tau_D (frozen from val 95th pct)", value=tau_D, lo=np.nan, hi=np.nan),
        dict(metric="validation false-alarm", value=val_fa, lo=np.nan, hi=np.nan),
        dict(metric="TEST detection in f7d3b72 episode", value=det_pt[0], lo=det_pt[1], hi=det_pt[2]),
        dict(metric="TEST false-alarm (stable tail)", value=fa_pt[0], lo=fa_pt[1], hi=fa_pt[2]),
        dict(metric="MAE 2024-H2", value=mae_2024[0], lo=mae_2024[1], hi=mae_2024[2]),
        dict(metric="MAE 2025 (held-out)", value=mae_2025[0], lo=mae_2025[1], hi=mae_2025[2]),
    ])
    rep.to_csv(os.path.join(OUT, "holdout_metrics.csv"), index=False)
    pd.set_option("display.width", 160)
    print("\n========= HELD-OUT 3-WAY TEMPORAL SPLIT + BOOTSTRAP 95% CI =========")
    print(f"reference {P_REF}  | validation {P_VAL}  | held-out test {P_TEST}")
    print(rep.round(4).to_string(index=False))
    print(f"\ndecay ratio 2025/2024-H2 = {mae_2025[0]/mae_2024[0]:.2f}x  "
          f"(CIs: 2024-H2 [{mae_2024[1]:.3f},{mae_2024[2]:.3f}], "
          f"2025 [{mae_2025[1]:.3f},{mae_2025[2]:.3f}])")

    # --- figure: point estimates with 95% CI on the held-out test ---
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = ["Detection\n(f7d3b72 episode)", "False-alarm\n(stable tail)"]
    pts = [det_pt, fa_pt]
    xs = np.arange(2)
    ax[0].bar(xs, [p[0] for p in pts], color=["C2", "C3"], alpha=.75,
              yerr=[[p[0] - p[1] for p in pts], [p[2] - p[0] for p in pts]],
              capsize=6)
    ax[0].axhline(0.05, ls="--", color="gray", lw=1, label="5% FA target")
    ax[0].set_xticks(xs); ax[0].set_xticklabels(labels)
    ax[0].set_ylabel("rate (held-out test)"); ax[0].set_ylim(0, 1)
    ax[0].set_title(f"Detector @ frozen tau_D={tau_D:.2f}\n(tuned on validation only)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, axis="y")

    mae_lbls = ["2024-H2", "2025 (held-out)"]
    maes = [mae_2024, mae_2025]
    ax[1].bar([0, 1], [m[0] for m in maes], color="C0", alpha=.75,
              yerr=[[m[0] - m[1] for m in maes], [m[2] - m[0] for m in maes]],
              capsize=6)
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(mae_lbls)
    ax[1].set_ylabel("BLER MAE"); ax[1].set_title("Frozen-model decay (95% CI)")
    ax[1].grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_holdout_ci.png"), dpi=130)
    print("figure -> results/fig6_holdout_ci.png")


if __name__ == "__main__":
    main()
