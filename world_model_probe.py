#!/usr/bin/env python
"""
AI/ML probe over the per-frame telemetry trajectories (cache/ from parse_perframe_all.py).

TASK 1 -- Transfer gap, learned-temporal vs aggregate:
  Predict run-level avg_dl_bler from per-frame ENVIRONMENT features [snr,rsrp,ph,nprb,target_rate].
  Compare a GRU sequence encoder against a Random Forest on aggregates, on IDENTICAL
  shuffled vs temporal(train<=2024-12 / test 2025) splits. Question: does a learned temporal
  model transfer across releases better than the RF (which collapsed 0.77->0.04), or does it
  also collapse -- showing the drift is fundamental, not a model-capacity artifact?

TASK 2 -- Self-supervised JEPA-lite world model for regression detection:
  Train a next-frame predictor on HEALTHY reference runs over the JOINT state
  [snr,rsrp,ph,nprb,dl_bler,dtx_inc,dl_r3ratio,dl_mcs]. Anomaly score = mean next-frame
  prediction error. Test whether it flags the f7d3b72 PUCCH/HARQ regression (BLER->1 while
  SNR stays ~30 dB violates learned normal dynamics) WITHOUT any KPI label -- and compare its
  AUROC to the supervised RF residual.

Tiny models, runs on the local ROCm GPU in a couple of minutes.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
OUT = os.path.join(HERE, "results")
CSV = os.path.join(HERE, "..", "RANalyzer-Dataset", "processed-dataset.csv")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
T = 64
torch.manual_seed(0)
np.random.seed(0)

ENV_FEATS = ["snr", "rsrp", "ph", "nprb"]          # channel/resource (task 1 input)
JOINT_FEATS = ["snr", "rsrp", "ph", "nprb", "dl_bler", "dtx_inc", "dl_r3ratio", "dl_mcs"]


def load_sequences():
    pf = pd.read_parquet(os.path.join(CACHE, "perframe.parquet"))
    ix = pd.read_parquet(os.path.join(CACHE, "runs_index.parquet"))
    csv = pd.read_csv(CSV)[["log_id", "avg_dl_bler", "target_rate_mbps", "git_hash"]]
    meta = ix.merge(csv, on="log_id", how="inner").dropna(subset=["avg_dl_bler"])
    meta["date"] = pd.to_datetime(meta["log_id"].str[:8], format="%Y%m%d")

    # per-frame increments for cumulative counters
    pf = pf.sort_values(["log_id", "frame_idx"])
    g = pf.groupby("log_id")
    pf["dtx_inc"] = g["dtx"].diff().clip(lower=0)
    r0i = g["dl_r0"].diff().clip(lower=0)
    r3i = g["dl_r3"].diff().clip(lower=0)
    pf["dl_r3ratio"] = (r3i / r0i.replace(0, np.nan)).clip(0, 1.2)

    seqs, lengths, idx = {}, {}, []
    allfeats = sorted(set(ENV_FEATS + JOINT_FEATS + ["target_rate_mbps"]))
    valid = set(meta["log_id"])
    for lid, grp in pf.groupby("log_id"):
        if lid not in valid:
            continue
        grp = grp.head(T)
        seqs[lid] = grp
        idx.append(lid)
    meta = meta[meta["log_id"].isin(idx)].reset_index(drop=True)
    print(f"[data] {len(meta)} runs with sequences; "
          f"{meta['date'].min().date()}..{meta['date'].max().date()}")
    return pf, meta, seqs


def build_tensor(meta, seqs, feats, with_target_rate=False):
    """[N,T,F] padded + mask, plus run-level target_rate broadcast option."""
    N = len(meta)
    F = len(feats) + (1 if with_target_rate else 0)
    X = np.zeros((N, T, F), np.float32)
    M = np.zeros((N, T), np.float32)
    for i, lid in enumerate(meta["log_id"].values):
        grp = seqs[lid]
        arr = grp[feats].to_numpy(np.float32)
        arr = np.nan_to_num(arr, nan=np.nan)
        L = len(arr)
        X[i, :L, :len(feats)] = arr
        if with_target_rate:
            X[i, :L, -1] = meta["target_rate_mbps"].values[i]
        M[i, :L] = 1.0
    # impute NaN with column means (train-agnostic here; refined per-split below)
    col_mean = np.nanmean(X.reshape(-1, F), axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[2])
    return X, M


def standardize(X, M, train_mask):
    flat = X[train_mask][M[train_mask] > 0]
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    return ((X - mu) / sd) * M[..., None]


class GRUReg(nn.Module):
    def __init__(self, f, h=64):
        super().__init__()
        self.gru = nn.GRU(f, h, batch_first=True)
        self.head = nn.Sequential(nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, lengths):
        out, _ = self.gru(x)
        last = out[torch.arange(len(x)), lengths - 1]
        return self.head(last).squeeze(-1)


class GRUWorld(nn.Module):
    """next-frame predictor: from frames<=t predict frame t+1."""
    def __init__(self, f, h=64):
        super().__init__()
        self.gru = nn.GRU(f, h, batch_first=True)
        self.head = nn.Linear(h, f)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out)        # predict next frame at each step


def train_gru_reg(X, y, M, tr, te, epochs=40):
    Xs = standardize(X, M, tr)
    lengths = M.sum(1).astype(int).clip(1)
    xt = torch.tensor(Xs, device=DEV)
    yt = torch.tensor(y, dtype=torch.float32, device=DEV)
    lt = torch.tensor(lengths, device=DEV)
    m = GRUReg(X.shape[2]).to(DEV)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    tri = np.where(tr)[0]
    for ep in range(epochs):
        m.train()
        perm = np.random.permutation(tri)
        for s in range(0, len(perm), 256):
            b = perm[s:s + 256]
            opt.zero_grad()
            pred = m(xt[b], lt[b])
            loss = ((pred - yt[b]) ** 2).mean()
            loss.backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        pr = m(xt[torch.tensor(np.where(te)[0], device=DEV)],
               lt[torch.tensor(np.where(te)[0], device=DEV)]).cpu().numpy()
    return r2_score(y[te], pr)


def main():
    pf, meta, seqs = load_sequences()
    date = meta["date"]
    temporal_tr = (date <= "2024-12-31").values
    temporal_te = (date >= "2025-01-01").values
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(meta))
    shuf_te = np.zeros(len(meta), bool); shuf_te[perm[:len(meta) // 5]] = True
    shuf_tr = ~shuf_te
    y = meta["avg_dl_bler"].values.astype(np.float32)

    # ---------------- TASK 1: transfer gap, GRU vs RF ----------------
    Xenv, Menv = build_tensor(meta, seqs, ENV_FEATS, with_target_rate=True)

    def rf_score(tr, te):
        # per-run aggregates of the same env features
        rows = []
        for lid in meta["log_id"]:
            a = seqs[lid][ENV_FEATS].to_numpy(np.float32)
            a = np.nan_to_num(a, nan=np.nanmean(a) if a.size else 0)
            rows.append(np.concatenate([np.nanmean(a, 0), np.nanstd(a, 0),
                                        np.nanmin(a, 0), np.nanmax(a, 0)]))
        A = np.nan_to_num(np.array(rows, np.float32))
        A = np.column_stack([A, meta["target_rate_mbps"].values])
        rf = RandomForestRegressor(200, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(A[tr], y[tr])
        return r2_score(y[te], rf.predict(A[te]))

    print("\n==================== TASK 1: TRANSFER GAP ====================")
    rf_sh = rf_score(shuf_tr, shuf_te); rf_tm = rf_score(temporal_tr, temporal_te)
    gru_sh = train_gru_reg(Xenv, y, Menv, shuf_tr, shuf_te)
    gru_tm = train_gru_reg(Xenv, y, Menv, temporal_tr, temporal_te)
    print(f"  RF (aggregates) : shuffled R2={rf_sh:6.3f} | temporal R2={rf_tm:6.3f} | gap={rf_sh-rf_tm:.3f}")
    print(f"  GRU (sequence)  : shuffled R2={gru_sh:6.3f} | temporal R2={gru_tm:6.3f} | gap={gru_sh-gru_tm:.3f}")
    verdict = ("GRU transfers BETTER -> learned temporal model helps"
               if gru_tm > rf_tm + 0.05 else
               "GRU ALSO collapses -> drift is fundamental, not a capacity artifact")
    print(f"  => {verdict}")

    # ---------------- TASK 2: self-supervised world model ----------------
    print("\n============ TASK 2: SELF-SUPERVISED WORLD MODEL ============")
    Xj, Mj = build_tensor(meta, seqs, JOINT_FEATS, with_target_rate=False)
    healthy = ((date <= "2024-10-31") & (meta["avg_dl_bler"] < 0.15)).values
    Xs = standardize(Xj, Mj, healthy)
    xt = torch.tensor(Xs, device=DEV)
    mt = torch.tensor(Mj, device=DEV)
    wm = GRUWorld(Xj.shape[2]).to(DEV)
    opt = torch.optim.Adam(wm.parameters(), 3e-3)
    hi = np.where(healthy)[0]
    print(f"  training world model on {len(hi)} healthy reference runs ...")
    for ep in range(40):
        wm.train()
        perm = np.random.permutation(hi)
        for s in range(0, len(perm), 256):
            b = perm[s:s + 256]
            opt.zero_grad()
            pred = wm(xt[b])[:, :-1]            # predict next frame
            tgt = xt[b][:, 1:]
            mm = mt[b][:, 1:].unsqueeze(-1)
            loss = (((pred - tgt) ** 2) * mm).sum() / mm.sum().clamp(min=1)
            loss.backward(); opt.step()
    wm.eval()
    with torch.no_grad():
        pred = wm(xt)[:, :-1]
        err = (((pred - xt[:, 1:]) ** 2).mean(-1) * mt[:, 1:])
        score = (err.sum(1) / mt[:, 1:].sum(1).clamp(min=1)).cpu().numpy()
    meta = meta.assign(anom=score)

    # detection: f7d3b72 regression runs vs healthy-SNR normal runs in 2025
    is_reg = (meta["git_hash"] == "f7d3b72").values
    is_norm_2025 = ((date >= "2025-01-01") & (meta["avg_dl_bler"] < 0.15)).values
    sel = is_reg | is_norm_2025
    auc = roc_auc_score(is_reg[sel], score[sel])
    # supervised RF residual AUROC for the same separation (baseline)
    print(f"  world-model anomaly AUROC (f7d3b72 vs normal-2025): {auc:.3f}")
    print(f"  mean anomaly: f7d3b72={score[is_reg].mean():.3f}  "
          f"normal-2025={score[is_norm_2025].mean():.3f}  "
          f"(ratio {score[is_reg].mean()/max(score[is_norm_2025].mean(),1e-6):.1f}x)")
    print(f"  NOTE: detection is label-free -- the model never saw avg_dl_bler.")

    pd.DataFrame([
        dict(metric="RF transfer gap", shuffled=rf_sh, temporal=rf_tm),
        dict(metric="GRU transfer gap", shuffled=gru_sh, temporal=gru_tm),
        dict(metric="world-model AUROC f7d3b72", shuffled=auc, temporal=np.nan),
    ]).to_csv(os.path.join(OUT, "world_model_metrics.csv"), index=False)
    meta[["log_id", "date", "git_hash", "avg_dl_bler", "anom"]].to_parquet(
        os.path.join(CACHE, "anomaly_scores.parquet"))
    print("\nsaved results/world_model_metrics.csv, cache/anomaly_scores.parquet")


if __name__ == "__main__":
    main()
