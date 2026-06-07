#!/usr/bin/env python
"""
Expanded world-model study:
 (1) SIGReg-regularized JEPA variant (LeJEPA-style isotropic-Gaussian embedding reg,
     EMA-free) vs the plain next-frame world model.
 (2) BROAD episode evaluation -- detect ALL software-induced degradations
     (304 runs, 17 commits, 12 months) label-free, not just f7d3b72.
 (3) Linear-identifiability test (LeJEPA's prediction): a LINEAR probe on the frozen
     embedding should recover the KPI in-distribution and collapse across releases.

Models trained only on HEALTHY reference runs; the GPU work is small.
Outputs: results/fig9_sigreg.png, results/world_model_broad.csv
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
OUT = os.path.join(HERE, "results")
CSV = os.path.join(HERE, "..", "RANalyzer-Dataset", "processed-dataset.csv")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
T, DZ = 64, 32
torch.manual_seed(0); np.random.seed(0)
JOINT = ["snr", "rsrp", "ph", "nprb", "dl_bler", "dtx_inc", "dl_r3ratio", "dl_mcs"]


def load():
    pf = pd.read_parquet(os.path.join(CACHE, "perframe.parquet")).sort_values(["log_id", "frame_idx"])
    ix = pd.read_parquet(os.path.join(CACHE, "runs_index.parquet"))
    csv = pd.read_csv(CSV)[["log_id", "avg_dl_bler", "avg_snr", "git_hash"]]
    meta = ix.merge(csv, on="log_id", how="inner").dropna(subset=["avg_dl_bler"]).reset_index(drop=True)
    meta["date"] = pd.to_datetime(meta["log_id"].str[:8], format="%Y%m%d")
    g = pf.groupby("log_id")
    pf["dtx_inc"] = g["dtx"].diff().clip(lower=0)
    r0i, r3i = g["dl_r0"].diff().clip(lower=0), g["dl_r3"].diff().clip(lower=0)
    pf["dl_r3ratio"] = (r3i / r0i.replace(0, np.nan)).clip(0, 1.2)
    seqs = {lid: grp.head(T) for lid, grp in pf.groupby("log_id") if lid in set(meta["log_id"])}
    meta = meta[meta["log_id"].isin(seqs)].reset_index(drop=True)
    N = len(meta)
    X = np.zeros((N, T, len(JOINT)), np.float32); M = np.zeros((N, T), np.float32)
    for i, lid in enumerate(meta["log_id"].values):
        arr = seqs[lid][JOINT].to_numpy(np.float32); L = len(arr)
        X[i, :L] = arr; M[i, :L] = 1
    cm = np.nanmean(X.reshape(-1, len(JOINT)), 0)
    ii = np.where(np.isnan(X)); X[ii] = np.take(cm, ii[2])
    return meta, X, M


def standardize(X, M, train):
    flat = X[train][M[train] > 0]; mu, sd = flat.mean(0), flat.std(0) + 1e-6
    return ((X - mu) / sd) * M[..., None]


class JEPA(nn.Module):
    """GRU encoder; predict future latent from past latent (EMA-free)."""
    def __init__(self, f, h=DZ):
        super().__init__()
        self.enc = nn.GRU(f, h, batch_first=True)
        self.pred = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))

    def forward(self, x, lengths):
        o, _ = self.enc(x)
        half = (lengths // 2).clamp(min=1)
        zc = o[torch.arange(len(x)), half - 1]      # context summary
        zt = o[torch.arange(len(x)), lengths - 1]   # future summary
        return zc, zt, self.pred(zc)


def sigreg(Z, K=64):
    """Sketched isotropic-Gaussian regularizer: every random 1D projection -> N(0,1)."""
    U = torch.randn(Z.shape[1], K, device=Z.device)
    U = U / U.norm(dim=0, keepdim=True)
    P = Z @ U                                       # (B,K)
    m = P.mean(0); v = P.var(0, unbiased=False)
    c = P - m
    sk = (c ** 3).mean(0) / (v ** 1.5 + 1e-6)
    ku = (c ** 4).mean(0) / (v ** 2 + 1e-6)
    return (m ** 2 + (v - 1) ** 2 + sk ** 2 + (ku - 3) ** 2).mean()


def train_model(X, M, healthy, lam):
    Xs = standardize(X, M, healthy)
    xt = torch.tensor(Xs, device=DEV)
    lt = torch.tensor(M.sum(1).astype(int).clip(2), device=DEV)
    m = JEPA(X.shape[2]).to(DEV)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    hi = np.where(healthy)[0]
    for ep in range(45):
        m.train(); perm = np.random.permutation(hi)
        for s in range(0, len(perm), 256):
            b = torch.tensor(perm[s:s + 256], device=DEV)
            opt.zero_grad()
            zc, zt, ph = m(xt[b], lt[b])
            lp = ((ph - zt.detach()) ** 2).mean()
            ls = sigreg(torch.cat([zc, zt])) if lam > 0 else torch.zeros((), device=DEV)
            (lp + lam * ls).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        zc, zt, ph = m(xt, lt)
        anom = ((ph - zt) ** 2).mean(1).cpu().numpy()       # latent prediction error
        emb = zc.cpu().numpy()
    return anom, emb


def main():
    meta, X, M = load()
    date = meta["date"]
    healthy = ((date <= "2024-10-31") & (meta["avg_dl_bler"] < 0.15)).values
    print(f"[data] {len(meta)} runs; healthy reference={healthy.sum()}")

    # broad labels: software-induced degradation (post training window, no leakage)
    post = (date > "2024-10-31").values
    snr = meta["avg_snr"].values
    reg = post & (meta["avg_dl_bler"].values > 0.3) & ((snr > 25) | np.isnan(snr))
    norm = post & (meta["avg_dl_bler"].values < 0.15)
    sel = reg | norm
    print(f"[broad] regressed={reg.sum()} ({meta.loc[reg,'git_hash'].nunique()} commits), normal={norm.sum()}")

    res = {}
    for name, lam in [("plain (lam=0)", 0.0), ("SIGReg", 1.0)]:
        anom, emb = train_model(X, M, healthy, lam)
        auc_broad = roc_auc_score(reg[sel].astype(int), anom[sel])
        f7 = (meta["git_hash"] == "f7d3b72").values
        sel_f7 = norm | f7
        auc_f7 = roc_auc_score(f7[sel_f7].astype(int), anom[sel_f7])
        # per-commit ranking: precision@5 of commits with most degraded runs
        dfm = meta.assign(anom=anom)
        commit_anom = dfm[post].groupby("git_hash")["anom"].mean().sort_values(ascending=False)
        bad_commits = set(dfm[reg]["git_hash"].unique())
        top5 = commit_anom.head(5).index
        p_at5 = len(set(top5) & bad_commits) / 5
        # linear-identifiability of the embedding (LeJEPA): probe BLER from frozen emb
        tr_iid = np.random.rand(len(meta)) < 0.8
        ridge = Ridge(1.0).fit(emb[healthy], meta.loc[healthy, "avg_dl_bler"])
        # in-dist: probe trained on healthy, tested on held-out healthy-era runs
        he = np.where(healthy)[0]; np.random.shuffle(he)
        cut = int(.8 * len(he))
        ridge2 = Ridge(1.0).fit(emb[he[:cut]], meta["avg_dl_bler"].values[he[:cut]])
        r2_iid = r2_score(meta["avg_dl_bler"].values[he[cut:]], ridge2.predict(emb[he[cut:]]))
        r2_tmp = r2_score(meta["avg_dl_bler"].values[post], ridge2.predict(emb[post]))
        res[name] = dict(auc_broad=auc_broad, auc_f7=auc_f7, p_at5=p_at5,
                         lin_iid=r2_iid, lin_tmp=r2_tmp)
        print(f"\n[{name}]")
        print(f"  broad AUROC (304 degraded vs normal, 17 commits): {auc_broad:.3f}")
        print(f"  f7d3b72 AUROC: {auc_f7:.3f}   commit-rank precision@5: {p_at5:.2f}")
        print(f"  linear-probe identifiability  in-dist R2={r2_iid:.3f}  temporal R2={r2_tmp:.3f}")

    rep = pd.DataFrame(res).T.reset_index().rename(columns={"index": "model"})
    rep.to_csv(os.path.join(OUT, "world_model_broad.csv"), index=False)

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    names = list(res); x = np.arange(len(names))
    ax[0].bar(x - .2, [res[n]["auc_broad"] for n in names], .4, label="broad AUROC (14 commits)", color="C0")
    ax[0].bar(x + .2, [res[n]["p_at5"] for n in names], .4, label="commit-rank P@5", color="C2")
    ax[0].set_xticks(x); ax[0].set_xticklabels(names); ax[0].set_ylim(0, 1.05)
    ax[0].axhline(0.5, ls=":", color="gray"); ax[0].set_title("Broad label-free detection")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, axis="y")
    ax[1].bar(x - .2, [res[n]["lin_iid"] for n in names], .4, label="in-distribution", color="C0")
    ax[1].bar(x + .2, [max(0, res[n]["lin_tmp"]) for n in names], .4, label="across releases", color="C3")
    ax[1].set_xticks(x); ax[1].set_xticklabels(names)
    ax[1].set_title("Linear-probe identifiability\n(LeJEPA prediction)")
    ax[1].set_ylabel("KPI R$^2$ from frozen embedding"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig9_sigreg.png"), dpi=130)
    print("\nfigure -> results/fig9_sigreg.png")


if __name__ == "__main__":
    main()
