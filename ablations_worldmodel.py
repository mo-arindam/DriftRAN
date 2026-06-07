#!/usr/bin/env python
"""
World-model ablations:
 (1) SIGReg lambda sweep {0,0.5,1,2,4}: broad AUROC, commit-rank P@5, and linear-probe
     identifiability (in-dist vs across-release) -- shows the SIGReg effect is a smooth
     trend, not a single-lambda artifact.
 (2) Feature ablation: channel-only [snr,rsrp,ph,nprb] vs full joint state. Channel-only
     should FAIL to detect (the BLER regression is invisible in pure-channel space),
     reinforcing the observation-scope limitation.

Output: results/fig12_wm_ablation.png, results/ablation_worldmodel.csv
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
T, DZ, EP = 64, 32, 30
torch.manual_seed(0); np.random.seed(0)
FULL = ["snr", "rsrp", "ph", "nprb", "dl_bler", "dtx_inc", "dl_r3ratio", "dl_mcs"]
CHAN = ["snr", "rsrp", "ph", "nprb"]


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
    return meta, seqs


def tensor(meta, seqs, feats):
    N = len(meta); X = np.zeros((N, T, len(feats)), np.float32); M = np.zeros((N, T), np.float32)
    for i, lid in enumerate(meta["log_id"].values):
        a = seqs[lid][feats].to_numpy(np.float32); L = len(a); X[i, :L] = a; M[i, :L] = 1
    cm = np.nanmean(X.reshape(-1, len(feats)), 0); ii = np.where(np.isnan(X)); X[ii] = np.take(cm, ii[2])
    return X, M


def standardize(X, M, train):
    flat = X[train][M[train] > 0]; mu, sd = flat.mean(0), flat.std(0) + 1e-6
    return ((X - mu) / sd) * M[..., None]


class JEPA(nn.Module):
    def __init__(self, f, h=DZ):
        super().__init__()
        self.enc = nn.GRU(f, h, batch_first=True)
        self.pred = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))

    def forward(self, x, lengths):
        o, _ = self.enc(x)
        half = (lengths // 2).clamp(min=1)
        zc = o[torch.arange(len(x)), half - 1]
        zt = o[torch.arange(len(x)), lengths - 1]
        return zc, zt, self.pred(zc)


def sigreg(Z, K=48):
    U = torch.randn(Z.shape[1], K, device=Z.device); U = U / U.norm(dim=0, keepdim=True)
    P = Z @ U; m = P.mean(0); v = P.var(0, unbiased=False); c = P - m
    sk = (c ** 3).mean(0) / (v ** 1.5 + 1e-6); ku = (c ** 4).mean(0) / (v ** 2 + 1e-6)
    return (m ** 2 + (v - 1) ** 2 + sk ** 2 + (ku - 3) ** 2).mean()


def run(meta, X, M, healthy, lam):
    Xs = standardize(X, M, healthy)
    xt = torch.tensor(Xs, device=DEV); lt = torch.tensor(M.sum(1).astype(int).clip(2), device=DEV)
    m = JEPA(X.shape[2]).to(DEV); opt = torch.optim.Adam(m.parameters(), 3e-3)
    hi = np.where(healthy)[0]
    for ep in range(EP):
        m.train(); perm = np.random.permutation(hi)
        for s in range(0, len(perm), 256):
            b = torch.tensor(perm[s:s + 256], device=DEV)
            opt.zero_grad(); zc, zt, ph = m(xt[b], lt[b])
            loss = ((ph - zt.detach()) ** 2).mean()
            if lam > 0:
                loss = loss + lam * sigreg(torch.cat([zc, zt]))
            loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        zc, zt, ph = m(xt, lt)
        anom = ((ph - zt) ** 2).mean(1).cpu().numpy(); emb = zc.cpu().numpy()
    return anom, emb


def evaluate(meta, anom, emb, healthy):
    date = meta["date"]; post = (date > "2024-10-31").values; snr = meta["avg_snr"].values
    reg = post & (meta["avg_dl_bler"].values > 0.3) & ((snr > 25) | np.isnan(snr))
    norm = post & (meta["avg_dl_bler"].values < 0.15); sel = reg | norm
    auc = roc_auc_score(reg[sel].astype(int), anom[sel])
    dfm = meta.assign(anom=anom)
    ca = dfm[post].groupby("git_hash")["anom"].mean().sort_values(ascending=False)
    bad = set(dfm[reg]["git_hash"].unique())
    p5 = len(set(ca.head(5).index) & bad) / 5
    he = np.where(healthy)[0]; np.random.shuffle(he); cut = int(.8 * len(he))
    rg = Ridge(1.0).fit(emb[he[:cut]], meta["avg_dl_bler"].values[he[:cut]])
    r2i = r2_score(meta["avg_dl_bler"].values[he[cut:]], rg.predict(emb[he[cut:]]))
    r2t = r2_score(meta["avg_dl_bler"].values[post], rg.predict(emb[post]))
    return dict(auc=auc, p5=p5, lin_iid=r2i, lin_tmp=r2t)


def main():
    meta, seqs = load()
    healthy = ((meta["date"] <= "2024-10-31") & (meta["avg_dl_bler"] < 0.15)).values
    Xf, Mf = tensor(meta, seqs, FULL)
    print(f"[data] {len(meta)} runs, healthy={healthy.sum()}")

    # (1) lambda sweep (full features)
    rows = []
    for lam in [0.0, 0.5, 1.0, 2.0, 4.0]:
        a, e = run(meta, Xf, Mf, healthy, lam)
        m = evaluate(meta, a, e, healthy); m["lam"] = lam; m["feats"] = "full"
        rows.append(m)
        print(f"  lam={lam}: AUROC={m['auc']:.3f} P@5={m['p5']:.2f} "
              f"lin_iid={m['lin_iid']:.3f} lin_tmp={m['lin_tmp']:.3f}")
    # (2) feature ablation: channel-only at lam=1
    Xc, Mc = tensor(meta, seqs, CHAN)
    a, e = run(meta, Xc, Mc, healthy, 1.0)
    mc = evaluate(meta, a, e, healthy); mc["lam"] = 1.0; mc["feats"] = "channel-only"
    rows.append(mc)
    print(f"  channel-only (lam=1): AUROC={mc['auc']:.3f} P@5={mc['p5']:.2f} "
          f"-- expected to FAIL (regression invisible in channel space)")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "ablation_worldmodel.csv"), index=False)

    sweep = res[res.feats == "full"].sort_values("lam")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(sweep["lam"], sweep["auc"], "-o", label="broad AUROC", color="C0")
    ax[0].plot(sweep["lam"], sweep["p5"], "-s", label="commit-rank P@5", color="C2")
    ax[0].axhline(mc["auc"], ls="--", color="C3", label="channel-only AUROC")
    ax[0].set_xlabel(r"SIGReg weight $\lambda$"); ax[0].set_ylim(0, 1.05)
    ax[0].set_title("Detection vs SIGReg $\\lambda$"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].plot(sweep["lam"], sweep["lin_iid"], "-o", label="in-distribution", color="C0")
    ax[1].plot(sweep["lam"], sweep["lin_tmp"].clip(lower=0), "-s", label="across releases", color="C3")
    ax[1].set_xlabel(r"SIGReg weight $\lambda$"); ax[1].set_ylabel("linear-probe KPI $R^2$")
    ax[1].set_title("Identifiability vs $\\lambda$"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig12_wm_ablation.png"), dpi=130)
    print("\nfigure -> results/fig12_wm_ablation.png")


if __name__ == "__main__":
    main()
