#!/usr/bin/env python
"""
Parse per-frame UE telemetry trajectories out of every nr-gnb.log and cache them.
Each run -> a multivariate time series of per-frame snapshots:
  PH, RSRP            (power headroom, avg RSRP)        -- channel
  dl_r0..r3, dl_err, dtx, dl_bler, dl_mcs               -- downlink reliability
  ul_r0, ul_err, uldtx, ul_bler, ul_mcs, nprb, snr      -- uplink + SNR

Cumulative counters (rounds/errors/dtx) are stored raw; the model code converts to
per-snapshot increments. Output: cache/perframe.parquet + cache/runs_index.parquet
"""
import os, re, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "..", "RANalyzer-Dataset", "cicd-dataset")
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)

RE_PH = re.compile(r"in-sync PH\s+(-?\d+)\s*dB.*?average RSRP\s+(-?\d+)")
RE_DL = re.compile(r"dlsch_rounds (\d+)/(\d+)/(\d+)/(\d+), dlsch_errors (\d+), "
                   r"pucch0_DTX (\d+), BLER ([\d.]+) MCS \((\d+)\)\s*(\d+)")
RE_UL = re.compile(r"ulsch_rounds (\d+)/(\d+)/(\d+)/(\d+), ulsch_errors (\d+), "
                   r"ulsch_DTX (\d+), BLER ([\d.]+) MCS \((\d+)\)\s*(\d+).*?"
                   r"NPRB (\d+)\s+SNR ([\d.\-]+)")
RE_VER = re.compile(r"Hash:\s*([0-9a-f]+)")

COLS = ["ph", "rsrp", "dl_r0", "dl_r1", "dl_r2", "dl_r3", "dl_err", "dtx",
        "dl_bler", "dl_mcs", "ul_r0", "ul_err", "uldtx", "ul_bler", "ul_mcs",
        "nprb", "snr"]


def parse_one(path):
    commit = None
    frames = []
    cur = None

    def flush():
        nonlocal cur
        if cur is not None and ("dl_bler" in cur or "snr" in cur):
            frames.append(cur)
        cur = None

    with open(path, errors="ignore") as fh:
        for ln in fh:
            if commit is None:
                m = RE_VER.search(ln)
                if m:
                    commit = m.group(1)
            mp = RE_PH.search(ln)
            if mp:
                flush()
                cur = {"ph": float(mp.group(1)), "rsrp": float(mp.group(2))}
                continue
            md = RE_DL.search(ln)
            if md:
                if cur is None:
                    cur = {}
                g = list(map(float, md.groups()))
                cur.update(dict(dl_r0=g[0], dl_r1=g[1], dl_r2=g[2], dl_r3=g[3],
                                dl_err=g[4], dtx=g[5], dl_bler=g[6], dl_mcs=g[8]))
                continue
            mu = RE_UL.search(ln)
            if mu:
                if cur is None:
                    cur = {}
                g = list(map(float, mu.groups()))
                cur.update(dict(ul_r0=g[0], ul_err=g[4], uldtx=g[5],
                                ul_bler=g[6], ul_mcs=g[8], nprb=g[9], snr=g[10]))
    flush()
    return commit, frames


def main():
    paths = sorted(glob.glob(os.path.join(LOGS, "2*", "*", "nr-gnb.log")))
    rows, idx = [], []
    n_ok = 0
    for i, p in enumerate(paths):
        parts = p.split(os.sep)
        log_id = f"{parts[-3]}_{parts[-2]}"
        commit, frames = parse_one(p)
        if not frames:
            continue
        n_ok += 1
        idx.append(dict(log_id=log_id, commit=commit, n_frames=len(frames)))
        for k, fr in enumerate(frames):
            r = {c: fr.get(c, np.nan) for c in COLS}
            r["log_id"] = log_id
            r["frame_idx"] = k
            rows.append(r)
        if (i + 1) % 1500 == 0:
            print(f"  parsed {i+1}/{len(paths)} logs, {n_ok} with frames")
    pf = pd.DataFrame(rows)
    ix = pd.DataFrame(idx)
    pf.to_parquet(os.path.join(CACHE, "perframe.parquet"))
    ix.to_parquet(os.path.join(CACHE, "runs_index.parquet"))
    print(f"\nDONE: {n_ok}/{len(paths)} logs had frames; "
          f"{len(pf)} frame-rows; median frames/run={ix['n_frames'].median():.0f}")
    print("frames/run quantiles:",
          ix["n_frames"].quantile([.1, .25, .5, .75, .9]).round(0).tolist())


if __name__ == "__main__":
    main()
