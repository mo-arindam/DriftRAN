# DriftRAN

**Drift-aware Regression detection for RAN software.**

Reproduction code for the paper *"Drift-Aware World Models for Continuous RAN
Software Regression Detection."* We study concept drift in continuous over-the-air
(OTA) evaluation of an OpenAirInterface 5G stack across 65 software releases, show
that environment→KPI models are predictive within a release era but collapse across
releases (the drift is software-driven), and detect/attribute regressions with an
effect-size residual detector and a self-supervised JEPA world model of per-frame
telemetry.

## Dataset (not bundled)

DriftRAN runs on the public **RANalyzer dataset** (Shirkhani *et al.*, NetSoft 2026).
It is **not** redistributed here. Clone it as a **sibling directory** of `DriftRAN`:

```
parent/
├── DriftRAN/                 # this repo
└── RANalyzer-Dataset/        # the public dataset (clone here)
    ├── processed-dataset.csv
    └── cicd-dataset/<date>/<time>/nr-gnb.log ...
```

```bash
git clone https://github.com/wineslab/RANalyzer-Dataset ../RANalyzer-Dataset
```
> Verify the dataset URL/citation against the published RANalyzer paper.

The scripts locate the dataset at `../RANalyzer-Dataset/` relative to this folder, so
no path edits are needed when the two repos sit side by side.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
PyTorch: install the build matching your accelerator (CPU, CUDA, or ROCm). The
world-model scripts use a small GRU and run on CPU in minutes; a GPU is optional.

## Reproduce

```bash
./run_all.sh            # runs T0 -> T1 -> T2 in dependency order
```
Outputs (figures + CSVs) are written to `results/`; parsed sequences are cached in
`cache/`. Both are git-ignored.

### Run order (dependencies)

| Tier | Script | Produces (paper) | Needs |
|---|---|---|---|
| **T0** | `parse_perframe_all.py` | builds `cache/` (per-frame sequences) | raw logs |
| T1 | `predictability.py` | **Fig. 7** (centerpiece) | CSV |
| T1 | `holdout_eval.py` | **Fig. 6** (held-out + bootstrap CIs) | CSV |
| T1 | `case_and_detector.py` | **Fig. 3** (case study) | CSV |
| T1 | `rach_target.py` | **Fig. 5** (control-plane RACH) | CSV |
| T1 | `baselines.py` | **Fig. 10**, Table V (baselines) | CSV + `cache/anomaly_scores`† |
| T1 | `ablations_detector.py` | **Fig. 11** (detector sensitivity) | CSV |
| T1 | `drift_feasibility.py` | covariate-vs-concept feasibility | CSV |
| T1 | `covariate_overlap.py` | covariate-overlap/domain-shift diagnostics | CSV |
| T1 | `placebo_controls.py` | commit-label and episode-window negative controls | CSV |
| T1 | `robustness_exclusions.py` | transfer-gap robustness excluding `f7d3b72` | CSV |
| T1 | `world_model_probe.py` | transfer gap; writes `cache/anomaly_scores` | `cache/perframe` |
| T1 | `world_model_sigreg.py` | **Fig. 9**, Table IV (SIGReg) | `cache/perframe` |
| T1 | `ablations_worldmodel.py` | **Fig. 12** (world-model ablations) | `cache/perframe` |
| **T2** | `world_model_analysis.py` | **Fig. 8** (anomaly over time) | `cache/anomaly_scores` |

† `baselines.py` and `world_model_analysis.py` read `cache/anomaly_scores.parquet`,
which is written by `world_model_probe.py` — run the probe first (handled by
`run_all.sh`).

## Layout

```
DriftRAN/
├── *.py                 # analysis scripts (see table)
├── run_all.sh           # T0 -> T1 -> T2
├── requirements.txt
├── cache/               # generated (git-ignored)
└── results/             # generated figures + CSVs (git-ignored)
```

## Citation

```bibtex
@article{driftran2026,
  author  = {Mondal, Arindam},
  title   = {Drift-Aware World Models for Continuous RAN Software Regression Detection},
  journal = {(under submission)},
  year    = {2026}
}
```

The dataset is the property of its original authors; please also cite the RANalyzer
paper if you use it.

## License

Code: MIT (see `LICENSE`). The RANalyzer dataset is licensed separately by its authors.
