#!/usr/bin/env bash
# Reproduce all DriftRAN figures/tables in dependency order (T0 -> T1 -> T2).
# Usage:  ./run_all.sh            (uses `python`)
#         PYTHON=/path/to/python ./run_all.sh
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
mkdir -p results cache

DATA="../RANalyzer-Dataset/processed-dataset.csv"
if [ ! -f "$DATA" ]; then
  echo "ERROR: dataset not found at $DATA"
  echo "Clone it as a sibling directory:  git clone <url> ../RANalyzer-Dataset"
  exit 1
fi

echo "== T0: parse per-frame telemetry into cache/ =="
"$PY" parse_perframe_all.py

echo "== T1: CSV-only analyses =="
for s in predictability holdout_eval case_and_detector rach_target \
         ablations_detector drift_feasibility; do
  echo "-- $s"; "$PY" "$s.py"
done

echo "== T1: world-model (needs cache/perframe) =="
"$PY" world_model_probe.py          # also writes cache/anomaly_scores.parquet
"$PY" world_model_sigreg.py
"$PY" ablations_worldmodel.py

echo "== T2: needs cache/anomaly_scores.parquet from the probe =="
"$PY" world_model_analysis.py
"$PY" baselines.py

echo "== done. figures + CSVs in results/ =="
