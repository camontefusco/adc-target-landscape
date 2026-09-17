#!/usr/bin/env bash
set -euo pipefail

LABELS_CSV="${1:?usage: run_all_benchmark.sh labels.csv [work_dir]}"
WORK_DIR="${2:-benchmark_run}"
mkdir -p "$WORK_DIR"

python scripts/build_patient_feature_cache.py "$LABELS_CSV"   --out "$WORK_DIR/patient_feature_cache.npz"

python scripts/run_cached_ablation.py "$LABELS_CSV"   "$WORK_DIR/patient_feature_cache.npz"   --out "$WORK_DIR/oof_predictions.csv"

python scripts/evaluate_benchmark.py "$WORK_DIR/oof_predictions.csv"   --patient-metrics-out "$WORK_DIR/patient_ranking_metrics.csv"   --summary-out "$WORK_DIR/ranking_summary.csv"

echo "Bulk benchmark complete: $WORK_DIR"
