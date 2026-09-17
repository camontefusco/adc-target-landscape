#!/usr/bin/env python3
"""Evaluate locked within-patient ADC target ranking metrics.

Input CSV columns: patient_id, split, target, y_true, y_pred.
The evaluator never refits models and is safe to run on exported OOF predictions.
"""
from __future__ import annotations
import argparse
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def pairwise_accuracy(true_row: np.ndarray, pred_row: np.ndarray) -> float:
    pairs = list(combinations(range(len(true_row)), 2))
    outcomes = [np.sign(true_row[j] - true_row[i]) == np.sign(pred_row[j] - pred_row[i])
                for i, j in pairs if true_row[j] != true_row[i] and pred_row[j] != pred_row[i]]
    return float(np.mean(outcomes)) if outcomes else np.nan


def evaluate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"patient_id", "target", "y_true", "y_pred"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    wide = frame.pivot_table(index="patient_id", columns="target", values=["y_true", "y_pred"], aggfunc="first")
    targets = sorted(wide["y_true"].columns)
    rows = []
    for patient_id in wide.index:
        truth = wide["y_true"].loc[patient_id, targets].to_numpy(float)
        pred = wide["y_pred"].loc[patient_id, targets].to_numpy(float)
        keep = np.isfinite(truth) & np.isfinite(pred)
        if keep.sum() < 2:
            continue
        t, p = truth[keep], pred[keep]
        rows.append({
            "patient_id": patient_id,
            "spearman": spearmanr(t, p).statistic if len(np.unique(t)) > 1 and len(np.unique(p)) > 1 else np.nan,
            "kendall": kendalltau(t, p).statistic if len(np.unique(t)) > 1 and len(np.unique(p)) > 1 else np.nan,
            "pairwise_accuracy": pairwise_accuracy(t, p),
            "top1_correct": float(np.argmax(t) == np.argmax(p)),
            "top2_recall": float(np.argmax(t) in np.argsort(p)[-2:]),
        })
    patient_metrics = pd.DataFrame(rows)
    summary = patient_metrics.drop(columns=["patient_id"]).agg(["mean", "median", "count"]).T.reset_index(names="metric")
    return patient_metrics, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_csv")
    parser.add_argument("--patient-metrics-out", default="patient_ranking_metrics.csv")
    parser.add_argument("--summary-out", default="ranking_summary.csv")
    args = parser.parse_args()
    patient_metrics, summary = evaluate(pd.read_csv(args.predictions_csv))
    patient_metrics.to_csv(args.patient_metrics_out, index=False)
    summary.to_csv(args.summary_out, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
