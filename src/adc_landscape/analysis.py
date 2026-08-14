"""Minimal analysis utilities for the LXAI ADC target-landscape pilot."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def mean_pool_patch_embeddings(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("Expected patch matrix with shape (n_patches, embedding_dim).")
    if x.shape[0] == 0:
        raise ValueError("Cannot pool an empty patch matrix.")
    return x.mean(axis=0)


def patient_target_zscores(y_train: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardize each target using training-fold statistics only."""
    mu = np.nanmean(y_train, axis=0)
    sd = np.nanstd(y_train, axis=0, ddof=0)
    if np.any(~np.isfinite(sd)) or np.any(sd == 0):
        raise ValueError("At least one target has zero/non-finite training-fold SD.")
    return (y_train - mu) / sd, (y_test - mu) / sd


def patient_ranking_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Per-patient ranking metrics across targets."""
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have identical shapes.")
    rows = []
    n_targets = y_true.shape[1]
    for i, (truth, pred) in enumerate(zip(y_true, y_pred)):
        rho = spearmanr(truth, pred).statistic
        tau = kendalltau(truth, pred).statistic
        top_true = int(np.argmax(truth))
        order_pred = np.argsort(pred)[::-1]
        pair_total = pair_correct = 0
        for a in range(n_targets):
            for b in range(a + 1, n_targets):
                if truth[a] == truth[b]:
                    continue
                pair_total += 1
                pair_correct += int(np.sign(truth[a] - truth[b]) == np.sign(pred[a] - pred[b]))
        rows.append({
            "row": i,
            "spearman": rho,
            "kendall_tau": tau,
            "top1_correct": int(order_pred[0] == top_true),
            "top2_correct": int(top_true in order_pred[:2]),
            "pairwise_accuracy": pair_correct / pair_total if pair_total else np.nan,
        })
    return pd.DataFrame(rows)


def target_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str]) -> pd.DataFrame:
    rows = []
    for j, target in enumerate(target_names):
        yt, yp = y_true[:, j], y_pred[:, j]
        rows.append({
            "target": target,
            "spearman": spearmanr(yt, yp).statistic,
            "mae": float(np.mean(np.abs(yt - yp))),
            "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
        })
    return pd.DataFrame(rows)


def repeated_ridge_oof(
    X: np.ndarray,
    Y: np.ndarray,
    target_names: list[str],
    n_splits: int = 5,
    seeds: tuple[int, ...] = (17, 41, 73),
    alpha: float = 10.0,
    pca_components: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Repeated patient-level K-fold Ridge with leakage-safe target z-scoring."""
    X, Y = np.asarray(X), np.asarray(Y)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must contain the same patients.")
    if Y.shape[1] != len(target_names):
        raise ValueError("target_names does not match Y columns.")

    prediction_rows, ranking_rows, target_rows = [], [], []
    for seed in seeds:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(kf.split(X)):
            ytr, yte = patient_target_zscores(Y[tr], Y[te])
            steps = [("xscale", StandardScaler())]
            if pca_components is not None:
                ncomp = min(pca_components, len(tr) - 1, X.shape[1])
                steps.append(("pca", PCA(n_components=ncomp, random_state=seed)))
            steps.append(("ridge", Ridge(alpha=alpha)))
            model = Pipeline(steps)
            model.fit(X[tr], ytr)
            pred = model.predict(X[te])

            rank = patient_ranking_metrics(yte, pred)
            rank["seed"], rank["fold"] = seed, fold
            rank["patient_index"] = te
            ranking_rows.append(rank)

            tm = target_metrics(yte, pred, target_names)
            tm["seed"], tm["fold"] = seed, fold
            target_rows.append(tm)

            for local_i, global_i in enumerate(te):
                for j, target in enumerate(target_names):
                    prediction_rows.append({
                        "seed": seed, "fold": fold, "patient_index": int(global_i),
                        "target": target, "y_true_z": float(yte[local_i, j]),
                        "y_pred_z": float(pred[local_i, j]),
                    })
    return pd.DataFrame(prediction_rows), pd.concat(ranking_rows, ignore_index=True), pd.concat(target_rows, ignore_index=True)


def shuffled_label_control(X: np.ndarray, Y: np.ndarray, target_names: list[str], seed: int = 2026, **kwargs):
    rng = np.random.default_rng(seed)
    shuffled = Y[rng.permutation(len(Y))]
    return repeated_ridge_oof(X, shuffled, target_names, **kwargs)


def summarize_rankings(ranking_df: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "median_patient_spearman": ranking_df["spearman"].median(),
        "median_patient_kendall_tau": ranking_df["kendall_tau"].median(),
        "mean_pairwise_accuracy": ranking_df["pairwise_accuracy"].mean(),
        "top1_accuracy": ranking_df["top1_correct"].mean(),
        "top2_recall": ranking_df["top2_correct"].mean(),
    })
