#!/usr/bin/env python3
"""Run locked pooling/PCA ablations on patient-level UNI tensors.

Input CSV must contain: patient_id, embedding_path, and one column per target.
embedding_path points to a .pt tensor shaped (n_patches, 1024).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


TARGETS = ["ERBB2", "TACSTD2", "PVRL4", "FOLR1", "SLC39A6", "MET"]


def pool(path: str, method: str) -> np.ndarray:
    x = torch.load(Path(path), map_location="cpu", weights_only=False).float().numpy()
    if x.ndim != 2 or x.shape[1] != 1024:
        raise ValueError(f"{path}: expected (n_patches, 1024), got {x.shape}")
    if method == "mean":
        return x.mean(axis=0)
    if method == "max":
        return x.max(axis=0)
    if method == "mean_max":
        return np.concatenate([x.mean(axis=0), x.max(axis=0)])
    raise ValueError(method)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels_csv")
    ap.add_argument("--out", default="oof_predictions.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.labels_csv)
    required = {"patient_id", "embedding_path", *TARGETS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = df.drop_duplicates("patient_id").reset_index(drop=True)
    splits = list(KFold(args.folds, shuffle=True, random_state=args.seed).split(df))
    output = []

    for pooling in ("mean", "max", "mean_max"):
        features = np.vstack([pool(p, pooling) for p in df.embedding_path])
        for use_pca in (False, True):
            for target in TARGETS:
                y = df[target].to_numpy(float)
                pred = np.full(len(df), np.nan)
                for train, test in splits:
                    scaler = StandardScaler().fit(features[train])
                    xtr, xte = scaler.transform(features[train]), scaler.transform(features[test])
                    if use_pca:
                        n_components = min(256, xtr.shape[0] - 1, xtr.shape[1])
                        reducer = PCA(n_components=n_components, random_state=args.seed).fit(xtr)
                        xtr, xte = reducer.transform(xtr), reducer.transform(xte)
                    model = Ridge(alpha=10.0).fit(xtr, y[train])
                    pred[test] = model.predict(xte)
                output.extend(
                    {"patient_id": pid, "target": target, "y_true": truth, "y_pred": estimate,
                     "pooling": pooling, "pca": use_pca, "seed": args.seed}
                    for pid, truth, estimate in zip(df.patient_id, y, pred)
                )

    out = pd.DataFrame(output)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} out-of-fold rows to {args.out}")


if __name__ == "__main__":
    main()
