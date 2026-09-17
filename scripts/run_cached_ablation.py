#!/usr/bin/env python3
"""Run pooling/PCA ablations from patient_feature_cache.npz."""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

TARGETS = ["ERBB2", "TACSTD2", "PVRL4", "FOLR1", "SLC39A6", "MET"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels_csv")
    ap.add_argument("cache_npz")
    ap.add_argument("--out", default="oof_predictions.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    labels = pd.read_csv(args.labels_csv).drop_duplicates("patient_id").set_index("patient_id")
    cache = np.load(args.cache_npz)
    cached_ids = [str(x) for x in cache["patient_id"]]
    common = [pid for pid in cached_ids if pid in labels.index]
    if len(common) < 10:
        raise ValueError("Too few cached patients overlap labels")
    idx = [cached_ids.index(pid) for pid in common]
    kf = KFold(5, shuffle=True, random_state=args.seed)
    output = []
    for pooling in ("mean", "max", "mean_max"):
        base = cache["mean"][idx] if pooling == "mean" else cache["max"][idx]
        if pooling == "mean_max":
            base = np.concatenate([cache["mean"][idx], cache["max"][idx]], axis=1)
        for use_pca in (False, True):
            for target in TARGETS:
                y = labels.loc[common, target].to_numpy(float)
                pred = np.full(len(common), np.nan)
                for train, test in kf.split(base):
                    scaler = StandardScaler().fit(base[train])
                    xtr, xte = scaler.transform(base[train]), scaler.transform(base[test])
                    if use_pca:
                        n = min(256, xtr.shape[0] - 1, xtr.shape[1])
                        reducer = PCA(n_components=n, random_state=args.seed).fit(xtr)
                        xtr, xte = reducer.transform(xtr), reducer.transform(xte)
                    pred[test] = Ridge(alpha=10.0).fit(xtr, y[train]).predict(xte)
                output.extend({"patient_id": pid, "target": target, "y_true": truth,
                                "y_pred": estimate, "pooling": pooling, "pca": use_pca,
                                "seed": args.seed}
                               for pid, truth, estimate in zip(common, y, pred))
    pd.DataFrame(output).to_csv(args.out, index=False)
    print(f"wrote {len(output)} rows for {len(common)} patients to {args.out}")

if __name__ == "__main__":
    main()
