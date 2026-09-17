#!/usr/bin/env python3
"""Score patch-level contributions for a fitted linear target model.

The coefficient vector must be in the original 1024-dimensional UNI space.
For PCA models, back-project coefficients first:
    beta_original = pca.components_.T @ beta_pca
Patch contributions sum to the prediction's centered linear contribution.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("embedding_pt")
    ap.add_argument("coefficient_npy")
    ap.add_argument("--scaler-mean", required=True)
    ap.add_argument("--scaler-scale", required=True)
    ap.add_argument("--out", default="patch_contributions.csv")
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    x = torch.load(Path(args.embedding_pt), map_location="cpu", weights_only=False).float().numpy()
    beta = np.load(args.coefficient_npy).astype(float).ravel()
    mean = np.load(args.scaler_mean).astype(float).ravel()
    scale = np.load(args.scaler_scale).astype(float).ravel()
    if x.ndim != 2 or x.shape[1] != len(beta):
        raise ValueError(f"embedding shape {x.shape} and coefficient length {len(beta)} disagree")
    if len(mean) != len(beta) or len(scale) != len(beta):
        raise ValueError("scaler arrays must match coefficient length")

    standardized = (x - mean) / np.where(scale == 0, 1.0, scale)
    contributions = standardized @ beta
    order = np.argsort(contributions)
    k = min(args.top_k, len(order))
    selected = np.concatenate([order[:k], order[-k:][::-1]])
    result = pd.DataFrame({
        "patch_index": selected,
        "contribution": contributions[selected],
        "direction": np.where(contributions[selected] >= 0, "positive", "negative"),
    })
    result.to_csv(args.out, index=False)
    print(f"wrote {len(result)} patch contributions to {args.out}")


if __name__ == "__main__":
    main()
