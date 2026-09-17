#!/usr/bin/env python3
"""Cache mean/max UNI features in one tensor pass before ablations."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels_csv")
    ap.add_argument("--out", default="patient_feature_cache.npz")
    args = ap.parse_args()
    df = pd.read_csv(args.labels_csv)
    required = {"patient_id", "embedding_path"}
    if not required.issubset(df.columns):
        raise ValueError("labels CSV needs patient_id and embedding_path")
    means, maxima, ids = [], [], []
    for row in df.drop_duplicates("patient_id").itertuples(index=False):
        x = torch.load(Path(row.embedding_path), map_location="cpu", weights_only=False).float().numpy()
        if x.ndim != 2 or x.shape[1] != 1024:
            raise ValueError(f"{row.embedding_path}: expected (n_patches, 1024), got {x.shape}")
        ids.append(str(row.patient_id))
        means.append(x.mean(axis=0))
        maxima.append(x.max(axis=0))
    np.savez_compressed(args.out, patient_id=np.array(ids), mean=np.vstack(means), max=np.vstack(maxima))
    print(f"cached {len(ids)} patients to {args.out}")


if __name__ == "__main__":
    main()
