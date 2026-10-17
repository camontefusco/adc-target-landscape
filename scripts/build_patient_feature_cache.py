#!/usr/bin/env python3
"""Cache mean/max UNI features with bounded per-patient memory."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_embedding(path: Path) -> torch.Tensor:
    kwargs = {"map_location": "cpu", "weights_only": False}
    try:
        return torch.load(path, mmap=True, **kwargs)
    except (TypeError, RuntimeError):
        return torch.load(path, **kwargs)


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
        path = Path(row.embedding_path)
        x = load_embedding(path)
        if x.ndim != 2 or x.shape[1] != 1024:
            raise ValueError(f"{path}: expected (n_patches, 1024), got {tuple(x.shape)}")
        if x.dtype != torch.float32:
            x = x.float()
        ids.append(str(row.patient_id))
        means.append(x.mean(dim=0).cpu().numpy().astype(np.float32, copy=False))
        maxima.append(x.amax(dim=0).cpu().numpy().astype(np.float32, copy=False))
        del x
        gc.collect()

    np.savez_compressed(
        args.out,
        patient_id=np.asarray(ids),
        mean=np.vstack(means),
        max=np.vstack(maxima),
    )
    print(f"cached {len(ids)} patients to {args.out}")


if __name__ == "__main__":
    main()
