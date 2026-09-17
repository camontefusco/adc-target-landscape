#!/usr/bin/env python3
"""Create the locked patient labels table for the benchmark."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

TARGETS = ["ERBB2", "TACSTD2", "PVRL4", "FOLR1", "SLC39A6", "MET"]
XENA_URL = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FHiSeqV2_PANCAN.gz"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest_csv")
    ap.add_argument("embedding_root")
    ap.add_argument("--out", default="patient_labels.csv")
    ap.add_argument("--xena", default=XENA_URL)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest_csv)
    if not {"case_id", "slide_id"}.issubset(manifest.columns):
        raise ValueError("manifest must contain case_id and slide_id")
    chosen = manifest.sort_values(["case_id", "slide_id"]).drop_duplicates("case_id")
    expression = pd.read_csv(args.xena, sep="\t", compression="gzip", index_col=0)
    primary_cols = [c for c in expression.columns if len(c) >= 15 and c[13:15] == "01"]
    primary = expression.loc[:, primary_cols]
    primary.columns = [c[:12] for c in primary.columns]
    primary = primary.T.groupby(level=0).mean().T
    chosen["patient_id"] = chosen["case_id"].astype(str).str[:12]
    chosen["embedding_path"] = [
        str(Path(args.embedding_root) / f"{slide}.pt") for slide in chosen["slide_id"]
    ]
    labels = chosen[["patient_id", "embedding_path"]].copy()
    for target in TARGETS:
        if target not in primary.index:
            raise ValueError(f"Xena target missing: {target}")
        labels[target] = labels.patient_id.map(primary.loc[target])
    labels = labels.dropna(subset=TARGETS).drop_duplicates("patient_id")
    labels.to_csv(args.out, index=False)
    print(f"wrote {len(labels)} patients to {args.out}")

if __name__ == "__main__":
    main()
