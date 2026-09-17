#!/usr/bin/env python3
"""Create a deterministic patient-level pilot labels CSV."""
from __future__ import annotations
import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels_csv")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="pilot_patient_labels.csv")
    args = ap.parse_args()
    df = pd.read_csv(args.labels_csv)
    if "patient_id" not in df.columns:
        raise ValueError("labels CSV must contain patient_id")
    if df.patient_id.duplicated().any():
        raise ValueError("labels CSV must have one row per patient")
    pilot = df.sample(n=min(args.n, len(df)), random_state=args.seed).sort_values("patient_id")
    pilot.to_csv(args.out, index=False)
    print(f"wrote {len(pilot)} patients to {args.out}")

if __name__ == "__main__":
    main()
