#!/usr/bin/env python3
"""Merge chunk CSV files into a single dataset, preserving original sample order."""
import sys, glob, pandas as pd, os

mat_index = int(sys.argv[1])
n_stages = int(sys.argv[2])
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

pattern = os.path.join(data_dir, f"dataset_material{mat_index}_{n_stages}stage_chunk*.csv")
chunks = sorted(glob.glob(pattern))

if not chunks:
    print(f"No chunk files found for material {mat_index}, {n_stages}-stage")
    sys.exit(1)

dfs = [pd.read_csv(f) for f in chunks]
merged = pd.concat(dfs, ignore_index=True).sort_values("sample_idx").reset_index(drop=True)
merged = merged.drop(columns=["sample_idx"])

outpath = os.path.join(data_dir, f"dataset_material{mat_index}_{n_stages}stage.csv")
merged.to_csv(outpath, index=False)
print(f"Merged {len(chunks)} chunks → {outpath} ({len(merged)} samples)")

# Clean up chunk files
for f in chunks:
    os.remove(f)
    print(f"  Removed {os.path.basename(f)}")
