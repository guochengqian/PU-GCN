#!/usr/bin/env python3
"""Inspect a PU-GCN HDF5 training file.

Expected datasets for the default PU-GCN configuration are:
- poisson_1024: dense 4x point patches / ground truth for --num_point 256 --up_ratio 4
- poisson_256: optional FPS/uniform input patches used when training with --fps
"""
import argparse
import os

import h5py
import numpy as np


def describe_dataset(name, dataset, check_finite=False):
    print(f"{name}: shape={dataset.shape}, dtype={dataset.dtype}")
    if len(dataset.shape) != 3 or dataset.shape[-1] < 3:
        print(f"  WARNING: expected shape [num_patches, num_points, >=3], got {dataset.shape}")
    if dataset.shape[0] == 0:
        print("  WARNING: dataset has zero patches")
    if check_finite:
        sample = dataset[: min(16, dataset.shape[0])]
        if not np.isfinite(sample[..., :3]).all():
            print("  WARNING: first patches contain NaN or Inf coordinates")


def main():
    parser = argparse.ArgumentParser(description="Inspect PU-GCN HDF5 file keys/shapes.")
    parser.add_argument("h5_file")
    parser.add_argument("--num_point", type=int, default=256)
    parser.add_argument("--up_ratio", type=int, default=4)
    parser.add_argument("--check_finite", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(args.h5_file):
        raise FileNotFoundError(args.h5_file)
    if os.path.getsize(args.h5_file) == 0:
        raise RuntimeError(f"{args.h5_file} is zero bytes; re-download/extract the dataset")

    dense_key = f"poisson_{args.num_point * args.up_ratio}"
    fps_key = f"poisson_{args.num_point}"

    with h5py.File(args.h5_file, "r") as handle:
        keys = sorted(handle.keys())
        print("keys:", ", ".join(keys) if keys else "<none>")
        if not keys:
            raise RuntimeError("HDF5 file contains no datasets")
        for key in keys:
            describe_dataset(key, handle[key], check_finite=args.check_finite)
        if dense_key not in handle:
            print(f"WARNING: missing {dense_key}; default random-input training expects this key")
        if fps_key not in handle:
            print(f"WARNING: missing {fps_key}; --fps training expects this key")


if __name__ == "__main__":
    main()
