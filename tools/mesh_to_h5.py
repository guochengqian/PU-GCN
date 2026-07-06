#!/usr/bin/env python3
"""Create a simple PU-GCN-compatible HDF5 file from mesh files.

This utility samples surface points from each mesh and writes the HDF5 keys used
by Upsampling/data_loader.py. It is intended for custom datasets and sanity
checks; for paper-level reproduction, use the released PU1K preprocessing.
"""
import argparse
import glob
import os

import h5py
import numpy as np

trimesh = None


def load_as_mesh(path):
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        geometries = [g for g in mesh.geometry.values() if hasattr(g, "faces") and len(g.faces) > 0]
        if not geometries:
            raise ValueError(f"{path}: scene has no triangle meshes")
        mesh = trimesh.util.concatenate(geometries)
    if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
        raise ValueError(f"{path}: no triangle faces found")
    return mesh


def sample_surface(mesh, count, rng):
    points, _ = trimesh.sample.sample_surface_even(mesh, count)
    if points.shape[0] < count:
        extra, _ = trimesh.sample.sample_surface(mesh, count - points.shape[0])
        points = np.concatenate([points, extra], axis=0)
    if points.shape[0] > count:
        points = points[rng.choice(points.shape[0], count, replace=False)]
    return points.astype(np.float32)


def normalize(points):
    centroid = points.mean(axis=0, keepdims=True)
    points = points - centroid
    radius = np.sqrt((points ** 2).sum(axis=1)).max()
    if radius > 0:
        points = points / radius
    return points.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Sample meshes into PU-GCN HDF5 format.")
    parser.add_argument("--mesh_glob", required=True, help='Example: "meshes/*.off"')
    parser.add_argument("--output", required=True, help="Output .h5 path")
    parser.add_argument("--num_input", type=int, default=256)
    parser.add_argument("--num_gt", type=int, default=1024)
    parser.add_argument("--samples_per_mesh", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    global trimesh
    try:
        import trimesh as trimesh_module
    except ImportError as exc:
        raise SystemExit("Install dependencies first: pip install trimesh h5py scipy") from exc
    trimesh = trimesh_module

    mesh_paths = sorted(glob.glob(args.mesh_glob))
    if not mesh_paths:
        raise SystemExit(f"No meshes matched: {args.mesh_glob}")

    rng = np.random.default_rng(args.seed)
    dense_patches = []
    input_patches = []
    names = []

    for path in mesh_paths:
        mesh = load_as_mesh(path)
        for repeat in range(args.samples_per_mesh):
            dense = normalize(sample_surface(mesh, args.num_gt, rng))
            input_idx = rng.choice(args.num_gt, args.num_input, replace=args.num_input > args.num_gt)
            sparse = dense[input_idx].astype(np.float32)
            dense_patches.append(dense)
            input_patches.append(sparse)
            names.append(f"{os.path.basename(path)}#{repeat}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with h5py.File(args.output, "w") as handle:
        handle.create_dataset(f"poisson_{args.num_gt}", data=np.asarray(dense_patches, dtype=np.float32))
        handle.create_dataset(f"poisson_{args.num_input}", data=np.asarray(input_patches, dtype=np.float32))
        handle.create_dataset("name", data=np.asarray(names, dtype="S"))

    print(f"Wrote {args.output}")
    print(f"  poisson_{args.num_gt}: {len(dense_patches)} patches x {args.num_gt} points")
    print(f"  poisson_{args.num_input}: {len(input_patches)} patches x {args.num_input} points")


if __name__ == "__main__":
    main()
