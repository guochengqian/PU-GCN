# PU-GCN troubleshooting notes for open GitHub issues

This page collects fixes and maintainer answers for the currently open PU-GCN issues.

## #21: `model.py` `IndexError` during test phase

Root cause: the TensorFlow farthest-point-sampling CUDA op is only safe when the requested sample count is not larger than the number of candidate points. Some test inputs, especially small or heavily cropped point clouds, requested more FPS samples than available. The op could then return invalid indices, causing crashes at patch extraction or final output sampling.

Fix:

- Added `Upsampling/sampling_utils.py` with `clamp_sample_count()` and `safe_index_array()`.
- In `Upsampling/model.py`, patch seed count is clamped to `[1, number_of_input_points]`.
- FPS output indices are filtered to valid NumPy indices before indexing.
- Final output FPS is clamped to available prediction candidates and logs a warning if the requested output count is impossible.

## #10: Hausdorff metric and point-to-surface metric questions

### Hausdorff

Root cause: `evaluate.py` used the sum of the two directional maxima. The symmetric Hausdorff distance should be the maximum of the two directional maxima:

```python
max(max(forward_nn_distances), max(backward_nn_distances))
```

Fix:

- Added `Common/metrics.py::hausdorff_from_nn_distances()`.
- Updated `evaluate.py` to use the correct formula.

### Point-to-surface (P2F)

The existing `evaluation_code/evaluation.cpp` reports prediction-point-to-mesh-surface distance, not a symmetric surface-to-point metric. The code now documents this behavior more explicitly through the issue report. A fully symmetric surface metric requires defining a surface sampling strategy and density; this is not silently changed because it would make new numbers incomparable with old PU-GCN/PU-GAN/MPU tables.

Related robustness fix:

- `evaluation.cpp` now skips non-finite prediction points and non-finite surface projections instead of writing NaN distances into the P2F output.

## #11: P2F metric returns NaN

Likely causes:

- prediction `.xyz` contains NaN/Inf values;
- unsupported/empty prediction file;
- invalid or unreadable mesh;
- CGAL cannot locate a stable closest surface projection for a problematic point/mesh.

Fix:

- Added file-open checks for mesh and prediction files.
- Added prediction extension validation (`.ply` or `.xyz`).
- Added non-finite point filtering.
- Added non-finite surface-projection/distance guards.
- Avoided NaN standard deviation when only one valid distance exists.

## #13: evaluation compilation prints CGAL warning then `./evaluation: No such file or directory`

Root cause: CMake returned successfully even when CGAL was missing, so `make` produced no executable and `compile.sh` still tried to run `./evaluation`.

Fix:

- `evaluation_code/CMakeLists.txt` now fails with `FATAL_ERROR` if CGAL is unavailable.
- `evaluation_code/compile.sh` now uses `set -euo pipefail`, documents install commands, uses a portable job count, and checks that `./evaluation` exists before running it.

Ubuntu/Debian dependency hint:

```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential libcgal-dev libboost-all-dev
```

## #16: "Empty HDF5 file" in PU1K train

The released `PU1K.zip` contains a non-empty training HDF5 file. If HDFView or Python shows an empty file, it is usually an incomplete Google Drive download, failed unzip, or a zero-byte placeholder.

Fix/support:

- Added `tools/check_h5.py` to inspect HDF5 keys, shapes, file size, and optional finite-coordinate checks.

Example:

```bash
python tools/check_h5.py ./data/PU1K/train/pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5 --check_finite
```

For the default config, expect at least `poisson_1024`; `poisson_256` is used when training with `--fps`.

## #17: Why does the PU1K HDF5 filename include `addpugan`?

The filename suffix `addpugan` indicates that the provided training HDF5 was prepared with additional PU-GAN-style training data/patches for compatibility with prior baselines. It is the intended default file used by this repository, not a sign that the download is corrupted.

## #18: Custom mesh dataset to `.h5`

Fix/support:

- Added `tools/mesh_to_h5.py`, a minimal custom-dataset converter that samples mesh surfaces and writes PU-GCN-compatible keys:
  - `poisson_1024` for the default `--num_point 256 --up_ratio 4` dense/GT patches;
  - `poisson_256` for `--fps` input patches.

Example:

```bash
pip install trimesh h5py scipy
python tools/mesh_to_h5.py --mesh_glob "./my_meshes/*.off" --output ./data/custom.h5 --num_input 256 --num_gt 1024 --samples_per_mesh 20
python main.py --phase train --data_dir ./data/custom.h5 --model pugcn --upsampler nodeshuffle --k 20
```

Note: this converter is a practical baseline for custom data. It does not reproduce every PU1K preprocessing detail such as patch extraction density, train/test split policy, or exact Poisson-disk sampling choices.

## #15: PU-GAN test data has only 27 `.off` meshes

PU-GAN's public test set is mesh-based. Papers commonly generate input and ground-truth point pairs from these meshes using their own sampling/evaluation scripts, which is why only `.off` files may be present in the original PU-GAN test download. PU1K includes explicit point-cloud test files, so prefer PU1K for reproducible point-cloud input/GT pairs.

## #14: KITTI raw test produces hollow-looking upsampled clouds

Likely cause: the released PU-GCN checkpoints were trained on normalized object/patch datasets, not raw sparse LiDAR scenes. Raw KITTI scans have very different density, holes, ground structure, and occlusion patterns, so the upsampler may fill surfaces in visually hollow or biased ways.

Mitigations:

- Use the checkpoint whose sampling mode matches your input (`random` vs uniform/FPS). The maintainer comment suggests using the pretrained PU-GCN trained with uniform sampling on PU-GAN data for this kind of test.
- Crop/normalize local object-like patches instead of feeding an entire raw LiDAR frame.
- Increase `--patch_num_ratio` for denser overlap during testing.
- Treat KITTI raw results as domain-transfer experiments, not paper-benchmark reproduction.
