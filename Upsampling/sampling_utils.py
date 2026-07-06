import numpy as np


def clamp_sample_count(requested_count, available_count, minimum=0):
    """Clamp a requested sampling count to the valid range for a point set.

    TensorFlow FPS kernels assume ``0 < npoint <= ndataset``. If callers ask
    for more samples than available, the CUDA op may return undefined/out-of-
    bounds indices. Keep the request inside bounds before invoking the op.
    """
    available_count = int(available_count)
    if available_count <= 0:
        raise ValueError("available_count must be positive")

    requested_count = int(requested_count)
    minimum = int(minimum)
    if minimum < 0:
        raise ValueError("minimum must be non-negative")
    if requested_count < minimum:
        requested_count = minimum
    return min(requested_count, available_count)


def safe_index_array(indices, available_count, desired_count=None):
    """Return valid, unique integer indices for a point set.

    Invalid FPS outputs are filtered instead of being passed to NumPy indexing.
    If ``desired_count`` is provided and the sampler returned too few valid
    values, deterministic missing indices are appended from the available range.
    """
    available_count = int(available_count)
    if available_count <= 0:
        raise ValueError("available_count must be positive")

    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    idx = idx[(idx >= 0) & (idx < available_count)]

    seen = set()
    valid = []
    for value in idx.tolist():
        if value not in seen:
            seen.add(value)
            valid.append(value)

    if desired_count is None:
        desired_count = len(valid)
    desired_count = clamp_sample_count(desired_count, available_count, minimum=0)

    if len(valid) < desired_count:
        for value in range(available_count):
            if value not in seen:
                valid.append(value)
                seen.add(value)
                if len(valid) == desired_count:
                    break

    return np.asarray(valid[:desired_count], dtype=np.int64)
