import numpy as np

from Common.metrics import hausdorff_from_nn_distances


def test_hausdorff_uses_max_of_directional_maxima_not_sum():
    forward = np.array([1.0, 4.0, 9.0])
    backward = np.array([16.0, 25.0])

    assert hausdorff_from_nn_distances(forward, backward) == 25.0


def test_hausdorff_rejects_empty_inputs():
    try:
        hausdorff_from_nn_distances([], [1.0])
    except ValueError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("expected ValueError")
