import numpy as np


def hausdorff_from_nn_distances(forward_distances, backward_distances):
    """Compute symmetric Hausdorff from directional nearest-neighbor distances.

    ``tf_nndistance.nn_distance`` returns one nearest-neighbor distance per
    point for each direction. The symmetric Hausdorff distance is the maximum of
    the two directional maxima, not the sum of the two maxima.
    """
    forward = np.asarray(forward_distances)
    backward = np.asarray(backward_distances)
    if forward.size == 0 or backward.size == 0:
        raise ValueError("Hausdorff distance requires non-empty directional distances")
    return float(max(np.max(forward), np.max(backward)))
