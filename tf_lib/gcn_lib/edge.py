"""
PU-GCN: Point Cloud upsampling using Graph Convolutional Networks.
"""

import tensorflow as tf
from tf_ops.grouping.tf_grouping import knn_point_2


def knn(x, k=16, self_loop=False):
    """Construct edge feature for each point
    Args:
        x: (batch_size, num_points, num_dims)
        k: int
        self_loop: include the key (center point) or not?
    Returns:
        edge idx: (batch_size, num_points, k, num_dims)
    """
    if len(x.get_shape())>3:
        x = tf.squeeze(x, axis=2)
    _, idx = knn_point_2(k + 1, x, x, unique=True, sort=True)

    # this is only a naive version of self_loop implementation.
    if not self_loop:
        idx = idx[:, :, 1:, :]
    else:
        idx = idx[:, :, 0:-1, :]
    return idx


def get_graph_features(x, idx, return_central=True):
    """
    get the features for the neighbors and center points from the x and inx
    :param x: input features
    :param idx: the index for the neighbors and center points
    :return: 
    """
    if len(x.get_shape())>3:
        x = tf.squeeze(x, axis=2)
    pc_neighbors = tf.gather_nd(x, idx)
    if return_central:
        pc_central = tf.tile(tf.expand_dims(x, axis=-2), [1, 1, idx.shape[2], 1])
        return pc_central, pc_neighbors
    else:
        return pc_neighbors


def dil_knn(x, k=16, d=1, use_fsd=False):
    if len(x.get_shape()) > 3:
        x = tf.squeeze(x, axis=2)
    idx = knn(x, k=k*d)  # [B N K 2]
    if d > 1:
        if use_fsd:
            idx = idx[:, :, k*(d-1):k*d, :]
        else:
            idx = idx[:, :, ::d, :]
    return idx


def dyn_dil_get_graph_feature(x, k=16, d=1, use_fsd=False, return_central=True):
    """
    dynamically get the feature of the dilated GCN
    :param x: input feature
    :param k: number of neighbors
    :param d: dilation rate
    :param use_fsd: farthest point sampling, default False. Use uniform sampling
    :return: central feature, neighbors feature, edge index
    """
    idx = dil_knn(x, k, d, use_fsd)
    if return_central:
        central, neighbors = get_graph_features(x, idx, True)
        return central, neighbors, idx
    else:
        neighbors = get_graph_features(x, idx, False)
        return neighbors, idx


