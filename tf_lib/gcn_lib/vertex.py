"""
PU-GCN: Point Cloud upsampling using Graph Convolutional Networks.
"""

import tensorflow as tf
from .edge import get_graph_features, dyn_dil_get_graph_feature, knn, dil_knn
from ..common import conv2d
import numpy as np


def edge_conv(x,
              out_channels=64,
              idx=None, k=16, d=1,
              n_layers=1,
              scope='edge_conv',
              **kwargs):
    """
    Modified EdgeConv. This is the default GCN used in our PU-GCN
    :param x: input features
    :param idx: edge index for neighbors and centers (if None, has to use KNN to build a graph at first)
    :param out_channels: output channel,
    :param n_layers: number of GCN layers.
    :param scope: the name
    :param kwargs:
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if idx is None:
            idx = dil_knn(x, k, d)
        central, neighbors = get_graph_features(x, idx)
        message = conv2d(neighbors - central, out_channels, [1, 1], padding='VALID',
                         scope='message', use_bias=True, activation_fn=None, **kwargs)
        x_center = conv2d(x, out_channels, [1, 1], padding='VALID',
                          scope='central', use_bias=False, activation_fn=None, **kwargs)
        edge_features = x_center + message
        edge_features = tf.nn.relu(edge_features)
        for i in range(n_layers - 1):
            edge_features = conv2d(edge_features, out_channels, [1, 1], padding='VALID',
                                   scope='l%d' % i, **kwargs)
        y = tf.reduce_max(edge_features, axis=-2, keepdims=True)
    return y


def densegcn(x,
             idx=None,
             growth_rate=12, n_layers=3, k=16, d=1,
             return_idx=False,
             scope='densegcn',
             **kwargs):
    """
    Modified dense EdgeConv used in PU-GCN:
    :param x: input features
    :param idx: edge index for neighbors and centers (if None, has to use KNN to build a graph at first)
    :param growth_rate: output channel of each path,
    :param n_layers: number of GCN layers.
    :param k: the kernel size (num of neighbors used in each path)
    :param d: dilation rate of each path
    :param scope: the name
    :param kwargs:
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if idx is None:
            central, neighbors, idx = dyn_dil_get_graph_feature(x, k, d)
        else:
            central, neighbors = get_graph_features(x, idx)

        message = conv2d(neighbors - central, growth_rate, [1, 1], padding='VALID',
                         scope='message', use_bias=True, activation_fn=None, **kwargs)
        x_center = conv2d(x, growth_rate, [1, 1], padding='VALID',
                          scope='central', use_bias=False, activation_fn=None, **kwargs)
        edge_features = x_center + message
        y = tf.nn.relu(edge_features)

        features = [y]
        for i in range(n_layers - 1):
            if i == 0:  # actually this layer is added by mistake.
                y = conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l_%d' % i, **kwargs)
            else:
                y = tf.concat(features, axis=-1)
            features.append(conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l_%d' % i, **kwargs))

        if n_layers > 1:
            y = tf.concat(features, axis=-1)
        y = tf.reduce_max(y, axis=-2, keepdims=True)

        if return_idx:
            return y, idx
        else:
            return y


def gcn(x,
        idx=None,
        growth_rate=12, n_layers=3, k=16, d=1,
        return_idx=False,
        scope='densegcn',
        **kwargs):
    """
    Ablation for densegcn used in PU-GCN. Here we use the same layers of edgeconvs but no dense connetions are used.
    :param x: input features
    :param idx: edge index for neighbors and centers (if None, has to use KNN to build a graph at first)
    :param growth_rate: output channel of each path,
    :param n_layers: number of GCN layers.
    :param k: the kernel size (num of neighbors used in each path)
    :param d: dilation rate of each path
    :param scope: the name
    :param kwargs:
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if idx is None:
            central, neighbors, idx = dyn_dil_get_graph_feature(x, k, d)
        else:
            central, neighbors = get_graph_features(x, idx)

        message = conv2d(neighbors - central, growth_rate, [1, 1], padding='VALID',
                         scope='message', use_bias=True, activation_fn=None, **kwargs)
        x_center = conv2d(x, growth_rate, [1, 1], padding='VALID',
                          scope='central', use_bias=False, activation_fn=None, **kwargs)
        edge_features = x_center + message
        y = tf.nn.relu(edge_features)

        y = conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l_%d' % 0, **kwargs)
        for i in range(n_layers - 1):
            y = conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l_%d' % i, **kwargs)
        y = tf.reduce_max(y, axis=-2, keepdims=True)

        if return_idx:
            return y, idx
        else:
            return y


# Inception DenseGCN used in PU-GCN
# The default Inception module used in PU-GCN
# -------------------------------------------------
def inception_densegcn(x,
                       growth_rate=12,
                       k=16, d=2, n_dense=3,
                       use_global_pooling=True,
                       use_residual=True,
                       use_dilation=True,
                       scope='inception_resgcn',
                       **kwargs):
    """
    Inception Residual DenseGCN used in PU-GCN
    :param x: input features
    :param growth_rate: output channel of each path,
    :param k: the kernel size (num of neighbors used in each path)
    :param d: dilation rate of each path (default: 2)
    :param n_dense: number of layers in each denseGCN block, default is 3
    :param scope: the name
    :param kwargs:
    :return:
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        idx = knn(x, k=k * d)  # [B N K 2]
        idx1 = idx[:, :, :k, :]
        if use_dilation:
            idx2 = idx[:, :, ::d, :]
        else:
            idx2 = idx
        inception_reduction = conv2d(x, growth_rate, 1,
                                     padding='VALID', scope='inception_1_reduction', **kwargs)
        inception_1 = densegcn(inception_reduction, idx1, growth_rate, n_layers=n_dense,
                               scope='inception_1', **kwargs)
        inception_2 = densegcn(inception_reduction, idx2, growth_rate, n_layers=n_dense,
                               scope='inception_2', **kwargs)

        if use_global_pooling:
            inception_3 = tf.reduce_max(x, axis=-1, keepdims=True)
            inception_out = tf.concat([inception_1, inception_2, inception_3], axis=-1)
        else:
            inception_out = tf.concat([inception_1, inception_2], axis=-1)

        if use_residual and x.get_shape()[-1] == inception_out.get_shape()[-1]:
            inception_out = inception_out + x
    return inception_out


# ablating the inception module in Inception DenseGCN
# only use 1 densegcn block
def inception_1densegcn(x,
                        growth_rate=12,
                        k=16, d=2, n_dense=3,
                        use_global_pooling=True,
                        use_residual=True,
                        use_dilation=True,
                        scope='inception_resgcn',
                        **kwargs):
    """
    Inception Residual DenseGCN used in PU-GCN
    :param x: input features
    :param growth_rate: output channel of each path,
    :param k: the kernel size (num of neighbors used in each path)
    :param d: dilation rate of each path (default: 2)
    :param n_dense: number of layers in each denseGCN block, default is 3
    :param scope: the name
    :param kwargs:
    :return:
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        idx = knn(x, k=k)  # [B N K 2]

        inception_reduction = conv2d(x, growth_rate, 1,
                                     padding='VALID', scope='inception_1_reduction', **kwargs)
        inception_1 = densegcn(inception_reduction, idx, growth_rate, n_layers=n_dense,
                               scope='inception_1', **kwargs)

        if use_global_pooling:
            inception_3 = tf.reduce_max(x, axis=-1, keepdims=True)
            inception_out = tf.concat([inception_1, inception_3], axis=-1)
        else:
            inception_out = inception_1

        if use_residual and x.get_shape()[-1] == inception_out.get_shape()[-1]:
            inception_out = inception_out + x
    return inception_out


# for ablating the DenseGCN in Inception DenseGCN for PU-GCN
# -------------------------------------------------
def inceptiongcn(x,
                 growth_rate=12,
                 k=16, d=2, n_dense=3,
                 use_global_pooling=True,
                 use_residual=True,
                 use_dilation=True,
                 scope='inception_resgcn',
                 **kwargs):
    """
    Inception Residual DenseGCN used in PU-GCN
    :param x: input features
    :param growth_rate: output channel of each path,
    :param k: the kernel size (num of neighbors used in each path)
    :param d: dilation rate of each path (default: 2)
    :param n_dense: number of layers in each denseGCN block, default is 3
    :param scope: the name
    :param kwargs:
    :return:
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        idx = knn(x, k=k * d)  # [B N K 2]
        idx1 = idx[:, :, :k, :]
        if use_dilation:
            idx2 = idx[:, :, ::d, :]
        else:
            idx2 = idx
        inception_reduction = conv2d(x, growth_rate, 1,
                                     padding='VALID', scope='inception_1_reduction', **kwargs)
        inception_1 = gcn(inception_reduction, idx1, growth_rate, n_layers=n_dense,
                          scope='inception_1', **kwargs)
        inception_2 = gcn(inception_reduction, idx2, growth_rate, n_layers=n_dense,
                          scope='inception_2', **kwargs)

        if use_global_pooling:
            inception_3 = tf.reduce_max(x, axis=-1, keepdims=True)
            inception_out = tf.concat([inception_1, inception_2, inception_3], axis=-1)
        else:
            inception_out = tf.concat([inception_1, inception_2], axis=-1)

        if use_residual and x.get_shape()[-1] == inception_out.get_shape()[-1]:
            inception_out = inception_out + x
    return inception_out


def point_shuffler(inputs, scale=2):
    """
    Periodic shuffling layer for point cloud
    """
    outputs = tf.reshape(inputs, [tf.shape(inputs)[0], tf.shape(inputs)[1], 1, tf.shape(inputs)[3] // scale, scale])
    outputs = tf.transpose(outputs, [0, 1, 4, 3, 2])
    outputs = tf.reshape(outputs, [tf.shape(inputs)[0], tf.shape(inputs)[1] * scale, 1, tf.shape(inputs)[3] // scale])
    return outputs


def nodeshuffle(x, scale=2,
                k=16, d=1, channels=64,
                idx=None,
                scope='nodeshuffle', **kwargs):
    """
    NodeShuffle upsampling module proposed in PU-GCN
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        y = conv2d(x, channels, 1,
                   padding='VALID', scope='up_reduction', **kwargs)
        y = edge_conv(y,
                      channels * scale,
                      idx,
                      n_layers=1,
                      scope='edge_conv',
                      **kwargs)
        y = point_shuffler(y, scale)
        return y


def mlpshuffle(x, scale=2,
               channels=64,
               scope='mlpshuffle', **kwargs):
    """
    the ablation for NodeShuffle upsampling module proposed in PU-GCN
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        y = conv2d(x, channels, 1,
                   padding='VALID', scope='up_reduction', **kwargs)

        # MLPs
        y = conv2d(y, channels, 1,
                   padding='VALID', scope='conv1', **kwargs)
        y = conv2d(y, channels, 1,
                   padding='VALID', scope='conv2', **kwargs)
        y = conv2d(y, channels, 1,
                   padding='VALID', scope='conv3', **kwargs)
        y = conv2d(y, channels, 1,
                   padding='VALID', scope='conv4', **kwargs)
        y = conv2d(y, channels * scale, 1,
                   padding='VALID', scope='conv5', **kwargs)

        y = point_shuffler(y, scale)
        return y


def multi_cnn(x, scale=2, scope='multi_cnn',
              **kwargs):
    """
    Our implementation of Multi-branch CNN used in PU-Net Paper
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        new_points_list = []
        n_channels = x.get_shape().as_list()[-1]
        for i in range(scale):
            branch_feat = conv2d(x, n_channels, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 use_bn=False,
                                 scope='branch_%d' % (i), **kwargs)
            new_points_list.append(branch_feat)
        out = tf.concat(new_points_list, axis=1)
    return out


def gen_grid(num_grid_point):
    """
    generate unique indicator for duplication based upsampling module.
    output [num_grid_point, 2]
    """
    x = tf.lin_space(-0.2, 0.2, num_grid_point)
    x, y = tf.meshgrid(x, x)
    grid = tf.reshape(tf.stack([x, y], axis=-1), [-1, 2])  # [2, 2, 2] -> [4, 2]
    return grid


def duplicate(x, scale, scope='duplicate',
              **kwargs):
    """
    Our implementation of duplicate-based upsampling module used in PU-Net Paper
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        batch_size, n_points, _, n_channels = x.get_shape().as_list()

        grid = gen_grid(np.round(np.sqrt(scale)).astype(np.int32))
        grid = tf.tile(tf.expand_dims(grid, 0),
                       [batch_size, n_points, 1])
        grid = tf.expand_dims(grid, axis=-2)

        x = tf.reshape(
            tf.tile(tf.expand_dims(x, 2), [1, 1, scale, 1, 1]),
            [batch_size, n_points * scale, 1, n_channels])
        x = tf.concat([x, grid], axis=-1)
        x = conv2d(x, 128, [1, 1],
                   padding='VALID', stride=[1, 1],
                   scope='up_layer1',
                   **kwargs)
        x = conv2d(x, 128, [1, 1],
                   padding='VALID', stride=[1, 1],
                   scope='up_layer2',
                   **kwargs)
    return x
