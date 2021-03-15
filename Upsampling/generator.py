# -*- coding: utf-8 -*-
# @Description :
# @Author      : Guocheng Qian
# @Email       : guocheng.qian@kaust.edu.sa

import tensorflow as tf
from Common import ops
from tf_ops.sampling.tf_sampling import gather_point, farthest_point_sample
from Common.pointnet_util import pointnet_sa_module, pointnet_fp_module


class PUGAN(object):
    """
    [PU-GAN](https://arxiv.org/abs/1907.10844)
    """
    def __init__(self, opts, is_training, name="Generator"):
        self.opts = opts
        self.is_training = is_training
        self.name = name
        self.reuse = False
        self.num_point = self.opts.patch_num_point
        self.up_ratio = self.opts.up_ratio
        self.up_ratio_real = self.up_ratio + self.opts.more_up
        self.out_num_point = int(self.num_point * self.up_ratio)

    def __call__(self, inputs):
        channels = 24  # 24 for PU-GAN
        with tf.variable_scope(self.name, reuse=self.reuse):
            features = ops.feature_extraction(inputs,
                                              channels,
                                              scope='feature_extraction', is_training=self.is_training,
                                              bn_decay=None)
            H = ops.up_projection_unit(features, self.up_ratio_real,
                                       scope="up_projection_unit",
                                       is_training=self.is_training, bn_decay=None)
            coord = ops.conv2d(H, 64, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=self.is_training,
                               scope='fc_layer1', bn_decay=None)

            coord = ops.conv2d(coord, 3, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=self.is_training,
                               scope='fc_layer2', bn_decay=None,
                               activation_fn=None, weight_decay=0.0)
            outputs = tf.squeeze(coord, [2])

            if self.up_ratio_real > self.up_ratio:
                outputs = gather_point(outputs, farthest_point_sample(self.out_num_point, outputs))
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        return outputs


class MPU(object):
    """
    [MPU (3PU)](https://arxiv.org/abs/1811.11286)
    """
    def __init__(self, opts, is_training, name="Generator"):
        self.opts = opts
        self.is_training = is_training
        self.name = name
        self.reuse = False
        self.num_point = self.opts.patch_num_point
        self.up_ratio = self.opts.up_ratio
        self.out_num_point = int(self.num_point * self.up_ratio)

    def __call__(self, inputs):
        channels = 12   # 12 for MPU
        with tf.variable_scope(self.name, reuse=self.reuse):
            features = ops.feature_extraction(inputs,
                                              channels,
                                              scope='feature_extraction', is_training=self.is_training,
                                              bn_decay=None)

            H = ops.up_unit(features, self.up_ratio,
                            self.opts.upsampler,
                            scope="up_block",
                            is_training=self.is_training, bn_decay=None)

            coord = ops.conv2d(H, 64, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=self.is_training,
                               scope='fc_layer1', bn_decay=None,
                               activation_fn=tf.nn.leaky_relu
                               )

            coord = ops.conv2d(coord, 3, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=self.is_training,
                               scope='fc_layer2', bn_decay=None,
                               activation_fn=None, weight_decay=0.0)
            outputs = tf.squeeze(coord, [2])
            outputs += tf.reshape(tf.tile(tf.expand_dims(inputs, 2), [1, 1, self.up_ratio, 1]),
                                  [inputs.shape[0], self.num_point * self.up_ratio, -1])  # B, N, 4, 3

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        return outputs


class PUNET(object):
    """
    PU-Net:
    https://arxiv.org/abs/1801.06761
    """
    def __init__(self, opts, is_training, name="Generator"):
        self.opts = opts
        self.is_training = is_training
        self.name = name
        self.reuse = False
        self.num_point = self.opts.patch_num_point
        self.up_ratio = self.opts.up_ratio
        self.out_num_point = int(self.num_point * self.up_ratio)

    def __call__(self, inputs, bradius):
        num_point = inputs.get_shape()[1].value
        l0_xyz = inputs[:, :, 0:3]
        l0_normals = None  # do not use normals
        use_bn = False
        use_ibn = False
        bn_decay = None
        is_training = self.is_training

        with tf.variable_scope(self.name, reuse=self.reuse):
            # Layer 1
            l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_normals, npoint=num_point,
                                                               radius=bradius * 0.05,
                                                               bn=use_bn, ibn=use_ibn,
                                                               nsample=32, mlp=[32, 32, 64], mlp2=None, group_all=False,
                                                               is_training=is_training, bn_decay=bn_decay,
                                                               scope='layer1')

            l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=num_point / 2,
                                                               radius=bradius * 0.1, bn=use_bn, ibn=use_ibn,
                                                               nsample=32, mlp=[64, 64, 128], mlp2=None,
                                                               group_all=False,
                                                               is_training=is_training, bn_decay=bn_decay,
                                                               scope='layer2')

            l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=num_point / 4,
                                                               radius=bradius * 0.2, bn=use_bn, ibn=use_ibn,
                                                               nsample=32, mlp=[128, 128, 256], mlp2=None,
                                                               group_all=False,
                                                               is_training=is_training, bn_decay=bn_decay,
                                                               scope='layer3')

            l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=num_point / 8,
                                                               radius=bradius * 0.3, bn=use_bn, ibn=use_ibn,
                                                               nsample=32, mlp=[256, 256, 512], mlp2=None,
                                                               group_all=False,
                                                               is_training=is_training, bn_decay=bn_decay,
                                                               scope='layer4')

            # Feature Propagation layers
            up_l4_points = pointnet_fp_module(l0_xyz, l4_xyz, None, l4_points, [64], is_training, bn_decay,
                                              scope='fa_layer1', bn=use_bn, ibn=use_ibn)

            up_l3_points = pointnet_fp_module(l0_xyz, l3_xyz, None, l3_points, [64], is_training, bn_decay,
                                              scope='fa_layer2', bn=use_bn, ibn=use_ibn)

            up_l2_points = pointnet_fp_module(l0_xyz, l2_xyz, None, l2_points, [64], is_training, bn_decay,
                                              scope='fa_layer3', bn=use_bn, ibn=use_ibn)

            concat_feat = tf.concat([up_l4_points, up_l3_points, up_l2_points, l1_points, l0_xyz], axis=-1)
            concat_feat = tf.expand_dims(concat_feat, axis=2)

            # concat feature
            if self.opts.upsampler == 'original':
                with tf.variable_scope('up_layer', reuse=tf.AUTO_REUSE):
                    new_points_list = []
                    for i in range(self.up_ratio):
                        concat_feat = ops.conv2d(concat_feat, 256, [1, 1],
                                                 padding='VALID', stride=[1, 1],
                                                 bn=False, is_training=is_training,
                                                 scope='fc_layer0_%d' % (i), bn_decay=bn_decay)

                        new_points = ops.conv2d(concat_feat, 128, [1, 1],
                                                padding='VALID', stride=[1, 1],
                                                bn=use_bn, is_training=is_training,
                                                scope='conv_%d' % (i),
                                                bn_decay=bn_decay)
                        new_points_list.append(new_points)
                    net = tf.concat(new_points_list, axis=1)
            else:
                net = ops.up_unit(concat_feat, self.up_ratio,
                                  self.opts.upsampler,
                                  scope="up_block",
                                  use_att=self.opts.use_att,
                                  is_training=self.is_training, bn_decay=None)

            # get the xyz
            coord = ops.conv2d(net, 64, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=self.is_training,
                               scope='fc_layer1', bn_decay=None,
                               activation_fn=tf.nn.leaky_relu
                               )

            coord = ops.conv2d(coord, 3, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=self.is_training,
                               scope='fc_layer2', bn_decay=None,
                               activation_fn=None, weight_decay=0.0)
            coord = tf.squeeze(coord, [2])  # B*(2N)*3

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        return coord


class PUGCN(object):
    """
    PU-GCN: Point Cloud Upsampling using Graph Convolutional Networks
    https://arxiv.org/abs/1912.03264.pdf
    """

    def __init__(self, opts, is_training, name="Generator"):
        self.opts = opts
        self.is_training = is_training
        self.name = name
        self.reuse = False
        self.num_point = self.opts.patch_num_point
        self.up_ratio = self.opts.up_ratio
        self.up_ratio_real = self.up_ratio + self.opts.more_up
        self.out_num_point = int(self.num_point * self.up_ratio)

    def __call__(self, inputs):
        with tf.variable_scope(self.name, reuse=self.reuse):
            features, idx = ops.feature_extractor(inputs,
                                                  self.opts.block, self.opts.n_blocks,
                                                  self.opts.channels, self.opts.k, self.opts.d,
                                                  use_global_pooling=self.opts.use_global_pooling,
                                                  scope='feature_extraction', is_training=self.is_training,
                                                  bn_decay=None)

            H = ops.up_unit(features, self.up_ratio_real,
                            self.opts.upsampler,
                            k=self.opts.k,
                            idx=idx,
                            scope="up_block",
                            use_att=self.opts.use_att,
                            is_training=self.is_training, bn_decay=None)

            coord = ops.conv2d(H, 32, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=self.is_training,
                               scope='fc_layer1', bn_decay=None,
                               activation_fn=tf.nn.leaky_relu
                               )

            coord = ops.conv2d(coord, 3, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=self.is_training,
                               scope='fc_layer2', bn_decay=None,
                               activation_fn=None, weight_decay=0.0)
            outputs = tf.squeeze(coord, [2])

            if self.up_ratio_real > self.up_ratio:
                outputs = gather_point(outputs, farthest_point_sample(self.out_num_point, outputs))
            outputs += tf.reshape(tf.tile(tf.expand_dims(inputs, 2), [1, 1, self.up_ratio, 1]),
                                  [inputs.shape[0], self.num_point * self.up_ratio, -1])  # B, N, 4, 3

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        return outputs

