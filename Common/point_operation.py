# -*- coding: utf-8 -*-
# @Time        : 16/1/2019 5:26 PM
# @Description :
# @Author      : li rui hui
# @Email       : ruihuili@gmail.com
# @File        : point_operation.py

import numpy as np


def nonuniform_sampling(num=4096, sample_num=1024):
    sample = set()
    loc = np.random.rand() * 0.8 + 0.1
    while len(sample) < sample_num:
        a = int(np.random.normal(loc=loc, scale=0.3) * num)
        if a < 0 or a >= num:
            continue
        sample.add(a)
    return list(sample)


def shuffle_point_cloud_and_gt(batch_data, batch_gt=None):
    B, N, C = batch_data.shape

    idx = np.arange(N)
    np.random.shuffle(idx)
    batch_data = batch_data[:, idx, :]
    if batch_gt is not None:
        np.random.shuffle(idx)
        batch_gt = batch_gt[:, idx, :]
    return batch_data, batch_gt


def rotate_point_cloud_and_gt(batch_data, batch_gt=None, z_rotated=False):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    batch_size = batch_data.shape[0]
    angles = np.random.uniform(size=(batch_size, 3)) * 2 * np.pi
    cos_x, cos_y, cos_z = np.split(np.cos(angles), indices_or_sections=3, axis=1)
    sin_x, sin_y, sin_z = np.split(np.sin(angles), indices_or_sections=3, axis=1)
    ones = np.ones((batch_size, 1))
    zeros = np.zeros((batch_size, 1))

    Rzs = np.stack(
        (
            np.concatenate((cos_z, -sin_z, zeros), axis=1),
            np.concatenate((sin_z, cos_z, zeros), axis=1),
            np.concatenate((zeros, zeros, ones), axis=1)
        ), axis=1)

    if z_rotated:
        rotation_matrix = Rzs
    else:
        Rxs = np.stack(
            (
                np.concatenate((ones, zeros, zeros), axis=1),
                np.concatenate((zeros, cos_x, -sin_x), axis=1),
                np.concatenate((zeros, sin_x, cos_x), axis=1)
            ), axis=1)
        Rys = np.stack(
            (
                np.concatenate((cos_y, zeros, sin_y), axis=1),
                np.concatenate((zeros, ones, zeros), axis=1),
                np.concatenate((-sin_y, zeros, cos_y), axis=1)
            ), axis=1)
        rotation_matrix = np.einsum("imj, ijk, ikl -> iml", Rzs, Rys, Rxs)

    batch_data[..., 0:3] = np.einsum("ijk, ikl -> ijl", batch_data[..., 0:3], rotation_matrix)
    if batch_gt is not None:
        batch_gt[..., 0:3] = np.einsum("ijk, ikl -> ijl", batch_gt[..., 0:3], rotation_matrix)
    return batch_data, batch_gt


def jitter_perturbation_point_cloud(batch_data, sigma=0.005, clip=0.02):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(* batch_data.shape), -1 * clip, clip)
    jittered_data[:, :, 3:] = 0
    jittered_data += batch_data
    return jittered_data


def shift_point_cloud_and_gt(batch_data, batch_gt=None, shift_range=0.3):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
    for batch_index in range(B):
        batch_data[batch_index, :, 0:3] += shifts[batch_index, 0:3]

    if batch_gt is not None:
        for batch_index in range(B):
            batch_gt[batch_index, :, 0:3] += shifts[batch_index, 0:3]

    return batch_data, batch_gt


def random_scale_point_cloud_and_gt(batch_data, batch_gt=None, scale_low=0.5, scale_high=2):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, (B, 1, 1))
    batch_data[..., 0:3] = np.multiply(batch_data[..., 0:3], scales)

    if batch_gt is not None:
        batch_gt[..., 0:3] = np.multiply(batch_gt[..., 0:3], scales)

    return batch_data, batch_gt, scales.squeeze()


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.03, angle_clip=0.09):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        batch_data[k, ..., 0:3] = np.dot(batch_data[k, ..., 0:3].reshape((-1, 3)), R)
        if batch_data.shape[-1] > 3:
            batch_data[k, ..., 3:] = np.dot(batch_data[k, ..., 3:].reshape((-1, 3)), R)

    return batch_data


def guass_noise_point_cloud(batch_data, sigma=0.005, mu=0.00):
    """ Add guassian noise in per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    batch_data += np.random.normal(mu, sigma, batch_data.shape)
    return batch_data
