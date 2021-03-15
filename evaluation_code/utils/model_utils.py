import os
import sys
import tensorflow as tf
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)
from tf_ops.nn_distance import tf_nndistance
from tf_ops.grouping.tf_grouping import query_ball_point, group_point, knn_point_2
from tf_ops.sampling.tf_sampling import gather_point, farthest_point_sample
import numpy as np


def covariance_matrix(pc):
    """
    :param pc [B, P, K, 3]
    :return barycenter [B, P, 1, 3]
            covariance [B, P, 3, 3]
    """
    with tf.name_scope("covariance_matrix"):
        barycenter = tf.reduce_mean(pc, axis=2, keepdims=True)
        # B, P, K, 3
        pc -= barycenter
        return barycenter, tf.matmul(pc, pc, transpose_a=True)  # B, P, 3, 3


def exponential_distance(query, points, scope="exponential_distance"):
    """
    return B, P, K, 1
    """
    with tf.name_scope(scope):
        # query_normalized = query / tf.sqrt(tf.reduce_sum(query ** 2, axis=-1, keepdims=True))
        # points_normalized = points / tf.sqrt(tf.reduce_sum(points ** 2, axis=-1, keepdims=True))
        # N, P, K, 1
        distance = tf.reduce_sum((query - points) ** 2, axis=-1, keepdims=True)
        # distance = tf.Print(distance, [distance], message="fm_distance")
        # mean(min(over K) over P)
        h = tf.reduce_mean(tf.reduce_min(distance, axis=2, keepdims=True), axis=1, keepdims=True)
        # h = tf.Print(h, [h])
        return distance, tf.exp(-distance / (h / 2))


def extract_patch_for_next_level(batch_xyz, k, batch_features=None, gt_xyz=None, gt_k=None, is_training=True):
    """
    :param batch_xyz [B, P, 3]
    """
    batch_size, num_point, _ = batch_xyz.shape.as_list()
    with tf.name_scope("extract_input"):
        if is_training:
            # B, 1, 3
            idx = tf.random_uniform([batch_size, 1], minval=0, maxval=num_point, dtype=tf.int32)
            # idx = tf.constant(250, shape=[batch_size, 1], dtype=tf.int32)
            batch_seed_point = gather_point(batch_xyz, idx)
            patch_num = 1
        else:
            assert (batch_size == 1)
            # remove residual, (B P 1) and (B, P, 1, 2)
            closest_d, _ = knn_point_2(2, batch_xyz, batch_xyz, unique=False)
            closest_d = closest_d[:, :, 1:]
            # (B, P)
            mask = tf.squeeze(closest_d < 5 * (tf.reduce_mean(closest_d, axis=1, keepdims=True)), axis=-1)
            # filter (B, P', 3)
            batch_xyz = tf.expand_dims(tf.boolean_mask(batch_xyz, mask), axis=0)
            # batch_xyz = tf.Print(batch_xyz, [tf.shape(batch_xyz)])
            # B, M, 3
            # batch_seed_point = batch_xyz[:, -1:, :]
            # patch_num = 1
            patch_num = int(num_point / k * 5)
            # idx = tf.random_uniform([batch_size, patch_num], minval=0, maxval=num_point, dtype=tf.int32)
            idx = tf.squeeze(farthest_point_sample(patch_num, batch_xyz), axis=0)
            # idx = tf.random_uniform([patch_num], minval=0, maxval=tf.shape(batch_xyz)[1], dtype=tf.int32)
            # B, P, 3 -> B, k, 3 (idx B, k, 1)
            # idx = tf.Print(idx, [idx], message="idx")
            batch_seed_point = tf.gather(batch_xyz, idx, axis=1)
            k = tf.minimum(k, tf.shape(batch_xyz)[1])
            # batch_seed_point = gather_point(batch_xyz, idx)
        # B, M, k, 2
        _, new_patch_idx = knn_point_2(k, batch_xyz, batch_seed_point, unique=False)
        # B, M, k, 3
        batch_xyz = tf.gather_nd(batch_xyz, new_patch_idx)
        # MB, k, 3
        batch_xyz = tf.concat(tf.unstack(batch_xyz, axis=1), axis=0)
    if batch_features is not None:
        with tf.name_scope("extract_feature"):
            batch_features = tf.gather_nd(batch_features, new_patch_idx)
            batch_features = tf.concat(tf.unstack(batch_features, axis=1), axis=0)
    if is_training and (gt_xyz is not None and gt_k is not None):
        with tf.name_scope("extract_gt"):
            _, new_patch_idx = knn_point_2(gt_k, gt_xyz, batch_seed_point, unique=False)
            gt_xyz = tf.gather_nd(gt_xyz, new_patch_idx)
            gt_xyz = tf.concat(tf.unstack(gt_xyz, axis=1), axis=0)
    else:
        gt_xyz = None

    return batch_xyz, batch_features, gt_xyz


def gen_grid(num_grid_point):
    """
    output [num_grid_point, 2]
    """
    x = tf.lin_space(-0.2, 0.2, num_grid_point)
    x, y = tf.meshgrid(x, x)
    grid = tf.reshape(tf.stack([x, y], axis=-1), [-1, 2])  # [2, 2, 2] -> [4, 2]
    return grid


def gen_1d_grid(num_grid_point):
    """
    output [num_grid_point, 2]
    """
    x = tf.lin_space(-0.2, 0.2, num_grid_point)
    grid = tf.reshape(x, [-1, 1])  # [2, 2, 2] -> [4, 2]
    return grid


def pre_load_checkpoint(checkpoint_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        # print(" [*] Reading checkpoint from {}".format(ckpt.model_checkpoint_path))
        try:
            epoch_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        except Exception:
            epoch_step = 0
        return epoch_step, ckpt.model_checkpoint_path
    else:
        return 0, None


def get_repulsion_loss(pred, nsample=20, radius=0.07, knn=False, use_l1=False, h=0.001):
    # # pred: (batch_size, npoint,3)
    # idx, pts_cnt = query_ball_point(radius, nsample, pred, pred)
    # tf.summary.histogram('smooth/unque_index', pts_cnt)

    # grouped_pred = group_point(pred, idx)  # (batch_size, npoint, nsample, 3)
    # grouped_pred -= tf.expand_dims(pred, 2)

    # # get the uniform loss
    # h = 0.03
    # dist_square = tf.reduce_sum(grouped_pred ** 2, axis=-1)
    # dist_square, idx = tf.nn.top_k(-dist_square, 5)
    # dist_square = -dist_square[:, :, 1:]  # remove the first one
    # dist_square = tf.maximum(1e-12,dist_square)
    # dist = tf.sqrt(dist_square)
    # weight = tf.exp(-dist_square/h**2)
    # uniform_loss = tf.reduce_mean(radius-dist*weight)
    # return uniform_loss
    # pred: (batch_size, npoint,3)
    if knn:
        _, idx = knn_point_2(nsample, pred, pred)
        pts_cnt = tf.constant(nsample, shape=(30, 1024))
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, pred, pred)
    tf.summary.histogram('smooth/unque_index', pts_cnt)

    grouped_pred = group_point(pred, idx)  # (batch_size, npoint, nsample, 3)
    grouped_pred -= tf.expand_dims(pred, 2)

    # get the uniform loss
    if use_l1:
        dists = tf.reduce_sum(tf.abs(grouped_pred), axis=-1)
    else:
        dists = tf.reduce_sum(grouped_pred ** 2, axis=-1)

    val, idx = tf.nn.top_k(-dists, 5)
    val = val[:, :, 1:]  # remove the first one

    if use_l1:
        h = np.sqrt(h) * 2
    print(("h is ", h))

    val = tf.maximum(0.0, h + val)  # dd/np.sqrt(n)
    repulsion_loss = tf.reduce_mean(val)
    return repulsion_loss


def get_cd_loss(pred, gt, radius, forward_weight=1.0, threshold=None):
    """
    pred: BxNxC,
    label: BxN,
    forward_weight: relative weight for forward_distance
    """
    with tf.name_scope("cd_loss"):
        dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(gt, pred)
        if threshold is not None:
            forward_threshold = tf.reduce_mean(dists_forward, keepdims=True, axis=1) * threshold
            backward_threshold = tf.reduce_mean(dists_backward, keepdims=True, axis=1) * threshold
            # only care about distance within threshold (ignore strong outliers)
            dists_forward = tf.where(dists_forward < forward_threshold, dists_forward, tf.zeros_like(dists_forward))
            dists_backward = tf.where(dists_backward < backward_threshold, dists_backward,
                                      tf.zeros_like(dists_backward))
        # dists_forward is for each element in gt, the closest distance to this element
        dists_forward = tf.reduce_mean(dists_forward, axis=1)
        dists_backward = tf.reduce_mean(dists_backward, axis=1)
        CD_dist = forward_weight * dists_forward + dists_backward
        # CD_dist_norm = CD_dist/radius
        cd_loss = tf.reduce_mean(CD_dist)
        return cd_loss, None


def get_hausdorff_loss(pred, gt, radius, forward_weight=1.0, threshold=None):
    """
    pred: BxNxC,
    label: BxN,
    forward_weight: relative weight for forward_distance
    """
    with tf.name_scope("cd_loss"):
        dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(gt, pred)
        # only care about distance within threshold (ignore strong outliers)
        if threshold is not None:
            dists_forward = tf.where(dists_forward < threshold, dists_forward, tf.zeros_like(dists_forward))
            dists_backward = tf.where(dists_backward < threshold, dists_backward, tf.zeros_like(dists_backward))
        # dists_forward is for each element in gt, the closest distance to this element
        dists_forward = tf.reduce_max(dists_forward, axis=1)
        dists_backward = tf.reduce_max(dists_backward, axis=1)
        CD_dist = forward_weight * dists_forward + dists_backward
        CD_dist_norm = CD_dist / radius
        cd_loss = tf.reduce_max(CD_dist_norm)
        return cd_loss, None


def get_cd_losses(preds, gt, radius, weights):
    losses = []
    for pred, weight in zip(preds, weights):
        loss, _ = get_cd_loss(pred, gt, radius)
        loss = weight * loss
        losses.append(loss)

    return losses, None


if __name__ == '__main__':
    import utils.pc_util as pc_util
    from glob import glob

    gt_files = glob("../data/test_data/sketchfab/poisson_10000/*.xyz")
    pc_files = glob("../data/test_data/sketchfab/poisson_2500/*.xyz")
    is_2D = False
    gt = []
    pc = []
    for b in range(4):
        gt.append(pc_util.load(gt_files[b])[np.newaxis, :, :3])
        pc.append(pc_util.load(pc_files[b])[np.newaxis, :, :3])

    import pdb

    pdb.set_trace()
    gt = np.concatenate(gt, axis=0)
    pc = np.concatenate(pc, axis=0)

    # fetcher = Fetcher(input_data, label, radius, batch_size=10,
    #     step_ratio=4, up_ratio=16, num_in_point=1024)
    gt = tf.constant(gt, dtype=tf.float32)
    pred = tf.constant(pc, dtype=tf.float32)
    # covariance matrix
    _, idx = knn_point_2(5, gt, pred)
    # [B, P, k, 3]
    grouped = tf.gather_nd(gt, idx)
    # B, P, 1, 3 and B, P, 3, 3
    b, cov_mat = covariance_matrix(grouped)
    # eigenvalue non-decreasing B, P, 3  B, P, 3, 3
    e, v = tf.self_adjoint_eig(cov_mat)
    # normals B, P, 3
    if is_2D:
        normals = v[:, :, :, 1]
    else:
        normals = v[:, :, :, 0]
    normals = tf.nn.l2_normalize(normals, axis=-1)

    rel_pred = pred - tf.squeeze(b, axis=2)
    # projection B, P
    projection = tf.reduce_sum(tf.abs(rel_pred * normals), axis=-1)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        normals_output, loss_output = sess.run([normals, projection])

    for b in range(4):
        # normalize normal B, P, 3
        pc_util.save_ply_property(pc[b], loss_output[b], "./normal_test_%d.ply" % b, normals=normals_output[b],
                                  cmap_name="jet")
