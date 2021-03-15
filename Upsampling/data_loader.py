# -*- coding: utf-8 -*-
# @Time        : 16/1/2019 5:11 PM
# @Description :
# @Author      : li rui hui
# @Email       : ruihuili@gmail.com
# @File        : data_loader.py

import numpy as np
import h5py
import queue
import threading
from Common import point_operation
import logging


def normalize_point_cloud(input):
    if len(input.shape) == 2:
        axis = 0
    elif len(input.shape) == 3:
        axis = 1
    centroid = np.mean(input, axis=axis, keepdims=True)
    input = input - centroid
    furthest_distance = np.amax(np.sqrt(np.sum(input ** 2, axis=-1)), axis=axis, keepdims=True)
    input = input / furthest_distance
    return input, centroid, furthest_distance


def batch_sampling(input_data, num):
    B, N, C = input_data.shape
    out_data = np.zeros([B, num, C])
    for i in range(B):
        idx = np.arange(N)
        np.random.shuffle(idx)
        idx = idx[:num]
        out_data[i, ...] = input_data[i, idx]
    return out_data


def load_h5_data(h5_filename='', opts=None, skip_rate=1, use_randominput=True):
    logging.info("========== Loading Data ==========")
    num_point = opts.num_point
    num_4X_point = int(opts.num_point * 4)
    num_out_point = int(opts.num_point * opts.up_ratio)

    logging.info("loading data from: {}".format(h5_filename))
    if use_randominput:
        logging.info("use random input")
        with h5py.File(h5_filename, 'r') as f:
            input = f['poisson_%d' % num_4X_point][:]
            gt = f['poisson_%d' % num_out_point][:]
    else:
        logging.info("Do not use random input")
        with h5py.File(h5_filename, 'r') as f:
            input = f['poisson_%d' % num_point][:]
            gt = f['poisson_%d' % num_out_point][:]

    # name = f['name'][:]
    assert len(input) == len(gt)

    logging.info("Normalize the data")
    data_radius = np.ones(shape=(len(input)))
    centroid = np.mean(input[:, :, 0:3], axis=1, keepdims=True)
    input[:, :, 0:3] = input[:, :, 0:3] - centroid
    furthest_distance = np.amax(np.sqrt(np.sum(input[:, :, 0:3] ** 2, axis=-1)), axis=1, keepdims=True)
    input[:, :, 0:3] = input[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)

    gt[:, :, 0:3] = gt[:, :, 0:3] - centroid
    gt[:, :, 0:3] = gt[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)

    input = input[::skip_rate]
    gt = gt[::skip_rate]
    data_radius = data_radius[::skip_rate]
    logging.info("total %d samples" % (len(input)))

    logging.info("========== Finish Data Loading ========== \n")
    return input, gt, data_radius


class Fetcher(threading.Thread):
    def __init__(self, opts):
        super(Fetcher, self).__init__()
        self.queue = queue.Queue(50)
        self.stopped = False
        self.opts = opts
        self.use_random_input = self.opts.random
        self.input_data, self.gt_data, self.radius_data = load_h5_data(self.opts.train_file, opts=self.opts,
                                                                       use_randominput=self.use_random_input)
        self.batch_size = self.opts.batch_size
        self.sample_cnt = self.input_data.shape[0]
        self.patch_num_point = self.opts.patch_num_point
        self.num_batches = self.sample_cnt // self.batch_size
        logging.info("NUM_BATCH is %s" % self.num_batches)

    def run(self):
        while not self.stopped:
            idx = np.arange(self.sample_cnt)
            np.random.shuffle(idx)
            self.input_data = self.input_data[idx, ...]
            self.gt_data = self.gt_data[idx, ...]
            self.radius_data = self.radius_data[idx, ...]

            for batch_idx in range(self.num_batches):
                if self.stopped:
                    return None
                start_idx = batch_idx * self.batch_size
                end_idx = (batch_idx + 1) * self.batch_size
                batch_input_data = self.input_data[start_idx:end_idx, :, :].copy()
                batch_data_gt = self.gt_data[start_idx:end_idx, :, :].copy()
                radius = self.radius_data[start_idx:end_idx].copy()

                if self.use_random_input:
                    new_batch_input = np.zeros((self.batch_size, self.patch_num_point, batch_input_data.shape[2]))
                    for i in range(self.batch_size):
                        idx = point_operation.nonuniform_sampling(self.input_data.shape[1],
                                                                  sample_num=self.patch_num_point)
                        new_batch_input[i, ...] = batch_input_data[i][idx]
                    batch_input_data = new_batch_input
                if self.opts.augment:
                    batch_input_data = point_operation.jitter_perturbation_point_cloud(batch_input_data,
                                                                                       sigma=self.opts.jitter_sigma,
                                                                                       clip=self.opts.jitter_max)
                    batch_input_data, batch_data_gt = point_operation.rotate_point_cloud_and_gt(batch_input_data,
                                                                                                batch_data_gt)
                    batch_input_data, batch_data_gt, scales = point_operation.random_scale_point_cloud_and_gt(
                        batch_input_data,
                        batch_data_gt,
                        scale_low=0.8,
                        scale_high=1.2)
                    radius = radius * scales
                self.queue.put((batch_input_data[:, :, :3], batch_data_gt[:, :, :3], radius))
        return None

    def fetch(self):
        if self.stopped:
            return None
        return self.queue.get()

    def shutdown(self):
        self.stopped = True
        logging.info("Shutdown .....")
        while not self.queue.empty():
            self.queue.get()
        logging.info("Remove all queue data")
