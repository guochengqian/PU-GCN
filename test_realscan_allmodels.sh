#!/usr/bin/env bash
conda activate pugcn

# kitti
source test_realscan.sh pretrain/pugan-punet/ /data/pugcn/real_scan_kitti_pugcn --model punet --upsampler original
source test_realscan.sh pretrain/pugan-mpu/  /data/pugcn/real_scan_kitti_pugcn --model mpu --upsampler duplicate
source test_realscan.sh pretrain/pugan-pugan/ /data/pugcn/real_scan_kitti_pugcn --model pugan --more_up 2
source test_realscan.sh pretrain/pugan-pugc/  /data/pugcn/real_scan_kitti_pugcn --model pugcn --k 20
