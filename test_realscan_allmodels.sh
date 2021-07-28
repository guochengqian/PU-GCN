#!/usr/bin/env bash
conda activate pugcn

# kitti
bash test_realscan.sh pretrain/pugan-punet/ ./data/real_scan_kitti_pugcn 0 --model punet --upsampler original
bash test_realscan.sh pretrain/pugan-mpu/  ./data/real_scan_kitti_pugcn 0 --model mpu --upsampler duplicate
bash test_realscan.sh pretrain/pugan-pugan/ ./data/real_scan_kitti_pugcn 0 --model pugan --more_up 2
bash test_realscan.sh pretrain/pugan-pugc/  ./data/real_scan_kitti_pugcn 0 --model pugcn --k 20
