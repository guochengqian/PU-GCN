#!/usr/bin/env bash
conda activate pugcn

cur_dir=$PWD
logdir=$1
datadir=$2
GPU=$3
PY_ARGS=${@:4}

CUDA_VISIBLE_DEVICES=${GPU} python main.py --phase test --restore $logdir --data_dir  $datadir ${PY_ARGS}
mkdir $logdir/result-realscan
cp -r $cur_dir/evaluation_code/result/ $logdir/result-realscan
