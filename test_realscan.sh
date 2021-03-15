#!/usr/bin/env bash
conda activate pugcn

cur_dir=$PWD
logdir=$1
datadir=$2
PY_ARGS=${@:3}

CUDA_VISIBLE_DEVICES=1 python main.py --phase test --restore $logdir --data_dir  $datadir ${PY_ARGS}
mkdir $logdir/result-realscan
cp -r $cur_dir/evaluation_code/result/ $logdir/result-realscan
