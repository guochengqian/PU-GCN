#!/usr/bin/env bash
conda activate pugcn

cur_dir=$PWD
echo "current project director is: $cur_dir"
log_folder=$1
is_folder=$2 # set to 1, if you wish to test all the experiment files inside a folder; else set to 0, test only one experiment file
PY_ARGS=${@:3}
in_data_dir=/data/PUGCN/PU1K/test/input_2048/input_2048
gt_data_dir=/data/PUGCN/PU1K/test/input_2048/gt_8192


if [ $is_folder == 1 ]
then
    for logdir in $log_folder/*; do
        echo "===> test the ckpt from ${logdir}"
        echo ;
        CUDA_VISIBLE_DEVICES=1 python main.py --phase test --restore ${logdir} --data_dir  ${in_data_dir} ${PY_ARGS}

        cd evaluation_code
        source eval_pu1k.sh
        cd ..
        CUDA_VISIBLE_DEVICES=1 python evaluate.py --pred evaluation_code/result/ --gt ${gt_data_dir} --save_path ${logdir}

    done
else
    logdir=$log_folder
    echo "===> test the ckpt from ${logdir}"
    echo ;
    CUDA_VISIBLE_DEVICES=1 python main.py --phase test --restore ${logdir} --data_dir  ${in_data_dir} ${PY_ARGS}

#    # save results.
#    rm -rf ${logdir}/result-pu1k
#    cp -r $cur_dir/evaluation_code/result/ ${logdir}/result-pu1k
#
    # evaluation
    cd evaluation_code
    source eval_pu1k.sh
    cd ..
    CUDA_VISIBLE_DEVICES=1 python evaluate.py --pred evaluation_code/result/ --gt ${gt_data_dir} --save_path ${logdir}

fi
