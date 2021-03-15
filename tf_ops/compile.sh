#!/usr/bin/env bash

## please be patient whn complie tf_ops. This is the hardest part in reproducing our results.

# Using different system may require different code for compiling. But is is not always correct.
# Try both to compile.
# If both of them does not work, make sure tf_lib, tf_inc, cuda_lib all correct and $nvcc is the same version as CUDA


# eg. Below is the env I used for the remote cluster
module load cuda/10.0.130
# If below does not work, you can try to set system='linux'
#system="centos" #
system="linux" # or centos, difference is we comment -D_GLIBCXX_USE_CXX11_ABI=0 out in centos
# below xxx is your account name. you may have to change the path to your own.
tf_lib="/home/xxxx/anaconda3/envs/pugcn/lib/python3.6/site-packages/tensorflow"
tf_inc="/home/xxxx/anaconda3/envs/pugcn/lib/python3.6/site-packages/tensorflow/include"
cuda_lib="/home/xxxx/local/cuda-10.0/lib64"
cuda_inc="/home/xxxx/local/cuda-10.0/include"
cuda_dir='/home/xxxx/local/cuda-10.0'



export LD_LIBRARY_PATH=$tf_lib:$LD_LIBRARY_PATH # you may need add this into path

# make sure you have NVCC and the environment ready (\eg, make sure you: conda activate pugcn)
cd nn_distance
source tf_nndistance_compile.sh $system $tf_lib $tf_inc $cuda_lib $cuda_inc
cd ../approxmatch/
source tf_approxmatch_compile.sh $system $tf_lib $tf_inc $cuda_lib $cuda_inc
cd ../grouping/
source tf_grouping_compile.sh $system $tf_lib $tf_inc $cuda_lib $cuda_inc
cd ../interpolation/
source tf_interpolate_compile.sh $system $tf_lib $tf_inc $cuda_lib $cuda_inc
cd ../renderball
source compile_render_balls_so.sh
cd ../sampling
source tf_sampling_compile.sh $system $tf_lib $tf_inc $cuda_lib $cuda_inc
cd ..

