#!/usr/bin/env bash

## please be patient whn complie tf_ops. This is the hardest part in reproducing our results.

# Using different system may require different code for compiling. But is is not always correct.
# Try both to compile.
# If both of them does not work, make sure tf_lib, tf_inc, cuda_lib all correct and $nvcc is the same version as CUDA


# eg. Below is the env I used for the remote cluster
#module load cuda/10.0.130 # uncomment this line if not using the remote cluster

# If below does not work, you can try to set system='linux'
system=$1 # linux or centos.
    #  usually set to linux, if does not work, try change to centos.
    #  difference is we comment -D_GLIBCXX_USE_CXX11_ABI=0 out in centos

tf_lib=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
tf_inc=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
cuda_dir="/usr/local/cuda-10.0"
cuda_lib="${cuda_dir}/lib64"
cuda_inc="${cuda_dir}/include"

echo $system
echo "===> tf_lib is located at: ${tf_lib}"
echo "===> tf_inc is located at: ${tf_inc}"
echo "===> cuda_dir is located at: ${cuda_dir}"
echo "===> change the location of them if wrong"
echo "current working dir is: $(pwd)"

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

