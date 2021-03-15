#!/usr/bin/env bash
#/bin/bash
system=$1
tf_lib=$2
tf_inc=$3
cuda_lib=$4
cuda_inc=$5

nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -I $tf_inc -std=c++11

if [ "$system" == "linux" ]; then
    g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I $tf_inc  -I $cuda_inc -I $tf_inc/external/nsync/public -lcudart -L $cuda_lib -L$tf_lib -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
elif [ "$system" == "centos" ]; then
    g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I $tf_inc  -I $cuda_inc -I $tf_inc/external/nsync/public -lcudart -L $cuda_lib -L$tf_lib -ltensorflow_framework -O2
else
    echo "unsupported system"
fi
