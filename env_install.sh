#!/usr/bin/env bash

# Environment Installnation is the hardest part in playing with our code.

# Step0: install Anaconda3
#cd ~/
#wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh
#bash Anaconda3-2019.07-Linux-x86_64.sh

# Step1: install pugcn environment
#conda remove --name pugcn --all
conda create -n pugcn python=3.6.8 cudatoolkit=10.0 cudnn numpy=1.16
conda activate pugcn
pip install matplotlib tensorflow-gpu==1.13.1 open3d==0.9 sklearn Pillow gdown plyfile
# please do not install tensorflow gpu by conda. It may effect the following compiling.


# Step2: compile tf_ops (this is the hardest part and a lot of people encounter different problems here.)
# you may need some compiling help from here: https://github.com/yulequan/PU-Net
cd tf_ops
bash compile.sh linux # please look at this file in detail if it does not work
cd ..


# Step 3 Optional (compile evaluation code)
# check evaluation_code/compile.sh
