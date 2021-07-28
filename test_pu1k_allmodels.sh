#!/usr/bin/env bash
conda activate pugcn

### result of
source test_pu1k.sh pretrain/pu1k-pugcn/ 0 1 --model pugcn --k 20
#source test_pu1k.sh pretrain/pu1k-punet/ 0 1 --model punet --upsampler original
#source test_pu1k.sh pretrain/pu1k-mpu/ 0 1 --model mpu --upsampler duplicate

### ablation for upsampling
#source test_pu1k.sh pretrain/pu1k-pugcn-duplicate/ 0 1 --model pugcn --upsampler duplicate --k 20
#source test_pu1k.sh pretrain/pu1k-punet-nodeshuffle/ 0 1 --model punet --upsampler nodeshuffle
#source test_pu1k.sh pretrain/pu1k-mpu-nodeshuffle/ 0 1 --model mpu --upsampler nodeshuffle


## ablation for inception DenseGCN
#source test_pu1k.sh pretrain/pu1k-pugcn-dense/ 0 1 --model pugcn --upsampler nodeshuffle --block dense --k 20
#source test_pu1k.sh pretrain/pu1k-pugcn-n1/ 0 1 --model pugcn --upsampler nodeshuffle --n_blocks 1 --k 20
#source test_pu1k.sh pretrain/pu1k-pugcn-nopool/ 0 1 --model pugcn --upsampler nodeshuffle --n_blocks 2 --k 20 --no_global_pooling
#source test_pu1k.sh pretrain/pu1k-pugcn-nodil/ 0 1 --model pugcn --upsampler nodeshuffle --n_blocks 2 --k 20 --block inception_nodil
#source test_pu1k.sh pretrain/pu1k-pugcn-nores/ 0 1 --model pugcn --upsampler nodeshuffle --n_blocks 2 --k 20 --block inception_nores
#source test_pu1k.sh pretrain/pu1k-pugcn-edgeconv/ 0 1 --model pugcn --upsampler nodeshuffle
