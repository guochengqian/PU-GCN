# PU-GCN: Point Cloud Upsampling using Graph Convolutional Networks (CVPR21')
This is the official implementation for paper [PU-GCN: Point Cloud Upsampling using Graph Convolutional Networks](https://arxiv.org/abs/1912.03264.pdf). We get accepted to CVPR21'. 

PU-GCN repo supports training our PU-GCN, and previous methods [PU-Net](https://arxiv.org/abs/1801.06761), [MPU (3PU)](https://arxiv.org/abs/1811.11286), [PU-GAN](https://arxiv.org/abs/1907.10844). Please kindly cite all of the methods. 

 
### Installation
This repository is based on Tensorflow (1.13.1) and the TF operators from PointNet++. Therefore, you need to install tensorflow and compile the TF operators. 

You can check the `env_install.sh` for details how to install the environment.  In the second step, for compiling TF operators, please check `compile.sh` in `tf_ops` folder, one has to manually change the path!!. 


### Usage

1. Clone the repository:

   ```shell
   https://github.com/guochengqian/PU-GCN.git
   cd PU-GCN
   ```
   
2. install the environment
   Once you have modified the path in `compile.sh` under `tf_ops`, you can simply install `pugcn` environment by:
   
   ```bash
    source env_install.sh
    conda activate pugcn
   ```
   
3. Download PU1K dataset  

4. Train models
    
   -  PU-GCN
   ```shell
   python main.py --phase train --model pugcn --upsampler nodeshuffle --k 20 --data_dir /data/PUGCN/PU1K/pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5
   ```
   
   -  PU-Net
   ```
   python main.py --phase train --model punet --upsampler original  --data_dir /data/PUGCN/PU1K/pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5
   ```
   
   -  mpu
   ```
   python main.py --phase train --model mpu --upsampler duplicate --data_dir /data/pugcn/PUGAN/train/PUGAN_poisson_256_poisson_1024.h5
   ```

   -  PU-GAN
   ```
   python main.py --phase train --model pugan --more_up 2 --data_dir /data/pugcn/PUGAN/train/PUGAN_poisson_256_poisson_1024.h5
   ```
   
4. Evaluate models:  
   Download the pretrained models here.
   
    Then run:
   ```shell
   source test_pu1k_allmodels.sh # please look this file and `test_pu1k.sh` for details
   ```

5. Test on real-scanned dataset

    ```bash
    source test_realscan_allmodels.sh
    ```

6. Visualization. 
    check below. You have to modify the path inside. 
    ```bash
    python vis_benchmark.py
    ```
    
## Citation

If PU-GCN and the repo are useful for your research, please consider citing:

    @article{Qian2019PUGCNPC,
      title={PU-GCN: Point Cloud Upsampling using Graph Convolutional Networks},
      author={Guocheng Qian and Abdulellah Abualshour and G. Li and A. Thabet and Bernard Ghanem},
      journal={ArXiv},
      year={2019},
      volume={abs/1912.03264}
    }
    
    @article{Yu2018PUNetPC,
      title={PU-Net: Point Cloud Upsampling Network},
      author={Lequan Yu and Xianzhi Li and Chi-Wing Fu and D. Cohen-Or and P. Heng},
      journal={2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year={2018},
      pages={2790-2799}
    }

    @article{Wang2019PatchBasedP3,
      title={Patch-Based Progressive 3D Point Set Upsampling},
      author={Yifan Wang and Shihao Wu and Hui Huang and D. Cohen-Or and O. Sorkine-Hornung},
      journal={2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2019},
      pages={5951-5960}
    }
    
    @inproceedings{li2019pugan,
         title={PU-GAN: a Point Cloud Upsampling Adversarial Network},
         author={Li, Ruihui and Li, Xianzhi and Fu, Chi-Wing and Cohen-Or, Daniel and Heng, Pheng-Ann},
         booktitle = {{IEEE} International Conference on Computer Vision ({ICCV})},
         year = {2019}
     }

    
### Acknowledgement
This repo is heavily built on [PU-GAN code](https://github.com/liruihui/PU-GAN). We also borrow something from MPU and PU-Net. 


