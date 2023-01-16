# PU-GCN: Point Cloud Upsampling using Graph Convolutional Networks (CVPR21')
[CVPR21](https://openaccess.thecvf.com/content/CVPR2021/html/Qian_PU-GCN_Point_Cloud_Upsampling_Using_Graph_Convolutional_Networks_CVPR_2021_paper.html) | [Arxiv](https://arxiv.org/abs/1912.03264.pdf) | [project](https://www.deepgcns.org/app/pu-gcn) | [code](https://github.com/guochengqian/PU-GCN) | [PU1K data](https://drive.google.com/drive/folders/1k1AR_oklkupP8Ssw6gOrIve0CmXJaSH3?usp=sharing)

This is the official implementation for our CVPR 21' paper [PU-GCN: Point Cloud Upsampling using Graph Convolutional Networks](https://arxiv.org/abs/1912.03264.pdf). This repository supports training our PU-GCN, and previous methods [PU-Net](https://arxiv.org/abs/1801.06761), [MPU (3PU)](https://arxiv.org/abs/1811.11286), [PU-GAN](https://arxiv.org/abs/1907.10844). 



### Update
* 2021/08/28: provide pretrained model. fix evaluation bug. add more tf_ops compilation instructions.


### Preparation

1. Clone the repository:

   ```shell
   https://github.com/guochengqian/PU-GCN.git
   cd PU-GCN
   ```
   
2. install the environment
   Once you have modified the path in `compile.sh` under `tf_ops`, you can simply install `pugcn` environment by:  
   
   ```bash
    bash env_install.sh
    conda activate pugcn
   ```
   
   Note this repository is based on Tensorflow (1.13.1) and the TF operators from PointNet++.  You can check the `env_install.sh` for details how to install the environment.  In the second step, for compiling TF operators, please check `compile.sh` in `tf_ops` folder, one may have to manually change the path!!
   
3. Download PU1K dataset from [Google Drive](https://drive.google.com/file/d/1oTAx34YNbL6GDwHYL2qqvjmYtTVWcELg/view?usp=sharing)  
    Link the data to `./data`:

    ```bash
    mkdir data
    ln -s /path/to/PU1K ./data/
    ```
4. Optional. The original meshes of PU1K dataset is avaialble in [Goolge Drive](https://drive.google.com/file/d/1tnMjJUeh1e27mCRSNmICwGCQDl20mFae/view?usp=sharing)
    
### Train on PU1K (Random input) 

**note**: If you favor uniform inputs, you have to retrain all models. Otherwise, the results might be really bad. To train with uniform inputs, simply add `--fps` in the command line below.
We provide the **pretrained PU-GCN on PU-GAN's dataset using the uniform inputs** [here](https://drive.google.com/file/d/1xdG3hUomPoUhdusuYjHqCyl8YMBwYrZg/view?usp=share_link) in case it is needed. 

To train models on PU1K using **random inputs**. Our pretrained models (PU-GCN on PU1K random and other models) are available [Google Drive](https://drive.google.com/file/d/1vusBIw7sd69gnyaeoWMiGaPHfkyHM5Qb/view?usp=sharing)

To train on other dataset, simply change the `--data_dir` to locate to your data file. 

-  PU-GCN
    ```shell
    python main.py --phase train --model pugcn --upsampler nodeshuffle --k 20 
    ```

-  PU-Net
    ```
    python main.py --phase train --model punet --upsampler original  
    ```

-  MPU
    ```
    python main.py --phase train --model mpu --upsampler duplicate 
    ```

-  PU-GAN
    ```
    python main.py --phase train --model pugan --more_up 2 
    ```



### Evaluation

1. Test on PU1K dataset
   ```bash
   bash test_pu1k_allmodels.sh # please modify this script and `test_pu1k.sh` if needed
   ```

5. Test on real-scanned dataset

    ```bash
    bash test_realscan_allmodels.sh
    ```

6. Visualization. 
    check below. You have to modify the path inside. 
    
    ```bash
    python vis_benchmark.py
    ```
    



## Citation

If PU-GCN and the repo are useful for your research, please consider citing:

    @InProceedings{Qian_2021_CVPR,
        author    = {Qian, Guocheng and Abualshour, Abdulellah and Li, Guohao and Thabet, Ali and Ghanem, Bernard},
        title     = {PU-GCN: Point Cloud Upsampling Using Graph Convolutional Networks},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2021},
        pages     = {11683-11692}
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


​    
### Acknowledgement
This repo is heavily built on [PU-GAN code](https://github.com/liruihui/PU-GAN). We also borrow the architecture and evaluation codes from MPU and PU-Net. 


