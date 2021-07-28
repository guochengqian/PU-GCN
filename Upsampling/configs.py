import argparse
import os
import time
import uuid
import logging
import pathlib
import shutil
import numpy as np


def str2bool(x):
    return x.lower() in 'true'


def generate_exp_directory(FLAGS):
    """
    Helper function to create checkpoint folder. We save
    model checkpoints using the provided model directory
    but we add a sub-folder for each separate experiment:
    """

    experiment_string = '_'.join([FLAGS.jobname, time.strftime('%Y%m%d-%H%M%S'), str(uuid.uuid4())])

    pathlib.Path(FLAGS.log_dir).mkdir(parents=True, exist_ok=True)
    if FLAGS.phase == 'train':
        FLAGS.log_dir = os.path.join(FLAGS.log_dir, experiment_string)
        FLAGS.code_dir = os.path.join(FLAGS.log_dir, "code")
        pathlib.Path(FLAGS.log_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(FLAGS.code_dir).mkdir(parents=True, exist_ok=True)
        # ===> save scripts
        shutil.copytree('Common', os.path.join(FLAGS.code_dir, 'Common'))
        shutil.copytree('tf_lib', os.path.join(FLAGS.code_dir, 'tf_lib'))
        shutil.copytree('Upsampling', os.path.join(FLAGS.code_dir, 'Upsampling'))


def configure_logger(FLAGS):
    """
    Configure logger on given level. Logging will occur on standard
    output and in a log file saved in model_dir.
    """
    log_format = logging.Formatter('%(asctime)s %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if FLAGS.phase == 'train':
        file_handler = logging.FileHandler(os.path.join(FLAGS.log_dir,
                                                        '{}.log'.format(os.path.basename(FLAGS.log_dir))))
    else:
        file_handler = logging.FileHandler(os.path.join(FLAGS.log_dir, 'evaluation.log'))

    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())


parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='train', help="train/test")
parser.add_argument('--seed', default=2, type=int, help="deterministic mode")

# log
parser.add_argument('--log_dir', default='log')

# data
parser.add_argument('--data_dir', default='./data/PU1K/pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5')
parser.add_argument('--no_augment', action="store_false", dest="augment", default=True)
parser.add_argument('--test_jitter', action='store_true', help='test with noise')
parser.add_argument('--jitter_sigma', type=float, default=0.01, help="jitter augmentation")
parser.add_argument('--jitter_max', type=float, default=0.03, help="jitter augmentation")
parser.add_argument('--num_point', type=int, default=256)
parser.add_argument('--patch_num_point', type=int, default=256)
parser.add_argument('--patch_num_ratio', type=int, default=3)
parser.add_argument('--fps', dest='random', action='store_false', default=True,
                    help='use random input, or farthest sample input(default)')

# model
parser.add_argument('--model', default='pugcn', help='model name: pugan, mpu, pugcn')
parser.add_argument('--up_ratio', type=int, default=4)
parser.add_argument('--more_up', type=int, default=0)
parser.add_argument('--block', default='inception',
                    help="dense, inception, inception_v0. [default: inception]")
parser.add_argument('--n_blocks', type=int, default=2, help="number of inception dense blocks [default: 2]")
parser.add_argument('--channels', type=int, default=32, help="number of channel size")
parser.add_argument('--d', type=int, default=2, help="dilation rate")
parser.add_argument('--k', type=int, default=20, help="number of neighbors (kernel size for GCN)")
parser.add_argument('--upsampler', default='nodeshuffle', type=str,
                    help="upsampler, multi_cnn, clone, duplicate, nodeshuffle")
parser.add_argument('--use_att', action='store_true', help="use attention in upsampling unit")
parser.add_argument('--no_global_pooling', action="store_false", dest="use_global_pooling", default=True)

# training
parser.add_argument('--restore', type=str, default=None,
                    help='set to the path of pretrained model directory if want to resume training')
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--base_lr_d', type=float, default=0.0001)
parser.add_argument('--base_lr_g', type=float, default=1e-3)
parser.add_argument('--beta', type=float, default=0.9)
parser.add_argument('--start_decay_step', type=int, default=50000)
parser.add_argument('--lr_decay_steps', type=int, default=50000)
parser.add_argument('--lr_decay_rate', type=float, default=0.7)
parser.add_argument('--lr_clip', type=float, default=1e-6)
parser.add_argument('--steps_per_print', type=int, default=100)

parser.add_argument('--vis', action='store_true')
parser.add_argument('--steps_per_visu', type=int, default=100)
parser.add_argument('--epoch_per_save', type=int, default=5)

# loss
parser.add_argument('--loss_type', default='cd', type=str, help="cd, emd")
parser.add_argument('--cd_threshold', default=0., type=float, help="threshold for the modified chamfer distance")
parser.add_argument('--fidelity_w', default=1.0, type=float, help="fidelity_weight")
parser.add_argument('--reg', action='store_true', default=False,
                    help='use regularization in loss')
parser.add_argument('--repulse', action='store_true',
                    help='use repulse loss (default: false)')
parser.add_argument('--repulsion_w', default=0.01, type=float, help="repulsion_weight")
parser.add_argument('--uniform', action='store_true', default=False,
                    help='use uniform loss')
parser.add_argument('--uniform_w', default=0.1, type=float, help="uniform_weight")
parser.add_argument('--use_gan', action='store_true')
parser.add_argument('--gan_w', default=0.5, type=float, help="gan_weight")
parser.add_argument('--gen_update', default=1, type=int, help="gen_update. set to 2 if use_gan, else 1")

FLAGS = parser.parse_args()
if FLAGS.seed < 0:  # enable random seed
    FLAGS.seed = np.random.randint(0, 1000)

FLAGS.dataset = os.path.basename(FLAGS.data_dir).split('_')[0].lower()
FLAGS.jobname = '{}_{}_{}_{}_n{}_C{}_d{}_k{}_{}_{}_{}_{}_sample-{}_lr{}_cd{}_SRx{}_moreupx{}_seed{}' \
    .format(FLAGS.dataset, FLAGS.model, FLAGS.upsampler, FLAGS.block,
            FLAGS.n_blocks, FLAGS.channels, FLAGS.d, FLAGS.k,
            'gan' if FLAGS.use_gan else 'wogan',
            'repulse' if FLAGS.repulse else 'worepulse',
            'uniform' if FLAGS.uniform else 'wouniform',
            'reg' if FLAGS.reg else 'woreg',
            'random' if FLAGS.random else 'FPS',
            FLAGS.base_lr_g, FLAGS.cd_threshold, FLAGS.up_ratio, FLAGS.more_up, FLAGS.seed)

if FLAGS.restore is None:
    generate_exp_directory(FLAGS)
else:
    FLAGS.log_dir = FLAGS.restore

configure_logger(FLAGS)
logging.info("========== Loading Config =============")
logging.info("Log file save to: %s \n", FLAGS.log_dir)
logging.info("========== Finish Config Loading =============\n")

