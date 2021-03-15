import tensorflow as tf
from Upsampling.model import Model
from Upsampling.configs import FLAGS, configure_logger
import os
import logging
import shutil
import numpy as np


def run():
    if FLAGS.phase == 'train':
        FLAGS.train_file = os.path.join(FLAGS.data_dir)
        logging.info('train_file: {}'.format(FLAGS.train_file))
    else:
        FLAGS.test_data = os.path.join(FLAGS.data_dir, '*.xyz')
        FLAGS.out_folder = os.path.join("evaluation_code/result")
        if os.path.exists(FLAGS.out_folder):
            shutil.rmtree(FLAGS.out_folder)
        os.makedirs(FLAGS.out_folder)
        logging.info('test_data: {}'.format(FLAGS.test_data))
        logging.info('checkpoints:'.format(FLAGS.log_dir))

    logging.info('loading config: \n {} \n'.format(FLAGS))
    # open session
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    with tf.Session(config=run_config) as sess:
        model = Model(FLAGS, sess)
        if FLAGS.phase == 'train':
            model.train()
        else:
            model.test()


def main(unused_argv):
    run()


if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    np.random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)
    logging.info('setting random seed to: {}'.format(FLAGS.seed))
    tf.app.run()
