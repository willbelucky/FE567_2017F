# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 11. 22.
"""
import argparse
import os
import sys

import tensorflow as tf

from dnn.classifying_dnn import run_training
from dnn.data_reader import read_data

FLAGS = None


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training(flags=FLAGS,
                 data_sets=read_data(FLAGS.sector_name, file_dir=FLAGS.file_dir,
                                     true_adjusting_rate=FLAGS.true_adjusting_rate, lasso_applied=FLAGS.lasso_applied))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # total, IT, consumegoods, energy, essentialconsumgoods, indusgoods, material, medical, unclassified
    parser.add_argument(
        '--sector_name',
        type=str,
        default='total',
        help='Name of the sector.'
    )
    parser.add_argument(
        '--lasso_applied',
        type=bool,
        default=True,
        help='Apply lasso or not.'
    )
    parser.add_argument(
        '--true_adjusting_rate',
        type=float,
        default=0.04,
        help='Adjust the portion of true data of train data.'
             'If true_adjusting_rate is None, true_adjusting_rate is not applied.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0001,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.5,
        help='The dropout rate, between 0 and 1. E.g. "rate=0.1" would drop out 10% of input units.'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=2000,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--hidden_units',
        nargs='+',
        type=int,
        default=[256, 256],
        help='Number of units in hidden layers.'
    )
    parser.add_argument(
        '--file_dir',
        type=str,
        default=os.getcwd().replace(chr(92), '/') + '/',
        help='Directory to put the source file.'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', os.getcwd()),
                             'images/'),
        help='Directory to put the image.'
    )
    parser.add_argument(
        '--excel_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', os.getcwd()),
                             'excels/'),
        help='Directory to put the excel file.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', os.getcwd()),
                             'logs/fully_connected_feed'),
        help='Directory to put the log data.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
