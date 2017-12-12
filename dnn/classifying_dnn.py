# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 11. 26.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def inference(units, hidden_units, column_number, class_number, dropout=None):
    """Build the mnist_example model up to where it may be used for inference.

    Args:
      units: Units placeholder, from inputs().
      hidden_units: Size of the hidden layers.
      column_number: Size of the input columns.
      class_number: Size of the output classes.
      dropout: The dropout rate, between 0 and 1. E.g. "rate=0.1" would drop out 10% of input units.

    Returns:
      softmax_linear: Output tensor with the computed logits.
    """
    pre_unit = column_number
    pre_layer = units
    hidden_layer = None
    for i, hidden_unit in enumerate(hidden_units):
        # Hidden n
        with tf.name_scope('hidden{}'.format(i + 1)):
            weights = tf.Variable(
                tf.truncated_normal([pre_unit, hidden_unit],
                                    stddev=1.0 / math.sqrt(float(pre_unit))),
                name='weights')
            biases = tf.Variable(tf.zeros([hidden_unit]),
                                 name='biases')
            hidden_layer = tf.nn.relu(tf.matmul(pre_layer, weights) + biases)
            if dropout is not None:
                hidden_layer = tf.layers.dropout(hidden_layer, rate=dropout, training=True)

            pre_unit = hidden_unit
            pre_layer = hidden_layer
    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([pre_unit, class_number],
                                stddev=1.0 / math.sqrt(float(pre_unit))),
            name='weights')
        biases = tf.Variable(tf.zeros([class_number]),
                             name='biases')
        logits = tf.matmul(hidden_layer, weights) + biases
    return logits


def do_loss(logits, labels):
    """Calculates the loss from the logits and the labels.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size].

    Returns:
      loss: Loss tensor of type float.
    """
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss, learning_rate):
    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.

    Returns:
      train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, targets):
    """Evaluate the quality of the logits at predicting the label.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      targets: Targets tensor, int32 - [batch_size], with values in the
        range [0, 2).

    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    correct = tf.nn.in_top_k(logits, targets, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))


def placeholder_inputs(batch_size, column_number):
    """Generate placeholder variables to represent the input tensors.
    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.
    Args:
      batch_size: The batch size will be baked into both placeholders.
      column_number: Size of the input columns.
    Returns:
      units_placeholder: Units placeholder.
      labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # unit and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    units_placeholder = tf.placeholder(tf.float32, shape=(batch_size, column_number))
    labels_placeholder = tf.placeholder(tf.int32, shape=batch_size)
    return units_placeholder, labels_placeholder


def fill_feed_dict(data_set, units_pl, labels_pl):
    """Fills the feed_dict for training the given step.
    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }
    Args:
      data_set: The set of units and labels, from data.read_data_sets()
      units_pl: The units placeholder, from placeholder_inputs().
      labels_pl: The labels placeholder, from placeholder_inputs().
    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    units_feed, labels_feed = data_set.next_batch()
    feed_dict = {
        units_pl: units_feed,
        labels_pl: labels_feed,
    }
    return feed_dict, labels_feed


def do_eval(sess,
            logits,
            units_placeholder,
            labels_placeholder,
            data_set):
    """Runs one evaluation against the full epoch of data.
    Args:
      sess: The session in which the model has been trained.
      logits: The Tensor that returns the number of correct predictions.
      units_placeholder: The units placeholder.
      labels_placeholder: The labels placeholder.
      data_set: The set of units and labels to evaluate, from
        data.read_data_sets().
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // data_set.batch_size
    num_examples = steps_per_epoch * data_set.batch_size
    for step in range(steps_per_epoch):
        feed_dict, labels_feed = fill_feed_dict(data_set,
                                                units_placeholder,
                                                labels_placeholder)
        prediction_table = sess.run(logits, feed_dict=feed_dict)
        prediction = np.argmax(prediction_table)
        if labels_feed[0] == prediction:
            true_count += 1

    false_count = num_examples - true_count
    return true_count, false_count


def do_true_eval(sess,
                 logits,
                 units_placeholder,
                 labels_placeholder,
                 data_set):
    """Runs one evaluation against the full epoch of data.
    Args:
      sess: The session in which the model has been trained.
      logits: The Tensor that returns the number of correct predictions.
      units_placeholder: The units placeholder.
      labels_placeholder: The labels placeholder.
      data_set: The set of units and labels to evaluate, from
        data.read_data_sets().
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // data_set.batch_size
    num_examples = steps_per_epoch * data_set.batch_size
    for step in range(steps_per_epoch):
        feed_dict, labels_feed = fill_feed_dict(data_set,
                                                units_placeholder,
                                                labels_placeholder)
        prediction_table = sess.run(logits, feed_dict=feed_dict)
        prediction = np.argmax(prediction_table)
        if labels_feed[0] == prediction:
            true_count += 1

    false_count = num_examples - true_count
    return true_count, false_count


def to_excel(lasso_result, dir, file_name):
    writer = pd.ExcelWriter(dir + file_name + '.xlsx')
    lasso_result.to_excel(writer)


def run_training(flags, data_sets):
    """Train mnist_example for a number of steps."""

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Generate placeholders for the units and labels.
        units_placeholder, labels_placeholder = placeholder_inputs(data_sets.batch_size, data_sets.column_number)

        # Build a Graph that computes predictions from the inference model.
        logits = inference(units_placeholder,
                           flags.hidden_units,
                           data_sets.column_number,
                           data_sets.class_number,
                           flags.dropout)

        # Add to the Graph the Ops for loss calculation.
        loss = do_loss(logits, labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = training(loss, flags.learning_rate)

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Run the Op to initialize the variables.
        sess.run(init)

        print("\t".join(['learning_rate', 'max_steps', 'hidden_units']))
        print("{:f}\t{:d}\t{}".format(flags.learning_rate, flags.max_steps, flags.hidden_units))
        print("")
        print("\t".join(['step', 'train_Accuracy', 'Recall', 'Accuracy', 'F1score']))

        result = []
        # Start the training loop.
        for step in range(flags.max_steps + 1):

            # Fill a feed dictionary with the actual set of units and labels
            # for this particular training step.
            feed_dict, _ = fill_feed_dict(data_sets.train,
                                          units_placeholder,
                                          labels_placeholder)

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            _, loss_value = sess.run([train_op, loss],
                                     feed_dict=feed_dict)

            # Save a checkpoint and evaluate the model periodically.
            if step % (flags.max_steps / 25) == 0:
                checkpoint_file = os.path.join(flags.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                # Evaluate against the training set.
                train_true_count, train_false_count = do_eval(sess,
                                                              logits,
                                                              units_placeholder,
                                                              labels_placeholder,
                                                              data_sets.train)
                # Evaluate against the true test set.
                # TP: 실제 1을 1이라고 예측한것
                # FN: 실제 1을 0이라고 예측한것
                TP, FN = do_true_eval(sess,
                                      logits,
                                      units_placeholder,
                                      labels_placeholder,
                                      data_sets.true_test)
                # Evaluate against the false test set.
                # TN: 실제 0을 0이라고 예측한것
                # FP: 실제 0을 1이라고 예측한것
                TN, FP = do_eval(sess,
                                 logits,
                                 units_placeholder,
                                 labels_placeholder,
                                 data_sets.false_test)

                Precision = TP / (TP + FP + 1e-20)  # 1이라고 예측한것 중 실제 1인것의 비중
                Recall = TP / (TP + FN + 1e-20)  # 실제 1인것들 중에서 예측결과가 1인 것의 비중
                Accuracy = (TP + TN) / (TP + TN + FP + FN)  # 정확히 예측(즉, 1을 1이라고, 0을 0이라고 예측)한 것의 비중
                ClassificationError = (FP + FN) / (TP + TN + FP + FN)  # 틀리게 예측(즉, 1을 0이라고, 0을 1이라고 예측)한 것의 비중
                F1score = 2 / (1 / (Precision + 1e-20) + 1 / (Recall + 1e-20))  # harmonic mean
                tmp = [step, TP, FP, FN, TN, Precision, Recall, Accuracy, ClassificationError, F1score]
                result.append(tmp)
                print("{:d}\t{:f}\t{:f}\t{:f}\t{:f}".format(step,
                                                            train_true_count / (train_true_count + train_false_count),
                                                            Recall, Accuracy, F1score))

        res_col = ['Step', 'True Positive', 'False Positive', 'False Negative', 'True Negative', 'Precision',
                   'Recall',
                   'Accuracy', 'Classification Error', 'F1 Score']
        neural_net_result = pd.DataFrame(result, columns=res_col)

        # plot
        plt.figure()
        title = 'Neural Net Performance - {}'.format(flags.sector_name)
        if flags.lasso_applied:
            title = title.format() + ' with LASSO'
        else:
            title = title.format() + ' without LASSO'
        plt.title(title)
        plt.plot(neural_net_result['Step'], neural_net_result['Accuracy'], label="Accuracy")
        plt.plot(neural_net_result['Step'], neural_net_result['Recall'], label="Recall")
        plt.plot(neural_net_result['Step'], neural_net_result['F1 Score'], label="F1 Score")
        plt.legend()
        result_file_name = '{}_{}_{}_{}_{}_{}_{}'.format(flags.sector_name, flags.lasso_applied,
                                                         flags.true_adjusting_rate, flags.max_steps,
                                                         flags.learning_rate,
                                                         flags.dropout, flags.hidden_units)
        plt.savefig(
            flags.image_dir + result_file_name + '.png')
        plt.close()

        to_excel(neural_net_result, flags.excel_dir, result_file_name)
