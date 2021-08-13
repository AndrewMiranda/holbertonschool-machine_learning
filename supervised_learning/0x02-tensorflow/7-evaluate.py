#!/usr/bin/env python3
"""Module for the function evaluate
"""

import tensorflow as tf


def evaluate(X, Y, save_path):
    """Evaluates the output of a neural network.
    Args:
        X is a numpy.ndarray containing the input data to evaluate
        Y is a numpy.ndarray containing the one-hot labels for X
        save_path is the location to load the model from
    Returns:
        The network’s prediction, accuracy, and loss, respectively
    """

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + ".meta")
        saver.restore(sess, save_path)
        graph = tf.get_default_graph()
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        acc = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        y_pred_oh, acc, loss = sess.run([y_pred, acc, loss], {x: X, y: Y})
    return y_pred_oh, acc, loss
