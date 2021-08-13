#!/usr/bin/env python3
"""Module for the function calculate_accuracy
"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """calculates the accuracy of a prediction:
    Args:
        y: placeholder for the labels of the input data
        y_pred: tensor containing the network’s predictions
    Returns:
        A tensor containing the decimal accuracy of the prediction
    """

    y_dec = tf.argmax(y, axis=1)
    yp_dec = tf.argmax(y_pred, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(y_dec, yp_dec), tf.float32))
