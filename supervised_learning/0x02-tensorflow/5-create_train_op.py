#!/usr/bin/env python3
"""Module for the function create_train_op
"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """Creates the training operation for the network
    Args:
        loss: The loss of the network’s prediction
        alpha: The learning rate
    Returns:
        An operation that trains the network using gradient descent
    """

    op = tf.train.GradientDescentOptimizer(alpha)
    return op.minimize(loss)
