#!/usr/bin/env python3
'''Module for function l2_reg_create_layer(prev, n, activation, lambtha).
'''

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    '''Creates a tensorflow layer that includes L2 regularization.
    Args.
        prev: A tensor containing the output of the previous layer.
        n: The number of nodes the new layer should contain.
        activation: The activation function that should be used on the layer.
        lambtha: is the L2 regularization parameter.
    Returns.
        The output of the new layer.
    '''
    W = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    l2 = tf.contrib.layers.l2_regularizer(lambtha)
    return tf.layers.Dense(
                           units=n,
                           activation=activation,
                           kernel_initializer=W,
                           kernel_regularizer=l2
                           )(prev)
