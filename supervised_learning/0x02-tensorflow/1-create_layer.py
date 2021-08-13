#!/usr/bin/env python3
'''Module for the function create_layer
'''

import tensorflow as tf


def create_layer(prev, n, activation):
    '''Createa a tensor layer
    Args:
        prev: The tensor output of the previous layer
        n: The number of nodes in the layer to create
        activation: The activation function that the layer should use
    '''

    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    x = tf.layers.Dense(units=n,
                        activation=activation,
                        name='layer',
                        kernel_initializer=init)(prev)
    return x
