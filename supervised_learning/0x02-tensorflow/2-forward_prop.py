#!/usr/bin/env python3
"""Module for the function forward_prop
"""

import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """creates the forward propagation graph for the neural network
    Args:
        x: The placeholder for the input data
        layer_sizes: A list containing the number of nodes in each layer of
        the network
        activations: A list containing the activation functions for each layer
        of the network
    Returns:
        The prediction of the network in tensor form
    """

    layers = create_layer(x, layer_sizes[0], activations[0])
    for idx in range(1, len(layer_sizes)):
        layers = create_layer(layers, layer_sizes[idx], activations[idx])
    return layers
