#!/usr/bin/env python3
"""Sequential - use the Sequential class"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """build_model: builds a neural network
                        with the Keras library
    Args:
        nx : is the number of input features to the network
        layers : is a list containing the number
                of nodes in each layer of the network
        activations : is a list containing the activation
                    functions used for each layer of the network
        lambtha : is the L2 regularization parameter
        keep_prob : is the probability that a node will
                    be kept for dropout
    Returns:
        the keras model
    """
    network = K.models.Sequential()
    for i in range(len(layers)):
        if i == 0:
            network.add(K.layers.Dense(
                layers[i], activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha),
                input_shape=(nx,)))
        else:
            network.add(K.layers.Dense(
                layers[i], activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha)))
        if i != len(layers) - 1:
            network.add(K.layers.Dropout(1 - keep_prob))
    return network
