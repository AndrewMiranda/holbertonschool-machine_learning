#!/usr/bin/env python3
"""Input - use the Input class"""
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
    inputs = K.layers.Input(shape=(nx,))
    for i in range(len(layers)):
        if i == 0:
            outputs = inputs
        outputs = K.layers.Dense(
            layers[i], activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha))(outputs)
        if i != len(layers) - 1:
            outputs = K.layers.Dropout(1 - keep_prob)(outputs)
    network = K.models.Model(inputs=inputs, outputs=outputs)
    return network
