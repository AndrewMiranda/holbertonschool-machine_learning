#!/usr/bin/env python3
'''Module for function dropout_forward_prop(X, weights, L, keep_prob).
'''

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    '''Conducts forward propagation using Dropout (the last layer uses softmax
    activation function, the rest use the tanh activation function).
    Args.
        X: A numpy.ndarray of shape (nx, m) containing the input data for the
        network.
            - nx: The number of input features.
            - m: The number of data points.
        weights: A dictionary of the weights and biases of the neural network.
        L: The number of layers in the network.
        keep_prob: The probability that a node will be kept.
    Returns.
        A dictionary containing the outputs of each layer and the dropout mask
        used on each layer.
    '''

    cache = {"A0": X}
    for i in range(1, L + 1):
        # find z of layer i using forward prop
        z_i1 = np.matmul(weights["W{}".format(i)], cache["A{}".format(i - 1)])
        z_i = z_i1 + weights["b{}".format(i)]

        # for last layer use softmax activation function
        if i == L:
            p = np.exp(z_i)
            cache["A{}".format(i)] = p / np.sum(p, axis=0, keepdims=True)
        # use Tanh activation function for the rest
        else:
            A_i = np.tanh(z_i)
            d_out = np.random.rand(A_i.shape[0], A_i.shape[1]) < keep_prob
            A_i = A_i * d_out
            cache["D{}".format(i)] = d_out.astype(int)
            cache["A{}".format(i)] = A_i / keep_prob

    return cache
