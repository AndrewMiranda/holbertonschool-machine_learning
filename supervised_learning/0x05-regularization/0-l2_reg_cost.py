#!/usr/bin/env python3
'''Module for function l2_reg_cost(cost, lambtha, weights, L, m).
'''

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    '''calculates the cost of a neural network with L2 regularization.
    Args.
        cost: The cost of the network without L2 regularization.
        lambtha: The regularization parameter.
        weights: A dictionary of the weights and biases (numpy.ndarrays) of
        the neural network.
        L: The number of layers in the neural network.
        m: The number of data points used
    Returns.
        The cost of the network accounting for L2 regularization.
    '''

    # Frobenius Norm
    l2_norm = 0
    for i in range(1, L + 1):
        l2_norm = l2_norm + np.sqrt(np.sum(weights["W{}".format(i)] ** 2))
    return (cost + (lambtha / (2 * m)) * l2_norm)
