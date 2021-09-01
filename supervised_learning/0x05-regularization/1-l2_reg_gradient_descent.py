#!/usr/bin/env python3
'''Module for the function
l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L)
'''

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    '''Updates the weights and biases of a neural network using
    gradient descent with L2 regularization (The neural network
    uses tanh activations on each layer except the last, which
    uses a softmax activation, the weights and biases of the
    network are updated in place).
    
    Args.
        Y: A one-hot numpy.ndarray of shape (classes, m) that contains the
        correct labels for the data.
            - classes: The number of classes.
            - m: The number of data points.
        weights: A dictionary of the weights and biases of the NN.
        cache: A dictionary of the outputs of each layer of the NN.
        alpha: The learning rate.
        lambtha: The L2 regularization parameter.
        L: The number of layers of the network.
    '''

    m = Y.shape[1]
    dz = cache["A{}".format(L)] - Y
    for i in range(L, 0, -1):
        # frobenius norm
        f_norm = (lambtha / m) * weights["W{}".format(i)]

        # calculate dw and db, replace values of W and b
        dw_i = ((1 / m) * np.matmul(dz, cache["A{}".format(i - 1)].T)) + f_norm
        db_i = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        weights["W{}".format(i)] -= (alpha * dw_i)
        weights["b{}".format(i)] -= (alpha * db_i)

        # define dz for next epoch
        dz_1 = np.matmul(weights["W{}".format(i)].T, dz)
        dz_af = 1 - np.square(cache["A{}".format(i - 1)])
        dz = dz_1 * dz_af
