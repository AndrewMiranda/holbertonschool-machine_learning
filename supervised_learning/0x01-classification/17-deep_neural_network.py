#!/usr/bin/env python3
"""Module that defines the class DeepNeuralNetwork
Defines a deep neural network performing binary classification
"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """Class constructor
        Args.
            nx: is the number of input features
            layers: is a list representing the number of nodes in each layer
            of the network
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")
        self.nx = nx
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.L):
            if type(layers[i]) is not int or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            W_key = "W{}".format(i+1)
            b_key = "b{}".format(i+1)
            if i == 0:
                self.weights[W_key] = (np.random.randn(layers[i],
                                       self.nx) * np.sqrt(2 / self.nx))
            else:
                self.weights[W_key] = (np.random.randn(layers[i],
                                       layers[i-1]) * np.sqrt(2/layers[i-1]))
            self.weights[b_key] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """L getter"""
        return self.__L

    @property
    def cache(self):
        """cache getter"""
        return self.__cache

    @property
    def weights(self):
        """weights getter"""
        return self.__weights
