#!/usr/bin/env python3
'''Defines a neural network with one hidden layer performing binary
classification.
'''

import numpy as np


class NeuralNetwork:
    '''Defines a neural network with one hidden layer performing binary
    classification.
    '''

    def __init__(self, nx, nodes):
        '''Class constructor.
        Args.
            nx: The number of input features.
            nodes: The number of nodes found in the hidden layer.
        '''

        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.W1 = np.random.normal(0, 1, (nodes, nx))
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.normal(0, 1, (1, nodes))
        self.b2 = 0
        self.A2 = 0
