#!/usr/bin/env python3
'''Module containing the neruon class
'''

import numpy as np


class Neuron:
    '''Class that defines a neuron.
    '''

    def __init__(self, nx):
        '''Initialization function for the Neuron class.
        Args.
            nx: The number of input features to the neuron.
        '''

        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.normal(0, 1, (1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        '''Returns the value of __W.
        '''

        return self.__W

    @property
    def b(self):
        '''Returns the value of __b.
        '''

        return self.__b

    @property
    def A(self):
        '''Returns the value of __A.
        '''

        return self.__A

    def forward_prop(self, X):
        '''Calculates the forward propagation of the neuron.
        Args.
            X: numpy.ndarray with shape (nx, m) that contains the input data.
        Returns.
            The private attribute __A.
        '''

        x = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-x))
        return self.__A

    def cost(self, Y, A):
        '''Calculates the cost of the model using logistic regression.
        Args.
            Y: A numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data.
            A: A numpy.ndarray with shape (1, m) containing the activated
            output of the neuron for each example.
        Returns.
            The cost of the model.
        '''
        loss_sum = np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        self.__cost = -(1 / A.size) * loss_sum
        return self.__cost

    def evaluate(self, X, Y):
        '''Evaluates the neuron’s predictions.
        Args.
            X: numpy.ndarray with shape (nx, m) that contains the input data.
            Y: numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data.
        Returns.
            The neuron’s prediction and the cost of the network as a
            numpy.ndarray with shape(1, m) containing the predicted
            labels for each example. With values, 1 if the output of
            the network is >= 0.5 and 0 otherwise
        '''

        self.forward_prop(X)
        pred = np.where(self.__A >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A)
        return pred, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        '''Calculates one pass of gradient descent on the neuron.
        Args.
            X: numpy.ndarray with shape (nx, m) that contains the input data.
            Y: numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data.
            A: numpy.ndarray with shape (1, m) containing the activated
            output of the neuron for each example.
            alpha: learning rate.
        '''

        dz = A - Y
        dw = (1 / A.size) * np.matmul(dz, X.T)
        db = (1 / A.size) * np.sum(dz)
        self.__W = self.__W - (alpha * dw)
        self.__b = self.__b - (alpha * db)
