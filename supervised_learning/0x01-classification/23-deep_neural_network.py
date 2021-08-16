#!/usr/bin/env python3
"""Module that defines the class DeepNeuralNetwork
Defines a deep neural network performing binary classification
"""

import numpy as np
import matplotlib.pyplot as plt


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

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network
        Args.
            X: numpy.ndarray with shape (nx, m) that contains the input data
        Returns:
            The output of the neural network and the cache, respectively
        """

        self.__cache["A0"] = X
        for i in range(self.__L):
            cache = self.__cache["A{}".format(i)]
            b_weights = self.__weights["b{}".format(i+1)]
            w_weights = self.__weights["W{}".format(i+1)]
            Z = np.matmul(w_weights, cache) + b_weights
            self.__cache["A{}".format(i + 1)] = (np.exp(Z) / (np.exp(Z) + 1))
        return (self.__cache["A{}".format(i + 1)], self.__cache)

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression
        Args.
            Y: numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data
            A: numpy.ndarray with shape (1, m) containing the activated
            output of the neuron for each example
        Returns:
            The cost
        """
        m = Y.shape[1]
        nm = np.multiply
        cs = -(np.sum(nm(Y, np.log(A)) + nm((1 - Y), np.log(1.0000001 - A))))
        return cs / m

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions
        Args.
            X: numpy.ndarray with shape (nx, m) that contains the input data
            Y: numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data
        Returns.
            The neuron’s prediction and the cost of the network
        """
        A3, _ = self.forward_prop(X)
        prediction = np.where(A3 >= 0.5, 1, 0)
        cost = self.cost(Y, A3)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network
        Args.
            Y: numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data
            cache: a dictionary containing all the intermediary values of the
            network
            alpha: the learning rate
        """
        m = Y.shape[1]
        dZ = self.__cache["A{}".format(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            A_key = "A{}".format(i-1)
            W_key = "W{}".format(i)
            b_key = "b{}".format(i)
            dW = (1/m)*np.matmul(dZ, self.__cache[A_key].T)
            db = (1/m)*np.sum(dZ, axis=1, keepdims=True)
            dZ_1 = np.matmul(self.__weights[W_key].T, dZ)
            dZ_2 = self.__cache[A_key] * (1 - self.__cache[A_key])
            dZ = dZ_1 * dZ_2

            self.__weights[W_key] = self.__weights[W_key] - alpha*dW
            self.__weights[b_key] = self.__weights[b_key] - alpha*db

    def train(self,
              X,
              Y,
              iterations=5000,
              alpha=0.05,
              verbose=True,
              graph=True,
              step=100):
        """Trains the deep neural network
        Args.
            X: numpy.ndarray with shape (nx, m) that contains the input data
            Y: numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data
            iterations: The number of iterations to train over
            alpha: the learning rate
            verbose: is a boolean that defines whether or not to print
            information about the training.
            graph: is a boolean that defines whether or not to graph
            information about the training once the training has completed.
            step: Point in wich to print the training info or training graph
        Returns:
            The evaluation of the training data after iterations of training
            have occurred
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        step_list = []
        cost_list = []
        for i in range(iterations + 1):
            self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)
            cost = self.cost(Y, self.__cache["A{}".format(self.__L)])

            if verbose:
                if i % step == 0 or step == iterations:
                    step_list.append(i)
                    cost_list.append(cost)
                    print("Cost after {} iterations: {}".format(i, cost))

        if graph:
            plt.plot(step_list, cost_list)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Trainig Cost")
            plt.show()

        return self.evaluate(X, Y)
