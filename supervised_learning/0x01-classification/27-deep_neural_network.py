#!/usr/bin/env python3
"""Module that defines the class DeepNeuralNetwork
Defines a deep neural network performing binary classification
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


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
            W_key = "W{}".format(i + 1)
            b_key = "b{}".format(i + 1)
            if i == 0:
                self.weights[W_key] = (np.random.randn(layers[i],
                                       self.nx) * np.sqrt(2 / self.nx))
            else:
                self.weights[W_key] = (np.random.randn(layers[i],
                                       layers[i - 1])
                                       * np.sqrt(2 / layers[i - 1]))
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
            W_key = "W{}".format(i + 1)
            b_key = "b{}".format(i + 1)
            k_pre = "A{}".format(i)
            k_for = "A{}".format(i + 1)

            Z_matmul = np.matmul(self.__weights[W_key], self.__cache[k_pre])
            Z = Z_matmul + self.__weights[b_key]
            if i == self.__L - 1:
                Ze = np.exp(Z)
                self.__cache[k_for] = (Ze / np.sum(Ze, axis=0, keepdims=True))
            else:
                self.__cache[k_for] = 1 / (1 + np.exp(-Z))

        return self.__cache[k_for], self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression
        Args.
            Y: a one-hot numpy.ndarray of shape (classes, m) that contains
            the correct labels for the input data
            A: numpy.ndarray with shape (1, m) containing the activated
            output of the neuron for each example
        Returns:
            The cost
        """

        return (-1 / (Y.shape[1])) * np.sum(Y * np.log(A))

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions
        Args.
            X: numpy.ndarray with shape (nx, m) that contains the input data
            Y: a one-hot numpy.ndarray of shape (classes, m) that contains
            the correct labels for the input data
        Returns.
            The neuron’s prediction and the cost of the network
        """

        self.forward_prop(X)[0]
        A_key = "A" + str(self.__L)
        tmp = np.amax(self.__cache[A_key], axis=0)
        pred = np.where(self.__cache[A_key] == tmp, 1, 0)
        return pred, self.cost(Y, self.__cache[A_key])

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
            dW = (1 / m) * np.matmul(dZ, self.__cache[A_key].T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            dZ_1 = np.matmul(self.__weights[W_key].T, dZ)
            dZ_2 = self.__cache[A_key] * (1 - self.__cache[A_key])
            dZ = dZ_1 * dZ_2

            self.__weights[W_key] = self.__weights[W_key] - alpha * dW
            self.__weights[b_key] = self.__weights[b_key] - alpha * db

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
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)

            if i % step == 0 or i == iterations:
                cost = self.cost(Y, self.__cache['A{}'.format(self.L)])
                cost_list.append(cost)
                step_list.append(i)
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))

        if graph:
            plt.plot(step_list, cost_list)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Trainig Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format
        Args.
            filename: the file to which the object should be saved
        """

        if filename[-4:] != ".pkl":
            filename = filename + ".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object
        Args.
            filename is the file from which the object should be loaded
        Returns.
            The loaded object, or None if filename doesn’t exist
        """

        try:
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
            return obj
        except FileNotFoundError:
            return None
