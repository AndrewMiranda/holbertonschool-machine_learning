#!/usr/bin/env python3
"""
Performs the Baum-Welch algorithm for a hidden markov model
https://www.adeveloperdiary.com/data-science/machine-learning/
derivation-and-implementation-of-baum-welch-algorithm-for-hidden-markov-model/
"""


import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Returns: P, F, or None, None on failure
    """
    T = Observation.shape[0]

    N, _ = Emission.shape

    F = np.zeros((N, T))

    # Initialization step
    F[:, 0] = Initial.transpose() * Emission[:, Observation[0]]

    # Recursion
    for i in range(1, T):
        F[:, i] = \
            np.matmul(F[:, i - 1], Transition) * Emission[:, Observation[i]]

    # Likelihood of the observations given the model
    P = np.sum(F[:, T - 1])

    return P, F


def backward(Observation, Emission, Transition, Initial):
    """
    Returns: P, B, or None, None on failure
    """
    T = Observation.shape[0]

    N, _ = Emission.shape

    B = np.zeros((N, T))

    # Initialization step
    B[:, T - 1] = np.ones(N)

    # Recursion
    for t in range(T - 2, -1, -1):
        prob = \
            np.sum(B[:, t + 1] *
                   Emission[:, Observation[t + 1]] * Transition, axis=1)
        B[:, t] = prob

    # Likelihood of the observations given the model
    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])

    return P, B


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Returns: the converged Transition, Emission, or None, None on failure
    """
    if type(Observations) is not np.ndarray or len(Observations.shape) != 1:
        return None, None

    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None

    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None

    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None

    T = Observations.shape[0]
    N, M = Emission.shape

    if Transition.shape[0] != N or Transition.shape[1] != N:
        return None, None

    if Initial.shape[0] != N or Initial.shape[1] != 1:
        return None, None

    if not np.sum(Transition, axis=1).all():
        return None, None

    if not np.sum(Emission, axis=1).all():
        return None, None

    if not np.sum(Initial) == 1:
        return None, None

    for _ in range(iterations):
        _, alpha = forward(Observations, Emission, Transition, Initial)
        _, beta = backward(Observations, Emission, Transition, Initial)

        xi = np.zeros((N, N, T - 1))
        for t in range(T - 1):
            a = np.matmul(alpha[:, t].transpose(), Transition)
            b = Emission[:, Observations[t + 1]].transpose()
            c = beta[:, t + 1]
            denominator = np.matmul(a * b, c)

            for i in range(N):
                a = alpha[i, t]
                b = Transition[i]
                c = Emission[:, Observations[t + 1]].transpose()
                d = beta[:, t + 1].transpose()
                numerator = a * b * c * d
                xi[i, :, t] = numerator / denominator

        gamma = np.sum(xi, axis=1)

        # TRANSITION CALCULATION
        num = np.sum(xi, 2)
        den = np.sum(gamma, axis=1).reshape((-1, 1))
        Transition = num / den

        # EMISSION CALCULATION
        # Add additional T'th element in gamma
        xi_sum = np.sum(xi[:, :, T - 2], axis=0)
        xi_sum = xi_sum.reshape((-1, 1))
        gamma = np.hstack((gamma, xi_sum))

        denominator = np.sum(gamma, axis=1)
        denominator = denominator.reshape((-1, 1))

        for i in range(M):
            gamma_i = gamma[:, Observations == i]
            Emission[:, i] = np.sum(gamma_i, axis=1)

        Emission = Emission / denominator

    return Transition, Emission
