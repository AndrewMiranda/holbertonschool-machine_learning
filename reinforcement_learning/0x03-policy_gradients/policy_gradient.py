#!/usr/bin/env python3
"""module"""
import numpy as np


def policy(matrix, weight):
    """find policy with a weight of a matrix """
    z = matrix.dot(weight)
    exp = np.exp(z)
    return exp/np.sum(exp)


def softmax_grad(softmax):
    """find the grad """
    s = softmax.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


def policy_gradient(state, weight):
    """ computes the Monte-Carlo policy
        gradient based on a state and a weight matrix"""
    probs = policy(state, weight)
    action = np.random.choice(len(probs[0]), p=probs[0])
    dsoftmax = softmax_grad(probs)[action, :]
    dlog = dsoftmax / probs[0, action]
    grad = state.T.dot(dlog[None, :])
    return action, grad
