#!/usr/bin/env python3
"""Module for the function one_hot_encode
Converts a numeric label vector into a one-hot matrix
"""

import numpy as np


def one_hot_encode(Y, classes):
    """converts a numeric label vector into a one-hot matrix:
    Args.
        Y: numpy.ndarray with shape (m,) containing numeric class labels
        classes: The maximum number of classes found in Y
    Returns:
        A one-hot encoding of Y with shape (classes, m), or None on failure
    """

    if not isinstance(Y, np.ndarray) or len(Y) < 1:
        return None
    if not isinstance(classes, int) or classes <= np.amax(Y):
        return None
    one_hot = np.zeros((classes, Y.shape[0]))
    col = np.arange(Y.size)
    one_hot[Y, col] = 1
    return one_hot
