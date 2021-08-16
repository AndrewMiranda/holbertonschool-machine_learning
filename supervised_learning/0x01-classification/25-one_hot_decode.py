#!/usr/bin/env python3
"""Module for the function one_hot_decode
Converts a one-hot matrix into a vector of labels
"""

import numpy as np


def one_hot_decode(one_hot):
    """Converts a one-hot matrix into a vector of labels
    Args.
        one_hot: a one-hot encoded numpy.ndarray with shape (classes, m)
    Returns.
        A numpy.ndarray with shape (m, ) containing the numeric labels
        for each example, or None on failure
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None

    return(np.argmax(one_hot, axis=0))
