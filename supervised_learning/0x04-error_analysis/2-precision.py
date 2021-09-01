#!/usr/bin/env python3
"""Precision"""
import numpy as np


def precision(confusion):
    """precision: calculates the precision for each class
                    in a confusion matrix:
    Args:
    confusion is a confusion numpy.ndarray of shape
            (classes, classes) where row indices represent the
            correct labels and column indices represent
            the predicted labels:
        classes is the number of classes
    Returns:
            a numpy.ndarray of shape (classes,) containing the precision
            of each class
"""
    classes, classes = confusion.shape
    precision = np.zeros(shape=(classes,))
    for i in range(classes):
        precision[i] = confusion[i][i] / np.sum(confusion, axis=0)[i]
    return precision
