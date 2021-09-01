#!/usr/bin/env python3
"""Sensitivity"""
import numpy as np


def sensitivity(confusion):
    """sensitivity: calculates the sensitivity for each class
                    in a confusion matrix:
    Args:
    confusion is a confusion numpy.ndarray of shape
                (classes, classes) where row indices represent
                the correct labels and column indices represent
                the predicted labels:
        classes is the number of classes
    Returns:
            a numpy.ndarray of shape (classes,) containing the sensitivity
            of each class
"""
    classes, classes = confusion.shape
    sensitivity = np.zeros(shape=(classes,))
    for i in range(classes):
        sensitivity[i] = confusion[i][i] / np.sum(confusion, axis=1)[i]
    return sensitivity
