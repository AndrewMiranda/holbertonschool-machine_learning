#!/usr/bin/env python3
"""Specificity"""
import numpy as np


def specificity(confusion):
    """specificity: calculates the specificity for each class in a
                    confusion matrix
    Args:
    confusion is a confusion numpy.ndarray of shape (classes, classes)
            where row indices represent the correct labels and column
            indices represent the predicted labels
        classes is the number of classes
    Returns:
            a numpy.ndarray of shape (classes,) containing
            the specificity of each class
"""
    classes, classes = confusion.shape
    specificity = np.zeros(shape=(classes,))
    for i in range(classes):
        specificity[i] = (
            np.sum(confusion) - np.sum(confusion, axis=1)[i]
            - np.sum(confusion, axis=0)[i] + confusion[i][i]
        ) / (np.sum(confusion) - np.sum(confusion, axis=1)[i])
    return specificity
