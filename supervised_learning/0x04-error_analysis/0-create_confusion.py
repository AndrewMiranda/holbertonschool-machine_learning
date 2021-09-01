#!/usr/bin/env python3
"""Create Confusion"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """create_confusion_matrix: creates a confusion matrix.
    Args:
    labels is a one-hot numpy.ndarray of shape (m, classes)
            containing the correct labels for each data point:
        m is the number of data points
        classes is the number of classes
    logits is a one-hot numpy.ndarray of shape (m, classes)
            containing the predicted labels
    Returns:
            a confusion numpy.ndarray of shape (classes, classes) with
            row indices representing the correct labels and column indices
            representing the predicted labels
"""
    m, classes = labels.shape
    v1 = np.argmax(labels, axis=1)
    v2 = np.argmax(logits, axis=1)
    confusion = np.zeros(shape=(classes, classes))
    for i in range(classes):
        for j in range(classes):
            for k in range(m):
                if i == v1[k] and j == v2[k]:
                    confusion[i][j] += 1
    return confusion
