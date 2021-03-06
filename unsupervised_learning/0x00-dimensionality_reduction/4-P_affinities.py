#!/usr/bin/env python3
"""Function that calculates the Shannon entropy
and P affinities relative to a data point"""

import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Binary search to get P affinities so each gaussian
    has the same perplexity
    Args:
        X:
        tol:
        perplexity:
    Returns:
    """
    (n, d) = X.shape
    D, P, betas, H = P_init(X, perplexity)

    if n == 0:
        return P

    for i in range(n):
        row = D[i].copy()
        row = np.delete(row, i, axis=0)
        Hi, Pi = HP(row, betas[i])
        Hdiff = Hi - H
        b_max = None
        b_min = None
        while np.abs(Hdiff) > tol:
            if Hdiff > 0:
                b_min = betas[i, 0]
                if b_max is None:
                    betas[i, 0] = betas[i, 0] * 2.
                else:
                    betas[i, 0] = (betas[i, 0] + b_max) / 2.
            else:
                b_max = betas[i, 0]
                if b_min is None:
                    betas[i, 0] = betas[i, 0] / 2.
                else:
                    betas[i, 0] = (betas[i, 0] + b_min) / 2.

            Hi, Pi = HP(row, betas[i])
            Hdiff = Hi - H
        Pi = np.insert(Pi, i, 0)
        P[i] = Pi

    P = (P.T + P) / (2 * n)
    return P
