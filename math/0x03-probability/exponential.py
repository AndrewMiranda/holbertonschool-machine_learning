#!/usr/bin/env python3
"""Contains the Exponential class"""


class Exponential:
    """Exponential class"""

    def __init__(self, data=None, lambtha=1.):
        """constructor"""
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = 1 / float(sum(data) / len(data))
        else:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

    def pdf(self, x):
        """value of the PDF for a given time period"""
        if x < 0:
            return 0

        return self.lambtha * 2.7182818285 ** (- self.lambtha * x)

    def cdf(self, x):
        """value of the CDF for a given time period"""
        if x < 0:
            return 0
        return 1 - 2.7182818285 ** (- self.lambtha * x)
