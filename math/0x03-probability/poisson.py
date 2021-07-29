#!/usr/bin/env python3
"""Contains the Poisson"""


class Poisson:
    """Poisson class"""

    def __init__(self, data=None, lambtha=1.):
        """constructor"""
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))
        else:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”"""
        k = int(k)

        if k < 0:
            return 0

        # Multiply elements one by one
        factorial = 1
        for x in range(1, k + 1):
            factorial *= x

        return ((2.7182818285 ** (- self.lambtha)) *
                (self.lambtha ** k)) / factorial

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of “successes”"""
        k = int(k)
        if k < 0:
            return 0

        summation = 0
        for i in range(k + 1):
            factorial = 1
            for x in range(1, i + 1):
                factorial *= x
            summation += (self.lambtha ** i) / factorial
        return (2.7182818285 ** (-self.lambtha)) * summation
