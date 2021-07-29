#!/usr/bin/env python3
"""Contains the binomial class"""


class Binomial:
    """Binomial class"""

    def __init__(self, data=None, n=1, p=0.5):
        """Constructor"""
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            mean = float(sum(data) / len(data))
            my_list = [(x - mean) ** 2 for x in data]
            variance = sum(my_list) / len(data)
            p = 1 - variance / mean
            if ((mean / p) - (mean // p)) >= 0.5:
                self.n = 1 + int(mean / p)
            else:
                self.n = int(mean / p)
            self.p = float(mean / self.n)

        else:
            if n <= 0:
                raise ValueError("n must be a positive value")
            self.n = int(n)
            if (p <= 0) or (p >= 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.p = float(p)

    def pmf(self, k):
        """value of the PMF for a given number of “successes”"""
        k = int(k)
        if k < 0:
            return 0

        nfact = 1
        for x in range(1, self.n + 1):
            nfact *= x
        kfact = 1
        for x in range(1, k + 1):
            kfact *= x
        nkfact = 1
        for x in range(1, self.n - k + 1):
            nkfact *= x
        return (nfact / (kfact * nkfact)) * \
            self.p ** k * (1 - self.p) ** (self.n - k)

    def cdf(self, k):
        """value of the CDF for a given number of “successes”"""
        k = int(k)
        if k < 0:
            return 0

        cdf = 0

        for x in range(k + 1):
            cdf += self.pmf(x)
        return cdf
