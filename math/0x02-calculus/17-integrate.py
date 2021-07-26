#!/usr/bin/env python3
"""Function to calculate the integral"""


def poly_integral(poly, C=0):
    """Function calculating the integral"""
    new_L = []
    i = 0

    if type(poly) != list or len(poly) == 0 or type(C) != int:
        return None
    new_L.append(C)
    if sum(poly) == 0:
        return new_L
    if len(poly) == 1:
        new_L.append(poly[0])
        return new_L
    while i < len(poly):
        if poly[i] % (i + 1) == 0:
            new_L.append(int(poly[i]/(i + 1)))
        else:
            new_L.append(poly[i]/(i + 1))
        i += 1
    return new_L
