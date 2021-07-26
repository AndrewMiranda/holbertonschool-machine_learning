#!/usr/bin/env python3
""" Derivation """


def poly_derivative(poly):
    """
    INPUT: polynomial
    OUTPUT derivative of the polynomial
    """
    if type(poly) is not list or len(poly) == 0:
        return None

    if type(poly[0]) is not int and type(poly[0]) is not float:
        return None

    derivative = [poly[i] * i for i in range(len(poly))]
    derivative = derivative[1:] if len(poly) > 1 else [0]
    return derivative
