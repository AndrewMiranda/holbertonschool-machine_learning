""" Integration """


def poly_integral(poly, C=0):
    """
    INPUT: polynomial
    OUTPUT integration polynomial
    """
    if type(poly) is not list or len(poly) == 0:
        return None
    if type(C) is not int and type(C) is not float:
        return None
    if type(poly[0]) is not int and type(poly[0]) is not float:
        return None
    flag = True
    aux = []
    for i in range(len(poly) - 1, -1, -1):
        if poly[i] == 0 and flag:
            pass
        else:
            aux.append(poly[i])
            flag = False
    aux.reverse()
    integration = [C] + [aux[i] / (i + 1) for i in range(len(aux))]
    integration = [int(x) if x % 1 == 0 else x for x in integration]
    return integration
