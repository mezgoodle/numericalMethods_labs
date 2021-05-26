from math import sqrt


def get_fault(x: list, xm: list) -> float:
    """
    Function for finding get_fault
    :param x: my solution vector
    :param xm: NumPy solution vector
    :return: fault
    """
    sum_k = 0
    n = len(x)
    for k in range(1, n):
        sum_k += (x[k] - xm[k]) ** 2
    result = sqrt(sum_k / n)
    return result
