import numpy as np


def get_sum(n: float, value: float) -> float:
    """
    Get sum from value to n
    :param n: number of elements
    :param value: start value
    :return: result of summering
    """
    result = np.sum([value, - 1])
    for x in range(n):
        result = np.sum([result, np.sum([x, 1])])
    return result


def get_product(n: float, value: float) -> float:
    """
    Get product from value to n
    :param n: number of elements
    :param value: start value
    :return: result of multiplying
    """
    result = value
    for x in range(n):
        result = np.multiply(result, np.sum([x, 1]))
    return result
