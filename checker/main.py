import numpy as np


def get_sum(n, value):
    """Get sum from value to n"""
    result = np.sum([value, - 1])
    for x in range(n):
        result = np.sum([result, np.sum([x, 1])])
    return result


def get_product(n, value):
    """Get product from value to n"""
    result = value
    for x in range(n):
        result = np.multiply(result, np.sum([x, 1]))
    return result


def increase_matrix(matrix, value):
    """Multiply matrix with value"""
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] = np.multiply(matrix[i][j], value)
    return matrix
