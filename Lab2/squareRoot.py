import numpy as np
from math import sqrt


def cholesky(a: list) -> list:
    """
    Cholesky decomposition
    :param a: start matrix
    :return: Lower triangular matrix
    """
    a = np.array(a, float)
    L = np.zeros_like(a)
    n = len(a)
    for j in range(n):
        for i in range(j, n):
            if i == j:
                sumK = 0
                for k in range(j):
                    sumK += L[i][k] ** 2
                L[i][j] = sqrt(a[i][j] - sumK)
            else:
                sumK = 0
                for k in range(j):
                    sumK += L[i][k] * L[j][k]
                L[i][j] = (a[i][j] - sumK) / L[j][j]
    return L


a = [[5.2, 3, 0.5, 1, 2],
     [3, 6.3, -2, 4, 0],
     [0.5, -2, 8, -3.1, 3],
     [1, 4, -3.1, 7.6, 2.6],
     [2, 0, 3, 2.6, 15]]
L = cholesky(a)
print(L)
print(np.dot(L, np.transpose(L)))
