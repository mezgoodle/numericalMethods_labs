import numpy as np
from math import sqrt


def get_transpose(matrix: list) -> list:
    """
    Function for getting transpose matrix
    :param matrix: input matrix
    :return: transpose input matrix
    """
    matrixT = matrix.copy()
    n = len(matrixT)
    for i in range(n):
        for j in range(i, n):
            matrixT[i][j], matrixT[j][i] = matrixT[j][i], matrixT[i][j]
    return matrixT


def cholesky_decomposition(a: list) -> list:
    """
    Cholesky decomposition
    :param a: start matrix
    :return: Lower-triangular matrix
    """
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


def solveLU(L: list, U: list, b: list) -> list:
    """
    The solve main function
    :param L: Lower-triangular matrix
    :param U: Upper-triangular matrix
    :param b: matrix B
    :return: vector x, the solution
    """
    n = len(L)
    y = np.zeros(n)
    x = np.zeros(n)

    # forward substitution
    for i in range(n):
        sumj = 0
        for j in range(i):
            sumj += L[i][j] * y[j]
        y[i] = (b[i] - sumj) / L[i][i]

    # backward substitution
    for i in range(n-1, -1, -1):
        sumj = 0
        for j in range(i+1, n):
            sumj += U[i][j] * x[j]
        x[i] = (y[i] - sumj) / U[i][i]

    return x


a = [[8.0, 3.22, 0.8, 0.0, 4.1],
     [3.22, 7.76, 2.33, 1.91, -1.03],
     [0.8, 2.33, 5.25, 1.0, 3.02],
     [0.0, 1.91, 1.0, 7.5, 1.03],
     [4.1, -1.03, 3.02, 1.03, 6.44]]
b = [9.45, -12.2, 7.78, -8.1, 10.0]
print('Matrix a:', np.matrix(a))
print('Matrix b:', np.matrix(b))
L = cholesky_decomposition(a)
U = get_transpose(L)
x = solveLU(L, U, b)
print('Solution vector:', x)

print('NumPy solution:', np.linalg.solve(a, b))