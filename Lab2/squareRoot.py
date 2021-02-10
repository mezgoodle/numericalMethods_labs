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
    T = np.zeros_like(a)
    n = len(a)
    for j in range(n):
        for i in range(j, n):
            if i == j:
                sumK = 0
                for k in range(j):
                    sumK += T[i][k] ** 2
                T[i][j] = sqrt(a[i][j] - sumK)
            else:
                sumK = 0
                for k in range(j):
                    sumK += T[i][k] * T[j][k]
                T[i][j] = (a[i][j] - sumK) / T[j][j]
    return T


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
    print('matrix y:', y)
    # backward substitution
    for i in range(n-1, -1, -1):
        sumj = 0
        for j in range(i+1, n):
            sumj += U[i][j] * x[j]
        x[i] = (y[i] - sumj) / U[i][i]

    return x


a = [[1.0, 0.42, 0.54, 0.66],
     [0.42, 1.0, 0.32, 0.44],
     [0.54, 0.32, 1.0, 0.22],
     [0.66, 0.44, 0.22, 1.0]]
b = [0.3, 0.5, 0.7, 0.9]
T = cholesky_decomposition(a)
print('matrix T:', T)
U = get_transpose(T)
print('matrix T-transpose:', U)
x = solveLU(T, U, b)
print('Solution vector:', x)

print('NumPy solution:', np.linalg.solve(a, b))
