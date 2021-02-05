import numpy as np


def GausePrime(matrix):
    """
    [[a11x1 + a12x2 + a13x3+...+a1nxn=b1],
    [a21x1 + a22x2 + a23x3+...+a2nxn=b2],
    [a31x1 + a32x2 + a33x3+...+a3nxn=b3]
    ...
    [an1x1 + an2x2 + an3x3+...+annxn=bn]]
    :param matrix:
    :return:
    """
    matrix1 = matrix.copy()
    column = len(matrix1)
    n = len(matrix1)
    for k in range(n):
        for i in range(k+1, n):
            for j in range(k, n):
                matrix1[i][j] -= matrix1[k][j] * (matrix1[i][k] / matrix1[k][k])
    return matrix1


matrix = np.arange(1, 21).reshape(4, 5)
print(matrix)
print(GausePrime(matrix))
