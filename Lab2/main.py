import numpy as np


def GausePrime(matrix: list) -> list:
    """
    [[a11x1 + a12x2 + a13x3+...+a1nxn=b1],
    [a21x1 + a22x2 + a23x3+...+a2nxn=b2],
    [a31x1 + a32x2 + a33x3+...+a3nxn=b3]
    ...
    [an1x1 + an2x2 + an3x3+...+annxn=bn]]
    :param matrix:
    :return matrix:
    """
    results = []
    matrix1 = matrix.copy()
    rows = len(matrix1)
    columns = len(matrix1[0])
    for row in range(rows):
        for i in range(row+1, rows):
            value = matrix1[i][row]/matrix1[row][row]
            for j in range(row, columns):
                matrix1[i][j] = matrix1[i][j] + matrix1[row][j] * (-1) * value
    return matrix1


matrix = [[1, 1, 1, 10], [2, 3, 4, 9], [4, 7, 8, 8]]
print(matrix)
print(GausePrime(matrix))
