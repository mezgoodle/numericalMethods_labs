import numpy as np


def methodGaussian(matrix: list) -> list:
    """
    The methodGaussian function.
    This function get matrix like this:
    [[A11X1 + A12X2 + A13X3+...+A1nXn=B1],
    [A21X1 + A22X2 + A23X3+...+a2nxn=B2],
    [A31X1 + A32X2 + A33X3+...+A3nXn=B3]
    ...
    [an1x1 + an2x2 + an3x3+...+annxn=bn]]

    Parameters:
        matrix (list): The start matrix.

    Returns:
        List: The result matrix.
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
print(methodGaussian(matrix))
