import numpy as np

np.set_printoptions(suppress=True)

a = [
    [2.2, 1, 0.5, 2],
    [1, 1.3, 2, 1],
    [0.5, 2, 0.5, 1.6],
    [2, 1, 1.6, 2]
]


def algorithm(matrix: list) -> list:
    """
    Get a Frubenius form
    :param matrix:
    :return:
    """
    length = len(matrix)
    for i in range(length-1, 0, -1):
        matrix_b = np.zeros((length, length))
        np.fill_diagonal(matrix_b, 1)
        matrix_b_minus = matrix_b.copy()

        # Fill matrix b and minus one b
        for j in range(length):
            if j == i-1:
                matrix_b[i - 1][j] = 1 / matrix[i][i-1]
            else:
                matrix_b[i-1][j] = matrix[i][j] / matrix[i][i-1] * (-1)
            matrix_b_minus[i-1][j] = matrix[i][j]
        matrix = np.dot(matrix_b_minus, np.dot(matrix, matrix_b))
    return matrix


normal_form = algorithm(a.copy())


print('Matrix A:')
print(np.matrix(a))
print('Normal form:')
print(normal_form)
