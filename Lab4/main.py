from string import Template

import numpy as np

template = Template('#' * 10 + ' $string ' + '#' * 10)
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
    for i in range(length - 1, 0, -1):
        matrix_b = np.zeros((length, length))
        np.fill_diagonal(matrix_b, 1)
        matrix_b_minus = matrix_b.copy()

        # Fill matrix b and minus one b
        for j in range(length):
            if j == i - 1:
                matrix_b[i - 1][j] = 1 / matrix[i][i - 1]
            else:
                matrix_b[i - 1][j] = matrix[i][j] / matrix[i][i - 1] * (-1)
            matrix_b_minus[i - 1][j] = matrix[i][j]
        print(template.substitute(string=f'Step: {abs(i - 4)}'))
        print(template.substitute(string='Matrix b'))
        print(matrix_b)
        print(template.substitute(string='Matrix b minus'))
        print(matrix_b_minus)
        matrix = np.dot(matrix_b_minus, np.dot(matrix, matrix_b))
        print(template.substitute(string='Temporary result'))
        print(matrix)
    return matrix


print('Matrix A:')
print(np.matrix(a))
normal_form = algorithm(a.copy())
print('Normal form:')
print(normal_form)
print()
