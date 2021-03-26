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
    Get a Frobenius form
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
        print(template.substitute(string=f'Step: {abs(i - length)}'))
        print(template.substitute(string='Matrix b'))
        print(matrix_b)
        print(template.substitute(string='Matrix b minus'))
        print(matrix_b_minus)
        matrix = np.dot(matrix_b_minus, np.dot(matrix, matrix_b))
        print(template.substitute(string='Temporary result'))
        print(matrix)
    return matrix


def print_equation(coefficients: list) -> None:
    coefficients = list(coefficients)
    coefficients = [round(coef * (-1), 5) for coef in coefficients]
    coefficients.insert(0, 1)
    equation = ''
    for index in range(len(coefficients)):
        equation += '+(' + str(coefficients[index]) + '*λ^' + str(abs(index - len(coefficients) + 1)) + ') '
    equation += '= 0'
    print(equation)


print('Matrix A:')
print(np.matrix(a))
normal_form = algorithm(a.copy())
print('Frobenius form:')
print(normal_form)
print_equation(normal_form[0])
