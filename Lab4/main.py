from string import Template
from checker import fault

import numpy as np
from typing import Tuple

template = Template('#' * 10 + ' $string ' + '#' * 10)
np.set_printoptions(suppress=True)

a = [
    [6.26, 1.14, 0.93, 1.24],
    [1.14, 4.16, 1.30, 0.16],
    [0.93, 1.30, 5.44, 2.10],
    [1.24, 0.16, 2.10, 6.10]
]


def frobenius(matrix: list) -> Tuple[list, list]:
    """
    Get a Frobenius form
    :param matrix: start matrix
    :return: normalised matrix and S matrix for future
    """
    length = len(matrix)
    s_matrix = np.identity(length)
    for i in range(length - 1, 0, -1):
        matrix_b = np.identity(length)
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
        s_matrix = np.dot(s_matrix, matrix_b)
        print(template.substitute(string='Matrix b minus'))
        print(matrix_b_minus)
        matrix = np.dot(matrix_b_minus, np.dot(matrix, matrix_b))
        print(template.substitute(string='Temporary result'))
        print(matrix)
    return matrix, s_matrix


def get_self_numbers(coefficients: list) -> list:
    """
    Function to get self numbers
    :param coefficients: numbers from the first raw in normalised matrix
    :return: self numbers of matrix
    """
    coefficients = list(coefficients)
    coefficients = [round(coef * (-1), 5) for coef in coefficients]
    coefficients.insert(0, 1)
    # Print equation area start
    equation = ''
    for index in range(len(coefficients)):
        equation += '+(' + str(coefficients[index]) + '*λ^' + str(abs(index - len(coefficients) + 1)) + ') '
    equation += '= 0'
    print(template.substitute(string='Characteristic equation'))
    print(equation)
    # Print equation area end
    roots = np.roots(coefficients)
    return roots


def get_self_vectors(self_numbers: list, s_matrix: list) -> list:
    """
    Function to get self vectors
    :param self_numbers: self numbers of matrix
    :param s_matrix: S matrix from earlier Frobenius function
    :return: self vectors
    """
    self_vectors = []
    y_array = []
    for number in self_numbers:
        temp_array = []
        for i in range(len(self_numbers)):
            temp_array.insert(0, pow(number, i))
        y_array.append(temp_array)
    print(template.substitute(string='Y array'))
    print(np.matrix(y_array))
    print(template.substitute(string='S matrix'))
    print(s_matrix)
    print(template.substitute(string='Self vectors'))
    for element in y_array:
        vector = np.dot(s_matrix, element)
        self_vectors.append(vector)
        print(np.matrix(vector))
    return self_vectors


def check_vectors(matrix_a: list, self_numbers: list, self_vectors: list) -> None:
    """
    Check the equation Ax = λx
    :param matrix_a: start matrix
    :param self_numbers: self numbers of matrix A
    :param self_vectors: self vectors of matrix A
    :return: nothing to return
    """
    print(template.substitute(string='Ax = λx'))
    for index in range(len(self_numbers)):
        print(np.dot(matrix_a, self_vectors[index]))
        print(np.dot(self_numbers[index], self_vectors[index]))
        print()


def main_part(matrix_a: list) -> None:
    """
    Get all functions in one place
    :param matrix_a: start matrix
    :return: nothing to return
    """
    normal_form, s_matrix = frobenius(matrix_a)
    print(template.substitute(string='Frobenius form'))
    print(normal_form)
    self_numbers = get_self_numbers(normal_form[0])
    print(template.substitute(string='Self numbers'))
    print(self_numbers)
    v, w = np.linalg.eigh(matrix_a)
    print(template.substitute(string='NumPy numbers'))
    print(v)
    self_vectors = get_self_vectors(self_numbers, s_matrix)
    check_vectors(matrix_a, self_numbers, self_vectors)
    print(template.substitute(string='Fault'))
    print(round(fault.get_fault(sorted(self_numbers), sorted(v)), 5))


print('Matrix A:')
print(np.matrix(a))
main_part(a.copy())

