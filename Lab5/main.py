import numpy as np
import matplotlib.pyplot as plt
from string import Template
from math import sin

template = Template('#' * 10 + ' $string ' + '#' * 10)


def main_function(x: int, alpha=3) -> float:
    """
    Main function on a graphic
    :param x: x value
    :param alpha: some value
    :return: result matrix_b = f(x)
    """
    y_value = sin(alpha / 2 * x) + (x * alpha) ** (1 / 3)
    return y_value


# Consts
indexes = {}
for i in range(len([2, 3, 5, 7])-1):
    indexes[f'b{i + 1}'] = i
    indexes[f'c{i + 1}'] = i + len([2, 3, 5, 7])-1
    indexes[f'd{i + 1}'] = i + len([2, 3, 5, 7])*2-2
indexes['y'] = (len([2, 3, 5, 7])-1)*3
k = 10 - 1
x_values = [-5 + k, -3 + k, -1 + k, 1 + k, 3 + k]
y_values = [main_function(x) for x in x_values]
print(template.substitute(string='X and Y values'))
print(x_values)
print(np.array(y_values).round(5))


def getCoeffsForNewton(x_elements: list, y_elements: list) -> list:
    """
    Creates pyramid and extracts coefficients
    :param x_elements: x values
    :param y_elements: results of f(x)
    :return: coefficients
    """
    length = len(y_elements)
    pyramid = []
    for _ in range(length):
        tmp_array = []
        for _ in range(length):
            tmp_array.append(0)
        pyramid.append(tmp_array)
    print(template.substitute(string='Zero pyramid'))
    print(np.matrix(pyramid))
    for index in range(length):
        pyramid[index][0] = y_elements[index]
    print(template.substitute(string='Pyramid with first y_elements'))
    print(np.matrix(pyramid))
    for step in range(1, length):
        for index in range(length - step):
            pyramid[index][step] = (pyramid[index + 1][step - 1] - pyramid[index][step - 1]) / (
                    x_elements[index + step] - x_elements[index])
    print(template.substitute(string='Final pyramid'))
    print(np.matrix(pyramid))
    return pyramid[0]  # return first row


coeff_vector = getCoeffsForNewton(x_values.copy(), y_values.copy())
print(template.substitute(string='Coefficients'))
print(np.array(coeff_vector).round(5))


def print_eval_newton(x, coeff_vector):
    print(template.substitute(string='Newton eval'))
    for i in range(len(coeff_vector)):
        print(f' +({round(coeff_vector[i], 5)}) ', end='')
        for j in range(i):
            print(f'(x-{x[j]})', end='')
    print(' = matrix_b', end='\n')

    # Create polynomial with NumPy
    final_pol = np.polynomial.Polynomial([0.])  # our target polynomial
    n = len(coeff_vector)  # get number of coeffs
    for i in range(n):
        p = np.polynomial.Polynomial([1.])  # create a dummy polynomial
        for j in range(i):
            # each vector has degree of i
            # their terms are dependant on 'x_elements' values
            p_temp = np.polynomial.Polynomial([-x[j], 1.])  # (x_elements - x_j)
            p = np.polymul(p, p_temp)  # multiply dummy with expression
        p *= coeff_vector[i]  # apply coefficient
        final_pol = np.polyadd(final_pol, p)  # add to target polynomial
    final_pol[0].coef = np.round_(final_pol[0].coef, decimals=5)
    print(final_pol[0])


def eval_poly(coeffs, x_old, x_current):
    n = len(x_old) - 1
    p = coeffs[n]
    for k in range(1, n + 1):
        p = coeffs[n - k] + (x_current - x_old[n - k]) * p
    return p


print_eval_newton(x_values, coeff_vector)

x_axis = np.linspace(4, 12, num=10000)
x_axis_2 = np.linspace(4, 12, num=2000)
plt.plot(x_axis, [main_function(x) for x in x_axis], color='blue')
plt.plot(x_axis, [eval_poly(coeff_vector, x_values, x) for x in x_axis], color='green')
plt.legend(['Main', 'Poly'])
plt.grid()
plt.show()


def create_matrix(x_array, y_array):
    matrix_a = []
    # I
    for i in range(1, len(x_array)):
        row = np.zeros(10)
        h = x_array[i] - x_array[i - 1]
        row[indexes[f'b{i}']] = h
        row[indexes[f'c{i}']] = h ** 2
        row[indexes[f'd{i}']] = h ** 3
        row[indexes['y']] = y_array[i] - y_array[i - 1]
        matrix_a.append(row)
    # II
    for i in range(1, len(x_array) - 1):
        row = np.zeros(10)
        h = x_array[i] - x_array[i - 1]
        row[indexes[f'b{i + 1}']] = 1
        row[indexes[f'b{i}']] = -1
        row[indexes[f'c{i}']] = -2 * h
        row[indexes[f'd{i}']] = -3 * h ** 2
        row[indexes['y']] = 0
        matrix_a.append(row)
    # III
    for i in range(1, len(x_array) - 1):
        row = np.zeros(10)
        h = x_array[i] - x_array[i - 1]
        row[indexes[f'c{i + 1}']] = 1
        row[indexes[f'c{i}']] = -1
        row[indexes[f'd{i}']] = -3 * h
        row[indexes['y']] = 0
        matrix_a.append(row)
    # IV
    row = np.zeros(10)
    row[indexes[f'c{len(x_array) - 1}']] = 1
    row[indexes[f'd{len(x_array) - 1}']] = 3 * (x_array[-1] - x_array[-2])
    row[indexes['y']] = 0
    matrix_a.append(row)
    row = np.zeros(10)
    row[indexes['c1']] = 1
    row[indexes['y']] = 0
    matrix_a.append(row)
    matrix_b = np.zeros(9)
    for i in range(len(matrix_a)):
        matrix_b[i] = matrix_a[i][-1]
        # del matrix_a[i][-1]
    matrix_a = np.delete(matrix_a, np.s_[-1:], axis=1)
    print(np.matrix(matrix_a))
    print(matrix_b)
    return matrix_a, matrix_b


def Kramer(matrix, matrix_copy, matrix_b):
    coeff_array = []
    for i in range(0, len(matrix_b)):
        for j in range(0, len(matrix_b)):
            matrix_copy[j][i] = matrix_b[j]
            if i > 0:
                matrix_copy[j][i - 1] = matrix[j][i - 1]
        coeff_array.append(np.linalg.det(matrix_copy) / np.linalg.det(matrix))
    coeff_array = np.array(coeff_array).round(5)
    return coeff_array


def print_s(coeffs_array, x_array, y_array):
    print(template.substitute(string='S evals'))
    for i in range(len(x_array)-1):
        print(f"{y_array[i]} +({coeffs_array[indexes[f'b{i+1}']]})(x-{x_array[i]}) +{coeffs_array[indexes[f'c{i+1}']]}(x-{x_array[i]})**2 +{coeffs_array[indexes[f'd{i+1}']]}(x-{x_array[i]})**3")


def eval_s(coeffs_array, x_array, y_array, x_value):
    for i in range(len(x_array)-1):
        if x_array[i] <= x_value <= x_array[i+1]:
            return y_array[i] + coeffs_array[indexes[f'b{i+1}']] * (x_value-x_array[i]) + coeffs_array[indexes[f'c{i+1}']] * (x_value-x_array[i]) ** 2 + coeffs_array[indexes[f'd{i+1}']] * (x_value-x_array[i]) ** 3


matrix_a, matrix_b = create_matrix([2, 3, 5, 7], [4, -2, 6, -3])
s_coeffs = Kramer(matrix_a, matrix_a.copy(), matrix_b)
print(s_coeffs)
print_s(s_coeffs, [2, 3, 5, 7], [4, -2, 6, -3])
print(eval_s(s_coeffs, [2, 3, 5, 7], [4, -2, 6, -3], 7))
