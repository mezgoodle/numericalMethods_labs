import numpy as np
import matplotlib.pyplot as plt
from string import Template
from math import sin

template = Template('#' * 10 + ' $string ' + '#' * 10)


def linear_function(x: int, alpha=3) -> float:
    """
    Main function on a graphic
    :param x: x value
    :param alpha: some value
    :return: result matrix_b = f(x)
    """
    y_value = sin(alpha / 2 * x) + (x * alpha) ** (1 / 3)
    return y_value


# Consts
def create_indexes(x_values):
    indexes = {}
    length = len(x_values)
    for i in range(length-1):
        indexes[f'b{i + 1}'] = i
        indexes[f'c{i + 1}'] = i + length-1
        indexes[f'd{i + 1}'] = i + length*2-2
    indexes['y'] = (length-1)*3
    return indexes


def get_coeffs_for_newton_polynomial(x_elements: list, y_elements: list) -> list:
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


def print_newton_polynomial(x, coeff_vector):
    print(template.substitute(string='Newton eval'))
    for i in range(len(coeff_vector)):
        print(f' +({round(coeff_vector[i], 5)}) ', end='')
        for j in range(i):
            print(f'(x-{x[j]})', end='')
    print(' = y', end='\n')

    # Create polynomial with NumPy
    final_pol = np.polynomial.Polynomial([0.])  # our target polynomial
    n = len(coeff_vector)  # get number of newton_coeffs
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


def solve_newton_polynomial(newton_coeffs, x_values, x_value):
    length = len(x_values) - 1
    result = newton_coeffs[length]
    for k in range(1, length + 1):
        result = newton_coeffs[length - k] + (x_value - x_values[length - k]) * result
    return result


def show_plot(x_values=None, y_values=None, newton_coeffs=None, spline_coeffs=None, indexes=None):
    x_axis = np.linspace(4, 12, num=10000)
    x_axis_2 = np.linspace(4, 12, num=2000)
    fig, ax = plt.subplots()
    ax.plot(x_values, [2.01001 ,3.03286 ,2.34793 ,3.75752 ,2.55094], 'o', label='Data')
    ax.plot(x_axis, [linear_function(x) for x in x_axis], label='Linear')
    if newton_coeffs is not None:
        ax.plot(x_axis_2, [solve_newton_polynomial(newton_coeffs, x_values, x) for x in x_axis_2], label='Newton Polynomial')
    elif spline_coeffs is not None:
        ax.plot(x_axis_2, [solve_spline_equation(x_values, y_values, x, spline_coeffs, indexes) for x in x_axis_2], label='Spline interpolation')
    ax.legend(loc='lower left', ncol=2)
    plt.grid()
    plt.show()


def create_matrix(x_array, y_array, indexes):
    matrix_a = []
    indexes_length = len(indexes)
    # I
    for i in range(1, len(x_array)):
        row = np.zeros(indexes_length)
        h = x_array[i] - x_array[i - 1]
        row[indexes[f'b{i}']] = h
        row[indexes[f'c{i}']] = h ** 2
        row[indexes[f'd{i}']] = h ** 3
        row[indexes['y']] = y_array[i] - y_array[i - 1]
        matrix_a.append(row)
    # II
    for i in range(1, len(x_array) - 1):
        row = np.zeros(indexes_length)
        h = x_array[i] - x_array[i - 1]
        row[indexes[f'b{i + 1}']] = 1
        row[indexes[f'b{i}']] = -1
        row[indexes[f'c{i}']] = -2 * h
        row[indexes[f'd{i}']] = -3 * h ** 2
        row[indexes['y']] = 0
        matrix_a.append(row)
    # III
    for i in range(1, len(x_array) - 1):
        row = np.zeros(indexes_length)
        h = x_array[i] - x_array[i - 1]
        row[indexes[f'c{i + 1}']] = 1
        row[indexes[f'c{i}']] = -1
        row[indexes[f'd{i}']] = -3 * h
        row[indexes['y']] = 0
        matrix_a.append(row)
    # IV
    row = np.zeros(indexes_length)
    row[indexes[f'c{len(x_array) - 1}']] = 1
    row[indexes[f'd{len(x_array) - 1}']] = 3 * (x_array[-1] - x_array[-2])
    row[indexes['y']] = 0
    matrix_a.append(row)
    row = np.zeros(indexes_length)
    row[indexes['c1']] = 1
    row[indexes['y']] = 0
    matrix_a.append(row)
    matrix_b = np.zeros(indexes_length-1)
    for i in range(len(matrix_a)):
        matrix_b[i] = matrix_a[i][-1]
    matrix_a = np.delete(matrix_a, np.s_[-1:], axis=1)
    print(template.substitute(string='Matrix A and matrix B'))
    print(np.matrix(matrix_a))
    print(matrix_b)
    return matrix_a, matrix_b


def solve_kramer_method(matrix_a, matrix_b, matrix_c):
    spline_coeffs = []
    for i in range(0, len(matrix_b)):
        for j in range(0, len(matrix_b)):
            matrix_c[j][i] = matrix_b[j]
            if i > 0:
                matrix_c[j][i - 1] = matrix_a[j][i - 1]
        spline_coeffs.append(np.linalg.det(matrix_c) / np.linalg.det(matrix_a))
    spline_coeffs = np.array(spline_coeffs).round(5)
    return spline_coeffs


def print_spline_equations(x_values, y_values, spline_coeffs, indexes):
    print(template.substitute(string='Spline equations'))
    for i in range(len(x_values) - 1):
        print(f"{y_values[i]} +({spline_coeffs[indexes[f'b{i + 1}']]})(x-{x_values[i]}) +{spline_coeffs[indexes[f'c{i + 1}']]}(x-{x_values[i]})**2 +{spline_coeffs[indexes[f'd{i + 1}']]}(x-{x_values[i]})**3")


def solve_spline_equation(x_values, y_values, x_value, spline_coeffs, indexes):
    for i in range(len(x_values) - 1):
        if x_values[i] <= x_value <= x_values[i + 1]:
            return y_values[i] + spline_coeffs[indexes[f'b{i + 1}']] * (x_value - x_values[i]) + spline_coeffs[indexes[f'c{i + 1}']] * (x_value - x_values[i]) ** 2 + spline_coeffs[indexes[f'd{i + 1}']] * (x_value - x_values[i]) ** 3


def main():
    k = 10 - 1
    x_values = [-5 + k, -3 + k, -1 + k, 1 + k, 3 + k]
    y_values = [linear_function(x) for x in x_values]
    print(template.substitute(string='X and Y values'))
    print(x_values)
    print(np.array(y_values).round(5))
    show_plot(x_values=x_values.copy())
    # Newton Polynomial
    newton_coeffs = get_coeffs_for_newton_polynomial(x_values.copy(), y_values.copy())
    print(template.substitute(string='Coefficients'))
    print(np.array(newton_coeffs).round(5))
    print_newton_polynomial(x_values.copy(), newton_coeffs.copy())
    show_plot(x_values=x_values.copy(), newton_coeffs=newton_coeffs)
    # Cubic spline
    # x_values_2 = [2, 3, 5, 7]
    # y_values_2 = [4, -2, 6, -3]
    indexes = create_indexes(x_values.copy())
    print(template.substitute(string='Indexes'))
    print(indexes)
    matrix_a, matrix_b = create_matrix(x_values.copy(), y_values.copy(), indexes.copy())
    matrix_c = matrix_a.copy()
    spline_coeffs = solve_kramer_method(matrix_a, matrix_b, matrix_c)
    print(template.substitute(string='Spline coeffiecients'))
    print(indexes)
    print_spline_equations(x_values.copy(), y_values.copy(), spline_coeffs.copy(), indexes.copy())
    show_plot(x_values=x_values.copy(), y_values=y_values.copy(), spline_coeffs=spline_coeffs.copy(), indexes=indexes.copy())


main()
