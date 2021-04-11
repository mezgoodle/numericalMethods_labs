import numpy as np
import matplotlib.pyplot as plt
from string import Template
from math import sin

template = Template('#' * 10 + ' $string ' + '#' * 10)


def linear_function(x: int, alpha=3) -> float:
    """
    Main linear function
    :param x: x value
    :param alpha: some value
    :return: result vector_b = f(x)
    """
    y_value = sin(alpha / 2 * x) + (x * alpha) ** (1 / 3)
    return y_value


def create_indexes(x_values: list) -> dict:
    """
    Function that creates indexes for spline coefficients matrix
    :param x_values: our nodes
    :return: dictionary with indexes
    """
    indexes = {}
    length = len(x_values)
    for i in range(length - 1):
        indexes[f'b{i + 1}'] = i
        indexes[f'c{i + 1}'] = i + length - 1
        indexes[f'd{i + 1}'] = i + length * 2 - 2
    indexes['y'] = (length - 1) * 3
    return indexes


def get_coeffs_for_newton_polynomial(x_elements: list, y_elements: list) -> list:
    """
    Creates pyramid and extracts coefficients
    :param x_elements: our nodes
    :param y_elements: results of f(x)
    :return: list of coefficients
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


def print_newton_polynomial(x_values: list, newton_coeffs: list) -> None:
    """
    Function that prints our newton polynomial
    :param x_values: our nodes
    :param newton_coeffs: coefficients for newton polynomial
    :return: nothing to return
    """
    print(template.substitute(string='Newton eval'))
    for i in range(len(newton_coeffs)):
        print(f' +({round(newton_coeffs[i], 5)}) ', end='')
        for j in range(i):
            print(f'(x-{x_values[j]})', end='')
    print(' = y', end='\n')

    # Create polynomial with NumPy
    final_polynomial = np.polynomial.Polynomial([0.])  # our target polynomial
    length = len(newton_coeffs)
    for i in range(length):
        polynomial = np.polynomial.Polynomial([1.])  # create a dummy polynomial
        for j in range(i):
            p_temp = np.polynomial.Polynomial([-x_values[j], 1.])  # (x_elements - x_j)
            polynomial = np.polymul(polynomial, p_temp)  # multiply dummy with expression
        polynomial *= newton_coeffs[i]  # apply coefficient
        final_polynomial = np.polyadd(final_polynomial, polynomial)  # add to target polynomial
    final_polynomial[0].coef = np.round_(final_polynomial[0].coef, decimals=5)
    print(final_polynomial[0])


def solve_newton_polynomial(newton_coeffs: list, x_values: list, x_value: float) -> float:
    """
    Function that calculate newton polynomial at x_valuew
    :param newton_coeffs: coefficients for newton polynomial
    :param x_values: our nodes
    :param x_value: current x
    :return: f(x)
    """
    length = len(x_values) - 1
    result = newton_coeffs[length]
    for k in range(1, length + 1):
        result = newton_coeffs[length - k] + (x_value - x_values[length - k]) * result
    return result


def show_plot(x_values: list, y_values: list, newton_coeffs=None, spline_coeffs=None, indexes=None) -> None:
    """
    Function for creating plots
    :param x_values: our nodes
    :param y_values: values at this nodes
    :param newton_coeffs: coefficients for newton polynomial
    :param spline_coeffs: coefficients for spline equations
    :param indexes:
    :return: nothing to return
    """
    x_axis = np.linspace(4, 12, num=10000)
    x_axis_2 = np.linspace(4, 12, num=2000)
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values, 'o', label='Data')
    ax.plot(x_axis, [linear_function(x) for x in x_axis], label='Linear')
    if newton_coeffs is not None:
        ax.plot(x_axis_2, [solve_newton_polynomial(newton_coeffs, x_values, x) for x in x_axis_2],
                label='Newton Polynomial')
    elif spline_coeffs is not None:
        ax.plot(x_axis_2, [solve_spline_equation(x_values, y_values, x, spline_coeffs, indexes) for x in x_axis_2],
                label='Spline interpolation')
    ax.legend(loc='lower left', ncol=2)
    plt.grid()
    plt.show()


def create_matrix(x_values: list, y_values: list, indexes: dict) -> [list, list]:
    """
    Function that create matrix to find coefficients for spline equations
    :param x_values: our nodes
    :param y_values: values at this nodes
    :param indexes: indexes for matrix
    :return: matrix a and vector b
    """
    matrix_a = []
    indexes_length = len(indexes)
    # I
    for i in range(1, len(x_values)):
        row = np.zeros(indexes_length)
        h = x_values[i] - x_values[i - 1]
        row[indexes[f'b{i}']] = h
        row[indexes[f'c{i}']] = h ** 2
        row[indexes[f'd{i}']] = h ** 3
        row[indexes['y']] = y_values[i] - y_values[i - 1]
        matrix_a.append(row)
    # II
    for i in range(1, len(x_values) - 1):
        row = np.zeros(indexes_length)
        h = x_values[i] - x_values[i - 1]
        row[indexes[f'b{i + 1}']] = 1
        row[indexes[f'b{i}']] = -1
        row[indexes[f'c{i}']] = -2 * h
        row[indexes[f'd{i}']] = -3 * h ** 2
        row[indexes['y']] = 0
        matrix_a.append(row)
    # III
    for i in range(1, len(x_values) - 1):
        row = np.zeros(indexes_length)
        h = x_values[i] - x_values[i - 1]
        row[indexes[f'c{i + 1}']] = 1
        row[indexes[f'c{i}']] = -1
        row[indexes[f'd{i}']] = -3 * h
        row[indexes['y']] = 0
        matrix_a.append(row)
    # IV
    row = np.zeros(indexes_length)
    row[indexes[f'c{len(x_values) - 1}']] = 1
    row[indexes[f'd{len(x_values) - 1}']] = 3 * (x_values[-1] - x_values[-2])
    row[indexes['y']] = 0
    matrix_a.append(row)
    row = np.zeros(indexes_length)
    row[indexes['c1']] = 1
    row[indexes['y']] = 0
    matrix_a.append(row)
    vector_b = np.zeros(indexes_length - 1)
    for i in range(len(matrix_a)):
        vector_b[i] = matrix_a[i][-1]
    matrix_a = np.delete(matrix_a, np.s_[-1:], axis=1)
    print(template.substitute(string='Matrix A and vector B'))
    print(np.matrix(matrix_a))
    print(vector_b)
    return matrix_a, vector_b


def solve_kramer_method(matrix_a: list, vector_b: list, matrix_c: list) -> list:
    """
    Kramer function to find spline coeffiecents
    :param matrix_a: matrix a
    :param vector_b: vector b
    :param matrix_c: matrix a copy
    :return: list of spline coefficients
    """
    spline_coeffs = []
    for i in range(0, len(vector_b)):
        for j in range(0, len(vector_b)):
            matrix_c[j][i] = vector_b[j]
            if i > 0:
                matrix_c[j][i - 1] = matrix_a[j][i - 1]
        spline_coeffs.append(np.linalg.det(matrix_c) / np.linalg.det(matrix_a))
    spline_coeffs = np.array(spline_coeffs).round(5)
    return spline_coeffs


def print_spline_equations(x_values: list, y_values: list, spline_coeffs: list, indexes: dict) -> None:
    """
    Function that print spline equations
    :param x_values: our nodes
    :param y_values: values at this nodes
    :param spline_coeffs: coefficients for spline equations
    :param indexes: indexes for spline equations
    :return: nothing to return
    """
    print(template.substitute(string='Spline equations'))
    for i in range(len(x_values) - 1):
        print(
            f"{y_values[i]} +({spline_coeffs[indexes[f'b{i + 1}']]})(x-{x_values[i]}) +{spline_coeffs[indexes[f'c{i + 1}']]}(x-{x_values[i]})**2 +{spline_coeffs[indexes[f'd{i + 1}']]}(x-{x_values[i]})**3")


def solve_spline_equation(x_values: list, y_values: list, x_value: float, spline_coeffs: list, indexes: dict) -> float:
    """
    Function to get value at x point of spline equation
    :param x_values: our nodes
    :param y_values: values at this nodes
    :param x_value: current x point
    :param spline_coeffs: coefficients for spline equations
    :param indexes: dictionary with indexes for spline equations
    :return: f(x)
    """
    for i in range(len(x_values) - 1):
        if x_values[i] <= x_value <= x_values[i + 1]:
            return y_values[i] + spline_coeffs[indexes[f'b{i + 1}']] * (x_value - x_values[i]) + spline_coeffs[
                indexes[f'c{i + 1}']] * (x_value - x_values[i]) ** 2 + spline_coeffs[indexes[f'd{i + 1}']] * (
                               x_value - x_values[i]) ** 3


def main():
    """Main function"""
    k = 10 - 1
    x_values = [-5 + k, -3 + k, -1 + k, 1 + k, 3 + k]
    y_values = [linear_function(x) for x in x_values]
    print(template.substitute(string='X and Y values'))
    print(x_values)
    print(np.array(y_values).round(5))
    show_plot(x_values=x_values.copy(), y_values=y_values.copy())
    # Newton Polynomial
    newton_coeffs = get_coeffs_for_newton_polynomial(x_values.copy(), y_values.copy())
    print(template.substitute(string='Coefficients'))
    print(np.array(newton_coeffs).round(5))
    print_newton_polynomial(x_values.copy(), newton_coeffs.copy())
    show_plot(x_values=x_values.copy(), y_values=y_values.copy(), newton_coeffs=newton_coeffs)
    # Cubic spline
    # x_values_2 = [2, 3, 5, 7]
    # y_values_2 = [4, -2, 6, -3]
    indexes = create_indexes(x_values.copy())
    print(template.substitute(string='Indexes'))
    print(indexes)
    matrix_a, vector_b = create_matrix(x_values.copy(), y_values.copy(), indexes.copy())
    matrix_c = matrix_a.copy()
    spline_coeffs = solve_kramer_method(matrix_a, vector_b, matrix_c)
    print(template.substitute(string='Spline coeffiecients'))
    print(indexes)
    print_spline_equations(x_values.copy(), y_values.copy(), spline_coeffs.copy(), indexes.copy())
    show_plot(x_values=x_values.copy(), y_values=y_values.copy(), spline_coeffs=spline_coeffs.copy(),
              indexes=indexes.copy())


main()
