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
    :return: result y = f(x)
    """
    y_value = sin(alpha / 2 * x) + (x * alpha) ** (1 / 3)
    return y_value


# Consts
k = 10 - 1
x_values = [-5 + k, -3 + k, -1 + k, 3 + k]
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
    print(' = y', end='\n')

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
    S = []
    matrix_eval = []
    # I
    for i in range(1, len(x_array)):
        row = []
        h = x_array[i] - x_array[i-1]
        for j in range(3):
            row.append(h ** (j+1))
        row.append(y_array[i] - y_array[i-1])
        matrix_eval.append(row)
    # II
    for i in range(1, len(x_array)-1):
        row = []
        h = x_array[i] - x_array[i - 1]
        row.append(1)
        row.append(-1)
        for j in range(2):
            row.append(-(j+2) * (h ** (j+1)))
        row.append(0)
        matrix_eval.append(row)
    # III
    for i in range(1, len(x_array) - 1):
        row = []
        h = x_array[i] - x_array[i - 1]
        row.append(1)
        row.append(-1)
        row.append(-3 * h)
        row.append(0)
        matrix_eval.append(row)
    # IV
    matrix_eval.append([1, 0])
    matrix_eval.append([1, 3 * (x_array[-1] - x_array[-2]), 0])
    print(np.array(matrix_eval))


create_matrix([2, 3, 5, 7], [4, -2, 6, -3])
