import numpy as np
import matplotlib.pyplot as plt
from string import Template
from math import sin

template = Template('#' * 10 + ' $string ' + '#' * 10)


def main_function(x: int, alpha=3) -> float:
    y_value = sin(alpha / 2 * x) + (x * alpha) ** (1 / 3)
    return y_value


k = 10 - 1
x_values = [-5 + k, -3 + k, -1 + k, 3 + k]
y_values = [main_function(x) for x in x_values]
print(template.substitute(string='X and Y values'))
print(x_values)
print(y_values)

def getCoeffsForNewton(x_elements: list, y_elements: list) -> list:
    """
    Creates NDD pyramid and extracts coeffs
    :param x_elements: parameters
    :param y_elements: results
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
print(coeff_vector)


def eval_newton(x, coeff_vector):
    print(template.substitute(string='Newton eval'))
    for i in range(len(coeff_vector)):
        print(f'+{coeff_vector[i]}', end='')
        for j in range(i):
            print(f'(x-{x[j]})', end='')
    print('=y', end='\n')

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
    print(final_pol[0])
    return final_pol


final_pol = eval_newton(x_values, coeff_vector)
print(template.substitute(string='Coeffs with polynomial from NumPy'))
p = np.flip(final_pol[0].coef, axis=0)
print(np.array(p).round(5))

x_axis = np.linspace(-1, 10, num=5000)
y_axis = np.polyval(p, x_axis)
plt.plot.title = 'hello'
plt.plot(x_axis, y_axis, color='blue')
plt.plot(x_values, np.polyval(p, x_values), color='green')
plt.legend(['first', 'second'])
plt.grid()
plt.show()
print(main_function(x_values[1]))
