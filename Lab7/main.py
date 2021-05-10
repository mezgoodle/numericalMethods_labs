import numpy as np
import scipy
import scipy.optimize as opt
from math import cos, sin
from string import Template

a, b = 0.8, 1.7
epsilon = 10 ** (-5)
template = Template('#' * 10 + ' $string ' + '#' * 10)

coeffs = {
    1: {'x1': 0.5, 'c1': 2},
    2: {'x1': -0.577350, 'x2': 0.577350, 'c1': 1, 'c2': 1},
    3: {'x1': -0.774597, 'x2': 0, 'x3': 0.774597, 'c1': 0.555555, 'c2': 0.888889, 'c3': 0.555555},
    4: {'x1': -0.861136, 'x2': -0.339981, 'x3': 0.339981, 'x4': 0.861136, 'c1': 0.347855, 'c2': 0.652145,
        'c3': 0.652145, 'c4': 0.347855},
    5: {'x1': -0.906180, 'x2': -0.538470, 'x3': 0, 'x4': 0.538470, 'x5': 0.906180, 'c1': 0.236927, 'c2': 0.478629,
        'c3': 0.568889, 'c4': 0.478629, 'c5': 0.236927},
    6: {'x1': -0.932470, 'x2': -0.661210, 'x3': -0.238620, 'x4': 0.238620, 'x5': 0.661210, 'x6': 0.932470,
        'c1': 0.171324, 'c2': 0.360761,
        'c3': 0.467914, 'c4': 0.467914, 'c5': 0.360761, 'c6': 0.171324},
    7: {'x1': -0.949108, 'x2': -0.741531, 'x3': -0.405845, 'x4': 0, 'x5': 0.405845, 'x6': 0.741531, 'x7': 0.949108,
        'c1': 0.129485, 'c2': 0.279705,
        'c3': 0.381830, 'c4': 0.417960, 'c5': 0.381830, 'c6': 0.279705, 'c7': 0.129485},
    8: {'x1': -0.960290, 'x2': -0.796666, 'x3': -0.525532, 'x4': -0.183434, 'x5': 0.183434, 'x6': 0.525532,
        'x7': 0.796666, 'x8': 0.960290,
        'c1': 0.101228, 'c2': 0.222381,
        'c3': 0.313707, 'c4': 0.362684, 'c5': 0.362684, 'c6': 0.313707, 'c7': 0.222381, 'c8': 0.101228},
}


def main_func(x):
    return cos(x) / (x + 1)


def main_func_second(x):
    return (-sin(x) / (x + 1)) - (cos(x) / (x + 1) ** 2)


def main_func_fourth(x):
    return (cos(x) - (4 * sin(x) / (x + 1)) - (12 * cos(x) / (x + 1) ** 2) + (24 * sin(x) / (x + 1) ** 3) + (
            24 * cos(x) / (x + 1) ** 4)) / (x + 1)


def trapezium_method(a, b):
    parts, granic_fault = trapezium_method_fault(a, b)
    result = (main_func(a) + main_func(b)) / 2
    h = (b - a) / parts
    print(parts)
    index = a + h
    while index < b:
        result += main_func(index)
        index += h
    return result * h


def simpson_method(a, b):
    parts, granic_fault = simpson_method_fault(a, b)
    sum = main_func(a) + main_func(b)
    width = (b - a) / (2 * parts)
    print(parts)
    firstPart = 0
    secondPart = 0
    for i in range(1, parts):
        firstPart += main_func(2 * width * i + a) * 2
    sum += firstPart
    for i in range(1, parts + 1):
        secondPart += main_func(width * (2 * i - 1) + a) * 4
    sum += secondPart
    return sum * width / 3


def gaussian_method(a, b):
    parts, granic_fault = simpson_method_fault(a, b)
    sum = main_func(a) + main_func(b)
    width = (b - a) / (2 * parts)
    print(parts)
    firstPart = 0
    secondPart = 0
    for i in range(1, parts):
        firstPart += main_func(2 * width * i + a) * 2
    sum += firstPart
    for i in range(1, parts + 1):
        secondPart += main_func(width * (2 * i - 1) + a) * 4
    sum += secondPart
    return sum * width / 3


def trapezium_method_fault(a, b):
    n = 1
    M = opt.fmin_l_bfgs_b(lambda x: -main_func_second(x), 1.0, bounds=[(a, b)], approx_grad=True)
    fault = abs(M[1][0]) * ((b - a) ** 3) / (12 * n ** 2)
    while epsilon < fault:
        fault = abs(M[1][0]) * ((b - a) ** 3) / (12 * n ** 2)
        n += 1
    return n, fault


def simpson_method_fault(a, b):
    n = 1
    M = opt.fmin_l_bfgs_b(lambda x: -main_func_fourth(x), 1.0, bounds=[(a, b)], approx_grad=True)
    fault = abs(M[1][0]) * ((b - a) ** 5) / (180 * n ** 4)
    while epsilon < fault:
        fault = abs(M[1][0]) * ((b - a) ** 5) / (180 * n ** 4)
        n += 1
    return n, fault


def gaussian_method_fault(a, b):
    n = 1
    M = opt.fmin_l_bfgs_b(lambda x: -main_func_fourth(x), 1.0, bounds=[(a, b)], approx_grad=True)
    fault = abs(M[1][0]) * ((b - a) ** 5) / (180 * n ** 4)
    while epsilon < fault:
        fault = abs(M[1][0]) * ((b - a) ** 5) / (180 * n ** 4)
        n += 1
    return n, fault


print(trapezium_method(a, b))
print(simpson_method(a, b))
