import numpy as np
import scipy
import scipy.optimize as opt
from math import cos, sin
from string import Template

a, b = 0.8, 1.7
epsilon = 10 ** (-5)
template = Template('#' * 10 + ' $string ' + '#' * 10)


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


print(trapezium_method(a, b))
print(simpson_method(a, b))
