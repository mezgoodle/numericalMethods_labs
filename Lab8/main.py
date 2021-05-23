from tabulate import tabulate
import pandas as pd
from string import Template
from math import e

n = 5
a = b = 1 + 0.4 * n
interval = [0, 4]
h = 0.1
x0 = y0 = 0
epsilon = 10 ** (-1)
template = Template('#' * 10 + ' $string ' + '#' * 10)


def dfunction(x: float, y: float) -> float:
    """
    Main diff function
    :param x: x-argument
    :param y: y-argument
    :return: result
    """
    return e ** (-a * x) * (y ** 2 + b)


def runge_kutte_method(interval, h, epsilon, x0, y0):
    rg_res = [[x0, y0]]
    table = []
    table.append([x0, y0])
    xi = x0
    yi = y0
    while xi < interval[1]:
        k1 = h * dfunction(xi, yi)
        k2 = h * dfunction(xi + h / 2, yi + k1 / 2)
        k3 = h * dfunction(xi + h / 2, yi + k2 / 2)
        k4 = h * dfunction(xi + h, yi + k3)
        delta_y = (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        xi += h
        yi += delta_y
        fault = abs((k2 - k3) / (k1 - k2))
        if fault > epsilon:
            h /= 2
        # error = delta_y - (dfunction(xi + h)) / h
        table.append([xi, yi])
        rg_res.append([xi, yi])
    print_table(table, ('x', 'y'))
    return rg_res


def adams_method(interval, h, epsilon, rg_res):
    table = []
    table.append([rg_res[0][0], rg_res[0][1], 0])
    table.append([rg_res[1][0], rg_res[1][1], 0])
    table.append([rg_res[2][0], rg_res[2][1], 0])
    i = 3
    step = h
    while i < ((interval[1] - interval[0]) / h):
        k1 = dfunction(rg_res[i][0], rg_res[i][1])
        k2 = dfunction(rg_res[i - 1][0], rg_res[i - 1][1])
        k3 = dfunction(rg_res[i - 2][0], rg_res[i - 2][1])
        k4 = dfunction(rg_res[i - 3][0], rg_res[i - 3][1])
        extra_y = rg_res[i][1] + h / 24 * (55 * k1 - 59 * k2 + 37 * k3 - 9 * k4)
        next_x = rg_res[i][0] + step
        intra_y = rg_res[i][1] + h / 24 * (9 * dfunction(next_x, extra_y) + 19 * k1 - 5 * k2 + k3)
        fault = abs(intra_y - extra_y)
        if fault > epsilon:
            step / 2
        if extra_y == intra_y:
            table.append([next_x, extra_y, fault])
            rg_res.append([next_x, extra_y])
        else:
            table.append([next_x, intra_y, fault])
            rg_res.append([next_x, intra_y])
        i += 1
    print_table(table, ('x', 'y', 'Fault'))
    return rg_res


def print_table(table: list, headers: tuple):
    df = pd.DataFrame(table)
    print(tabulate(df, headers=headers, tablefmt='github'))


print(template.substitute(string='Runge-kutta method'))
rg_res = runge_kutte_method(interval, h, epsilon, x0, y0)
print(template.substitute(string='Adams method'))
ad_res = adams_method(interval, h, epsilon, rg_res[:4])
