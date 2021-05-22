from tabulate import tabulate
import pandas as pd
import numpy as np
from math import e

n = 5
a = b = 1 + 0.4 * n
interval = [0, 4]
h = 0.1
x0 = y0 = 0
epsilon = 10 ** (-1)


def dfunction(x: float, y: float) -> float:
    """
    Main diff function
    :param x: x-argument
    :param y: y-argument
    :return: result
    """
    return e ** (-a * x) * (y ** 2 + b)


def runge_kutte_method(interval, h, epsilon, x0, y0):
    table = []
    table.append([x0, y0, 0, 0])
    xi = interval[0]
    yi = y0
    while xi <= interval[1]:
        k1 = h * dfunction(xi, yi)
        k2 = h * dfunction(xi * h / 2, yi + k1 / 2)
        k3 = h * dfunction(xi * h / 2, yi + k2 / 2)
        k4 = h * dfunction(xi * h, yi + k3)
        delta_y = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        xi += h
        yi += delta_y
        fault = abs((k2 - k3) / (k1 - k2))
        if fault > epsilon:
            h /= 2
        table.append([xi, yi, delta_y, fault])
        print_table(table, ('x', 'y', 'Delta y', 'Fault'))
        input()
    print_table(table, ('x', 'y', 'Delta y', 'Fault'))


def print_table(table: list, headers: tuple):
    df = pd.DataFrame(table)
    # # displaying the DataFrame
    print(tabulate(df, headers=headers, tablefmt='github'))


runge_kutte_method(interval, h, epsilon, x0, y0)
