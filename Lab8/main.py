from tabulate import tabulate
import pandas as pd
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from string import Template
from math import e

# Constants
n = 5
a = b = 1 + 0.4 * n
interval = [0, 4]
h = 0.1
x0 = y0 = 0
epsilon = 10 ** (-1)
template = Template('#' * 10 + ' $string ' + '#' * 10)


def dfunction(y: float, x: float) -> float:
    """
    Main diff function
    :param x: x-argument
    :param y: y-argument
    :return: result
    """
    return e ** (-a * x) * (y ** 2 + b)


def system_function(y: list, x: float) -> list:
    """
    Main diff system
    :param y: y-argument
    :param x: x-argument
    :return: list of results, tow arguments
    """
    k = 10
    return [y[1], ((k - 10) / 10 * y[1]) - y[0]]


def show_plot(x_axis_1: list, y_axis_1: list, x_axis_2: list, y_axis_2: list, labels: list) -> None:
    """
    Function for showing plot
    :param x_axis_1: x-values for the first function
    :param y_axis_1: y-values for the first function
    :param x_axis_2: x-values for the second function
    :param y_axis_2: y-values for the second function
    :param labels: labels on a plot
    :return: nothing to return
    """
    fig, ax = plt.subplots()
    ax.plot(x_axis_1, y_axis_1, label=labels[0])
    ax.plot(x_axis_2, y_axis_2, label=labels[1])
    ax.legend(loc='upper left', ncol=2)
    plt.grid()
    plt.show()


def show_plot_for_system(x_axis: list, y_axis: list, labels: list) -> None:
    """
    Function for showing plots for system
    :param x_axis: x-values of the function
    :param y_axis: y-values of the function
    :param labels: labels on a plot
    :return: nothing to return
    """
    plt.title('Portrait')
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.grid()
    plt.plot(x_axis, y_axis, 'k')
    plt.show()


def runge_kutte_method(interval, h, epsilon, x0, y0):
    table = []
    table.append([x0, y0, 0])
    xi = x0
    yi = y0
    while xi < interval[1]:
        k1 = h * dfunction(yi, xi)
        k2 = h * dfunction(yi + k1 / 2, xi + h / 2)
        k3 = h * dfunction(yi + k2 / 2, xi + h / 2)
        k4 = h * dfunction(yi + k3, xi + h)
        delta_y = (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        xi += h
        yi += delta_y
        fault = abs((k2 - k3) / (k1 - k2))
        if fault > epsilon:
            h /= 2
        table.append([xi, yi])
    return table


def adams_method(interval, h, epsilon, rg_res):
    i = 3
    step = h
    while i < ((interval[1] - interval[0]) / h):
        k1 = dfunction(rg_res[i][1], rg_res[i][0])
        k2 = dfunction(rg_res[i - 1][1], rg_res[i - 1][0])
        k3 = dfunction(rg_res[i - 2][1], rg_res[i - 2][0])
        k4 = dfunction(rg_res[i - 3][1], rg_res[i - 3][0])
        extra_y = rg_res[i][1] + h / 24 * (55 * k1 - 59 * k2 + 37 * k3 - 9 * k4)
        next_x = rg_res[i][0] + step
        intra_y = rg_res[i][1] + h / 24 * (9 * dfunction(extra_y, next_x) + 19 * k1 - 5 * k2 + k3)
        fault = abs(intra_y - extra_y)
        if fault > epsilon:
            step / 2
        if extra_y == intra_y:
            rg_res.append([next_x, extra_y])
        else:
            rg_res.append([next_x, intra_y])
        i += 1
    return rg_res


def search_error_for_rg(rg_res, h):
    errors = []
    for i in range(len(rg_res) - 1):
        k1 = dfunction(rg_res[i][1], rg_res[i][0])
        k2 = dfunction(rg_res[i][1], rg_res[i][0] + h / 2)
        k3 = dfunction(rg_res[i][1], rg_res[i][0] + h / 2)
        k4 = dfunction(rg_res[i][1], rg_res[i][0] + h)
        delta_y = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        right_part = (rg_res[i + 1][1] - rg_res[i][1]) / h
        error = delta_y - right_part
        errors.append(error)
    return errors


def search_error_for_ad(ad_res, ad_res_less):
    errors = []
    for i in range(len(ad_res)):
        error = (ad_res[i][1] - ad_res_less[i * 2][1]) / (16 - 1)
        errors.append(error)
    return errors


def print_table(table: list, headers: tuple):
    df = pd.DataFrame(table)
    print(tabulate(df, headers=headers, tablefmt='github'))


def solve_np(x_axis, y0):
    results = odeint(dfunction, y0, x_axis)
    return results


def solve_system():
    y_axis = [0.1, 0]
    x_axis = np.linspace(0, 60, 100)
    results = odeint(system_function, y_axis, x_axis)
    first_y_results = results.transpose()[0]
    second_y_results = results.transpose()[1]
    show_plot_for_system(x_axis, first_y_results, ['u<0>', 'u<1>'])
    show_plot_for_system(x_axis, second_y_results, ['u<0>', 'u<2>'])
    show_plot_for_system(first_y_results, second_y_results, ['u<1>', 'u<2>'])


# print(template.substitute(string='Runge-kutta method'))
# rg_res = runge_kutte_method(interval, h, epsilon, x0, y0)
# rg_res_less = runge_kutte_method(interval, h / 2, epsilon, x0, y0)
# errors = search_error_for_ad(rg_res, rg_res_less)
# for i in range(1, len(errors)):
#     rg_res[i].append(abs(errors[i]))
# print_table(rg_res, ('x', 'y', 'Error'))
# print(template.substitute(string='Adams method'))
# ad_res = adams_method(interval, h, epsilon, rg_res[:4])
# ad_res_less = adams_method(interval, h / 2, epsilon, rg_res_less[:4])
# errors = search_error_for_ad(ad_res, ad_res_less)
# for i in range(len(errors)):
#     if i < 4:
#         ad_res[i][2] = abs(errors[i])
#     else:
#         ad_res[i].append(abs(errors[i]))
# print_table(ad_res, ('x', 'y', 'Error'))
# x_axis = np.arange(interval[0], interval[1] + 0.1, h)
# np_res = solve_np(x_axis, y0)
# print(template.substitute(string='SciPy solution'))
# print_table([[x_axis[i], np_res[i]] for i in range(len(np_res))], ('x', 'y'))
# show_plot([el[0] for el in rg_res], [el[1] for el in rg_res], [el[0] for el in ad_res], [el[1] for el in ad_res], ['Runge-Kutta', 'Adams'])
# show_plot([el[0] for el in rg_res], [el[2] for el in rg_res], [el[0] for el in ad_res], [el[2] for el in ad_res], ['Runge-Kutta errors', 'Adams errors'])

solve_system()
