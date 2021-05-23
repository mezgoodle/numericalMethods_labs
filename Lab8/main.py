from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt
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


def show_plot(x_values: list, y_values: list) -> None:
    """
    Function for creating plots
    :param x_values: our nodes
    :param y_values: values at this nodes
    :param newton_coeffs: coefficients for newton polynomial
    :param spline_coeffs: coefficients for spline equations
    :param indexes: dictionary with indexes for spline equations
    :param cubic_spline: CubicSpline class from SciPy
    :return: nothing to return
    """
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values, 'o', label='Data')
    ax.plot(x_axis, [linear_function(x) for x in x_axis], label='Linear')
    ax.legend(loc='lower left', ncol=2)
    plt.grid()
    plt.show()



def runge_kutte_method(interval, h, epsilon, x0, y0):
    rg_res = [[x0, y0]]
    table = []
    table.append([x0, y0, 0])
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
        table.append([xi, yi])
        rg_res.append([xi, yi])
    return rg_res, table


def adams_method(interval, h, epsilon, rg_res):
    table = []
    table.append([rg_res[0][0], rg_res[0][1]])
    table.append([rg_res[1][0], rg_res[1][1]])
    table.append([rg_res[2][0], rg_res[2][1]])
    table.append([rg_res[3][0], rg_res[3][1]])
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
            table.append([next_x, intra_y])
            rg_res.append([next_x, intra_y])
        i += 1
    return rg_res, table


def search_error_for_rg(rg_res, h):
    errors = []
    for i in range(len(rg_res) - 1):
        k1 = dfunction(rg_res[i][0], rg_res[i][1])
        k2 = dfunction(rg_res[i][0] + h / 2, rg_res[i][1])
        k3 = dfunction(rg_res[i][0] + h / 2, rg_res[i][1])
        k4 = dfunction(rg_res[i][0] + h, rg_res[i][1])
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


print(template.substitute(string='Runge-kutta method'))
rg_res, table = runge_kutte_method(interval, h, epsilon, x0, y0)
rg_res_less, table_ = runge_kutte_method(interval, h / 2, epsilon, x0, y0)
errors = search_error_for_ad(rg_res, rg_res_less)
for i in range(1, len(errors)):
    table[i].append(abs(errors[i]))
print_table(table, ('x', 'y', 'Error'))
print(template.substitute(string='Adams method'))
ad_res, table_finish = adams_method(interval, h, epsilon, rg_res[:4])
ad_res_less, table = adams_method(interval, h / 2, epsilon, rg_res_less[:4])
errors = search_error_for_ad(ad_res, ad_res_less)
for i in range(len(errors)):
    table_finish[i].append(abs(errors[i]))
print_table(table_finish, ('x', 'y', 'Error'))
