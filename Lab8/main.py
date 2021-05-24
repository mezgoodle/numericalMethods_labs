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


def runge_kutte_method(limits: list, h_value: float, epsilon_value: float, first_x: float, first_y: float) -> list:
    """
    Implementation of the Runge-Kutta method
    :param limits: limits of x-values
    :param h_value: step
    :param epsilon_value: value for controlling the fault
    :param first_x: known x-value
    :param first_y: known y-value
    :return: list with x and y values
    """
    results = []
    results.append([first_x, first_y, 0])
    current_x = first_x
    current_y = first_y
    while current_x < limits[1]:
        k1 = h_value * dfunction(current_y, current_x)
        k2 = h_value * dfunction(current_y + k1 / 2, current_x + h_value / 2)
        k3 = h_value * dfunction(current_y + k2 / 2, current_x + h_value / 2)
        k4 = h_value * dfunction(current_y + k3, current_x + h_value)
        delta_y = (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        current_x += h_value
        current_y += delta_y
        fault = abs((k2 - k3) / (k1 - k2))
        if fault > epsilon_value:
            h_value /= 2
        results.append([current_x, current_y])
    return results


def adams_method(limits: list, h_value: float, epsilon_value: float, runge_kutta_results: list) -> list:
    """
    Implementation of the Adams method
    :param limits: limits of x-values
    :param h_value: step
    :param epsilon_value: alue for controlling the fault
    :param runge_kutta_results: known results from the previous method
    :return: list with x and y values
    """
    index = 3
    step_value = h_value
    number_of_steps = ((limits[1] - limits[0]) / h_value)
    while index < number_of_steps:
        k1 = dfunction(runge_kutta_results[index][1], runge_kutta_results[index][0])
        k2 = dfunction(runge_kutta_results[index - 1][1], runge_kutta_results[index - 1][0])
        k3 = dfunction(runge_kutta_results[index - 2][1], runge_kutta_results[index - 2][0])
        k4 = dfunction(runge_kutta_results[index - 3][1], runge_kutta_results[index - 3][0])
        extra_y = runge_kutta_results[index][1] + h_value / 24 * (55 * k1 - 59 * k2 + 37 * k3 - 9 * k4)
        next_x = runge_kutta_results[index][0] + step_value
        intra_y = runge_kutta_results[index][1] + h_value / 24 * (9 * dfunction(extra_y, next_x) + 19 * k1 - 5 * k2 + k3)
        fault = abs(intra_y - extra_y)
        if fault > epsilon_value:
            step_value / 2
        if extra_y == intra_y:
            runge_kutta_results.append([next_x, extra_y])
        else:
            runge_kutta_results.append([next_x, intra_y])
        index += 1
    return runge_kutta_results


def search_error_for_runge_kutta(runge_kutta_results: list, h_value: float) -> list:
    """
    Function for calculating errors for Runge-Kutta method
    :param runge_kutta_results: results from this method
    :param h_value: step
    :return: list with errors
    """
    errors = []
    for index in range(len(runge_kutta_results) - 1):
        k1 = dfunction(runge_kutta_results[index][1], runge_kutta_results[index][0])
        k2 = dfunction(runge_kutta_results[index][1], runge_kutta_results[index][0] + h_value / 2)
        k3 = dfunction(runge_kutta_results[index][1], runge_kutta_results[index][0] + h_value / 2)
        k4 = dfunction(runge_kutta_results[index][1], runge_kutta_results[index][0] + h_value)
        delta_y = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        right_part = (runge_kutta_results[index + 1][1] - runge_kutta_results[index][1]) / h_value
        error = delta_y - right_part
        errors.append(error)
    return errors


def search_error_for_adams(adams_results: list, adams_results_less: list) -> list:
    """
    Function for calculating errors for Adams method
    :param adams_results: results from this method
    :param adams_results_less: results from this method with divided step
    :return: list with errors
    """
    errors = []
    for index in range(len(adams_results)):
        error = (adams_results[index][1] - adams_results_less[index * 2][1]) / (16 - 1)
        errors.append(error)
    return errors


def print_table(table: list, headers: tuple) -> None:
    """
    Function for printing the table
    :param table: values
    :param headers: headers of the table
    :return: nothing
    """
    dataframe = pd.DataFrame(table)
    format_style = 'github'
    print(tabulate(dataframe, headers=headers, tablefmt=format_style))


def scipy_solver(x_axis: list, first_y: float) -> list:
    """
    Solve the diff equation with SciPy
    :param x_axis: x-values
    :param first_y: known y-value
    :return: list with results
    """
    results = odeint(dfunction, first_y, x_axis)
    return results


def system_solver():
    y_axis = [0.1, 0]
    x_axis = np.linspace(0, 60, 100)
    results = odeint(system_function, y_axis, x_axis)
    first_y_results = results.transpose()[0]
    second_y_results = results.transpose()[1]
    show_plot_for_system(x_axis, first_y_results, ['u<0>', 'u<1>'])
    show_plot_for_system(x_axis, second_y_results, ['u<0>', 'u<2>'])
    show_plot_for_system(first_y_results, second_y_results, ['u<1>', 'u<2>'])


print(template.substitute(string='Runge-kutta method'))
runge_kutta_results = runge_kutte_method(interval, h, epsilon, x0, y0)
runge_kutta_results_less = runge_kutte_method(interval, h / 2, epsilon, x0, y0)
runge_kutta_errors = search_error_for_adams(runge_kutta_results, runge_kutta_results_less)
for index in range(1, len(runge_kutta_errors)):
    runge_kutta_results[index].append(abs(runge_kutta_errors[index]))
print_table(runge_kutta_results, ('x', 'y', 'Error'))
print(template.substitute(string='Adams method'))
adams_results = adams_method(interval, h, epsilon, runge_kutta_results[:4])
adams_results_less = adams_method(interval, h / 2, epsilon, runge_kutta_results_less[:4])
adams_errors = search_error_for_adams(adams_results, adams_results_less)
for index in range(len(adams_errors)):
    if index < 4:
        adams_results[index][2] = abs(adams_errors[index])
    else:
        adams_results[index].append(abs(adams_errors[index]))
print_table(adams_results, ('x', 'y', 'Error'))
x_axis = np.arange(interval[0], interval[1] + 0.1, h)
scipy_results = scipy_solver(x_axis, y0)
print(template.substitute(string='SciPy solution'))
print_table([[x_axis[i], scipy_results[i]] for i in range(len(scipy_results))], ('x', 'y'))
show_plot([el[0] for el in runge_kutta_results], [el[1] for el in runge_kutta_results], [el[0] for el in adams_results], [el[1] for el in adams_results], ['Runge-Kutta', 'Adams'])
show_plot([el[0] for el in runge_kutta_results], [el[2] for el in runge_kutta_results], [el[0] for el in adams_results], [el[2] for el in adams_results], ['Runge-Kutta errors', 'Adams errors'])
system_solver()
