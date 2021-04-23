import numpy as np
import matplotlib.pyplot as plt
from string import Template
import math

template = Template('#' * 10 + ' $string ' + '#' * 10)
epsilon = 10 ** (-5)


class StartFunctions:
    @staticmethod
    def y(x):
        return -2 * x ** 4 + x ** 3 + 5 * x ** 2 - 2 * x + 3

    @staticmethod
    def y_derivative(x):
        return -8 * x ** 3 + 3 * x ** 2 + 10 * x - 2

    @staticmethod
    def y1(x):
        return -2 * x ** 4 + x ** 3

    @staticmethod
    def y2(x):
        return -5 * x ** 2 + 2 * x - 3


# np_roots = np.roots([-2, 1, 5, -2, +3])
# np_roots = np_roots[np.isreal(np_roots)]
# print(np_roots)
#
# x_axis = np.linspace(-5, 5, num=10000)
# x_axis_2 = np.linspace(-5, 5, num=10000)
# fig, ax = plt.subplots()
# # ax.plot(x_axis, [y(x) for x in x_axis], label='Data')
# ax.plot(x_axis, [y1(x) for x in x_axis], label='Data')
# ax.plot(x_axis, [y2(x) for x in x_axis], label='Data')
# plt.grid()
# plt.show()

intervals = [
    [-2, -1.5],
    [1.5, 2]
]


class Polynomial:
    def __init__(self, epsilon, intervals, function):
        self.epsilon = epsilon
        self.intervals = intervals
        self.function = function

    def bisectionMethod(self):
        answers = []
        iterations = 0
        for interval in self.intervals:
            root = 1000
            a, b = interval[0], interval[1]
            while abs(b - a) > self.epsilon and abs(self.function.y(root)) > self.epsilon:
                root = (a + b) / 2
                if self.function.y(root) * self.function.y(a) <= 0:
                    a, b = a, root
                elif self.function.y(root) * self.function.y(b) <= 0:
                    a, b = root, b
                iterations += 1
            answers.append(root)
        print(template.substitute(string='Bisection method'))
        print(f'Answers: {answers}, iterations: {iterations}')


polynomial = Polynomial(epsilon, intervals.copy(), StartFunctions())
polynomial.bisectionMethod()

# def Bisection(i, f):
#     counter = 0
#     xs = []
#     for k in range(len(i)):
#         x = (i[k][0] + i[k][1]) / 2
#         while math.fabs(f(x)) >= epsilon:
#             counter += 1
#             x = (i[k][0] + i[k][1]) / 2
#             i[k][0], i[k][1] = (i[k][0], x) if f(i[k][0]) * f(x) < 0 else (x, i[k][1])
#         xs.append((i[k][0] + i[k][1]) / 2)
#     return xs, counter
#
#
# bis, count1 = Bisection(intervals, y)
#
# print(f"Результати методу бiсекції: {bis}\nКiлькiсть iтерацiй: {count1}")
#
#
# def Newton(i, f, f1):
#     counter = 0
#     xs = []
#     for k in range(len(i)):
#         x0 = (i[k][0] + i[k][1]) / 2
#         x1 = x0 - (f(x0) / f1(x0))
#         while True:
#             counter += 1
#             if math.fabs(x1 - x0) < epsilon:
#                 xs.append(x1)
#                 break
#             x0 = x1
#             x1 = x0 - (f(x0) / f1(x0))
#     return xs, counter
#
#
# new, count2 = Newton(intervals, y, y_pohidna)
#
# print(f"Результати методу Ньютона: {new}\nКiлькiсть iтерацiй: {count2}")
#
#
# def Chorde(i, f):
#     counter = 0
#     xs = []
#     for k in range(len(i)):
#         x = i[k][0] + (f(i[k][1]) * (i[k][1] - i[k][0])) / (f(i[k][1]) - f(i[k][0]))
#         while math.fabs(f(x)) >= epsilon:
#             counter += 1
#             x = i[k][0] + (f(i[k][1]) * (i[k][1] - i[k][0])) / (f(i[k][1]) - f(i[k][0]))
#             i[k][0], i[k][1] = (i[k][0], x) if f(i[k][0]) * f(x) < 0 else (x, i[k][1])
#         xs.append((i[k][0] + i[k][1]) / 2)
#     return xs, counter
#
#
# chor, count3 = Chorde(intervals, y)
#
# print(f"Результати методу хорд: {chor}\nКiлькiсть iтерацiй: {count3}")
