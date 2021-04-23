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

    def newtonMethod(self):
        answers = []
        iterations = 0
        for interval in self.intervals:
            start_x = 0
            a, b = interval[0], interval[1]
            if self.function.y(a) * self.function.y_derivative(a) > 0:
                start_x = a
            else:
                start_x = b
            root = start_x - self.function.y(start_x) / self.function.y_derivative(start_x)
            iterations += 1
            while abs(self.function.y(root)) > self.epsilon:
                root = root - self.function.y(root) / self.function.y_derivative(root)
                iterations += 1
            answers.append(root)
        print(template.substitute(string='Newton method'))
        print(f'Answers: {answers}, iterations: {iterations}')

    def chordsMethod(self):
        answers = []
        iterations = 0
        for interval in self.intervals:
            root = 1000
            a, b = interval[0], interval[1]
            while abs(b - a) > self.epsilon and abs(self.function.y(root)) > self.epsilon:
                root = (a * self.function.y(b) - b * self.function.y(a)) / (self.function.y(b) - self.function.y(a))
                if self.function.y(root) * self.function.y(a) <= 0:
                    a, b = a, root
                elif self.function.y(root) * self.function.y(b) <= 0:
                    a, b = root, b
                iterations += 1
            answers.append(root)
        print(template.substitute(string='Chords method'))
        print(f'Answers: {answers}, iterations: {iterations}')


functions = StartFunctions()
polynomial = Polynomial(epsilon, intervals.copy(), functions)
polynomial.bisectionMethod()
polynomial.newtonMethod()
polynomial.chordsMethod()
