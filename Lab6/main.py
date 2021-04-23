import numpy as np
import matplotlib.pyplot as plt
from string import Template
import math

template = Template('#' * 10 + ' $string ' + '#' * 10)
epsilon = 10 ** (-5)


def y(x):
    return 10 * x ** 5 - 3 * x ** 4 + 7 * x ** 2 - 27


def y_derivative(x):
    return 50 * x ** 4 - 12 * x ** 3 + 14 * x


def y1(x):
    return 10 * x ** 5 - 3 * x ** 4


def y2(x):
    return 27 - 7 * x ** 2


np_roots = np.roots([10, -3, 0, 7, 0, -27])
np_roots = np_roots[np.isreal(np_roots)]
print(np_roots)

x_axis = np.linspace(-2, 2, num=1000)
x_axis_2 = np.linspace(-5, 5, num=1000)
fig, ax = plt.subplots()
ax.plot(x_axis, y1(x_axis), label='y1')
ax.plot(x_axis_2, y2(x_axis_2), label='y2')

ax.legend(loc='lower left', ncol=2)
plt.grid()
plt.show()

intervals = [
    [1, 2]
]


class Polynomial:
    def __init__(self, epsilon, intervals):
        self.epsilon = epsilon
        self.intervals = intervals

    def bisectionMethod(self):
        answers = []
        iterations = 0
        for interval in self.intervals:
            root = 1000
            a, b = interval[0], interval[1]
            while abs(b - a) > self.epsilon and abs(y(root)) > self.epsilon:
                root = (a + b) / 2
                if y(root) * y(a) <= 0:
                    a, b = a, root
                elif y(root) * y(b) <= 0:
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
            if y(a) * y_derivative(a) > 0:
                start_x = a
            else:
                start_x = b
            root = start_x - y(start_x) / y_derivative(start_x)
            iterations += 1
            while abs(y(root)) > self.epsilon:
                root = root - y(root) / y_derivative(root)
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
            while abs(b - a) > self.epsilon and abs(y(root)) > self.epsilon:
                root = (a * y(b) - b * y(a)) / (y(b) - y(a))
                if y(root) * y(a) <= 0:
                    a, b = a, root
                elif y(root) * y(b) <= 0:
                    a, b = root, b
                iterations += 1
            answers.append(root)
        print(template.substitute(string='Chords method'))
        print(f'Answers: {answers}, iterations: {iterations}')

# functions = StartFunctions()
polynomial = Polynomial(epsilon, intervals.copy())
polynomial.bisectionMethod()
polynomial.newtonMethod()
polynomial.chordsMethod()
