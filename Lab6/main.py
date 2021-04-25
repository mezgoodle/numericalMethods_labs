import numpy as np
import matplotlib.pyplot as plt
from string import Template


template = Template('#' * 10 + ' $string ' + '#' * 10)
epsilon = 10 ** (-6)


def y(x):
    return 10 * x ** 5 - 3 * x ** 4 + 7 * x ** 2 - 27


def y_derivative(x):
    return 50 * x ** 4 - 12 * x ** 3 + 14 * x


def y1(x):
    return 10 * x ** 5 - 3 * x ** 4


def y2(x):
    return 27 - 7 * x ** 2


intervals = [
    [1.1, 1.3]
]


class Polynomial:
    def __init__(self, epsilon, intervals, roots):
        self.epsilon = epsilon
        self.intervals = intervals
        self.roots = roots

    @classmethod
    def printPolynomial(cls):
        print(template.substitute(string='Start polynomial'))
        polynom = np.polynomial.Polynomial([27, 0, 7, 0, -3, 10])
        print(polynom)

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
        self.get_faults(answers, self.roots, 'bisection')

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
        self.get_faults(answers, self.roots, 'newton')

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
        self.get_faults(answers, self.roots, 'chords')

    def get_faults(self, self_values, true_values, method):
        print(template.substitute(string=f'Fault for {method} method'))
        fault = 0
        for index in range(len(self_values)):
            fault = abs(true_values[index] - self_values[index])
        print(round(fault, 6))


np_roots = np.roots([10, -3, 0, 7, 0, -27])
print(template.substitute(string='All roots from NumPy'))
print(np_roots)
print(template.substitute(string='Real roots from NumPy'))
np_roots = np_roots[np.isreal(np_roots)]
print(np_roots)

polynomial = Polynomial(epsilon, intervals.copy(), np_roots.copy())
polynomial.printPolynomial()
polynomial.bisectionMethod()
polynomial.newtonMethod()
polynomial.chordsMethod()

x_axis = np.linspace(-2, 2, num=1000)
x_axis_2 = np.linspace(-5, 5, num=1000)
fig, ax = plt.subplots()
ax.plot(x_axis, y(x_axis), label='y')
ax.plot(x_axis, y1(x_axis), label='y1')
ax.plot(x_axis_2, y2(x_axis_2), label='y2')

ax.legend(loc='lower left', ncol=2)
plt.grid()
plt.show()
