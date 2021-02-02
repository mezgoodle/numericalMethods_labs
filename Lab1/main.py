import numpy as np
from math import factorial, sqrt, e, log
from checker import main

# Calculating area start

"""Simple functions"""
x = 10                                                                    # assigning value to a variable
y = pow(x - 3, 2) + 1                                                     # assigning a calculation to a variable
print(float(pow(x, y)) == np.power(10., 50.))                             # comparison
print(y == np.sum([np.power(float(np.sum([x, -3])), float(2)), 1]))

"""Math functions"""
print(1 + 3 - 7 == np.sum([1, 3, -7]))                                    # Sum
print(-1 * -2 == np.multiply(-1, -2), -1 * -2 == np.negative(-2))         # Multiply
print(5 / 2 == np.divide(5, 2))                                           # Division
print(2 + 3 / 4 == np.sum([2, np.divide(3, 4)]))
print(factorial(5))                                                       # Factorial
print(abs(-10) == np.abs(-10))                                            # Module
print(sqrt(4) == np.sqrt(4))                                              # Radical
print(pow(125, (1 / 3)) == np.power(float(125), float(np.divide(1, 3))))
print(e ** log(3, e) == np.power(np.e, float(np.log(3))))                 # Logarithm
print(main.get_sum(10, 1))                                                # Summation
print(main.get_product(10, 1))                                            # Product

# Calculating area end
