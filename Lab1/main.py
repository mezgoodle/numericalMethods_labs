import numpy as np
import cmath
import math
from checker import main

"""Simple functions area start"""
x = 10                                                                    # assigning value to a variable
y = pow(x - 3, 2) + 1                                                     # assigning a calculation to a variable
print(float(pow(x, y)) == np.power(10., 50.))                             # comparison
print(y == np.sum([np.power(float(np.sum([x, -3])), float(2)), 1]))
"""Simple functions area end"""

"""Math functions area start"""
print(1 + 3 - 7 == np.sum([1, 3, -7]))                                    # Sum
print(-1 * -2 == np.multiply(-1, -2), -1 * -2 == np.negative(-2))         # Multiply
print(5 / 2 == np.divide(5, 2))                                           # Division
print(2 + 3 / 4 == np.sum([2, np.divide(3, 4)]))
print(math.factorial(5))                                                  # Factorial
print(abs(-10) == np.abs(-10))                                            # Module
print(math.sqrt(4) == np.sqrt(4))                                         # Radical
print(pow(125, (1 / 3)) == np.power(float(125), float(np.divide(1, 3))))
print(math.e ** math.log(3, math.e) == np.power(np.e, float(np.log(3))))  # Logarithm
print(main.get_sum(10, 1))                                                # Summation
print(main.get_product(10, 1))                                            # Product
"""Math functions area end"""

"""Complex numbers area start"""
x = 5                                                                     # Initializing real numbers
y = 3
z = complex(x, y)                                                         # Converting x and y into complex number
z1 = 5 + 3j
z2 = 23 * (cmath.e ** 0.1j)
                                                                          # An complex number is represented by x + yi
print(f'The real part of complex number is : {z.real}')                   # Printing real part of complex number
print(f'The imaginary part of complex number is : {z.imag}')              # Printing imaginary part of complex number
print(np.iscomplex([z, z1, z2]))
print(z + z1, np.iscomplex(z + z1))
print(abs(z2))
"""Complex numbers area end"""

"""Matrix area start"""
a = [[1, 2],                                                              # Create matrix in Python
     [3, 4]]
a1 = np.matrix('1 2; 3 4')                                                # Create matrix in NumPy
print(a == a1)

value = 2                                                                 # Multiply number and matrix
for i in range(len(a)):
    for j in range(len(a[i])):
        a[i][j] *= value
a1 = main.increase_matrix(a1, value)
print(a == a1)
print(a * a1 == a1.dot(a1))
print(a1.getT())                                                          # Transpose of the matrix in NumPy
print(np.linalg.det(a1))                                                  # Get determinant of matrix in NumPy
print(np.linalg.matrix_power(a1, 2))                                      # Raise in power matrix in NumPy
print(np.dot(np.linalg.matrix_power(a1, -1), a1))
"""Matrix area end"""
