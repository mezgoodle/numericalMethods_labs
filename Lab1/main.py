import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import cmath
import math
from checker import main

"""Simple functions area start"""
x = 12                                                                      # assigning value to a variable
y = pow(x - 4, 2) + 5                                                       # assigning a calculation to a variable
print(float(pow(x, y)) == np.power(12., 69.))                               # comparison
print(y == np.sum([np.power(float(np.sum([x, -4])), 2.), 5]))
"""Simple functions area end"""

"""Math functions area start"""
print(3 + 4 - 5 == np.sum([3, 4, -5]))                                      # Sum
print(-1 * -4 == np.multiply(-1, -4), -1 * -4 == np.negative(-4))           # Multiply
print(7 / 2 == np.divide(7, 2))                                             # Division
print(3 + 3 / 5 == np.sum([3, np.divide(3, 5)]))
print(math.factorial(7) == np.math.factorial(7))                            # Factorial
print(abs(-17) == np.abs(-17))                                              # Module
print(math.sqrt(9) == np.sqrt(9))                                           # Radical
print(pow(8, (1 / 3)) == np.power(float(8), float(np.divide(1, 3))))
print(math.e ** math.log(50, math.e) == np.power(np.e, float(np.log(50))))  # Logarithm
print(main.get_sum(11, 2))                                                  # Summation
print(main.get_product(11, 2))                                              # Product
"""Math functions area end"""

"""Complex numbers area start"""
x = 6                                                                       # Initializing real numbers
y = 2
z = complex(x, y)                                                           # Converting x and y into complex number
z1 = 6 + 2j
z2 = 21 * (cmath.e ** 0.7j)
                                                                            # An complex number is represented by x + yi
print(f'The real part of complex number is : {z.real}')                     # Printing real part of complex number
print(f'The imaginary part of complex number is : {z.imag}')                # Printing imaginary part of complex number
print(np.iscomplex([z, z1, z2]))
print(z + z1, np.iscomplex(z + z1))
print(abs(z2) == np.abs(z2))
"""Complex numbers area end"""

"""Matrix area start"""
a = [[4, 6],                                                                # Create matrix in Python
     [5, 7]]
a1 = np.matrix('4 6; 5 7')                                                  # Create matrix in NumPy
print(a == a1)

value = 3                                                                   # Multiply number and matrix
for i in range(len(a)):
    for j in range(len(a[i])):
        a[i][j] *= value
a1 = main.increase_matrix(a1, value)
print(a == a1)
print(a * a1 == a1.dot(a1))
print(a1.getT())                                                            # Raise in power matrix in NumPy
print(np.dot(np.linalg.matrix_power(a1, -1), a1))
print(np.linalg.det(a))                                                     # Print determination
"""Matrix area end"""

"""Vectors area start"""
a = np.array([1, 2, 3])                                                     # Create the vectors
b = np.array([3, 3, 2])
print(np.vdot(a, b))                                                        # Scalar product
"""Vectors area end"""

"""Graphs area start"""
# 2D graphic
k = 3
t = np.arange(0.0, 5.01, 0.01)                                              # Data for plotting
s = k / t

fig, ax = plt.subplots()                                                    # Get some variables from plot
ax.plot(t, s)

ax.set(xlabel='x', ylabel='y',                                              # Set labels
       title='y=k/x')
ax.grid()                                                                   # Set grid

fig.savefig('Lab1/graphic1.png')                                                  # Save graphic as image
plt.show()                                                                  # Show the plot

# 3D graphic
fig = plt.figure()
ax = plt.axes(projection='3d')                                              # Set projection to 3d

z_line = np.linspace(0, 15, 1000)                                           # Create data
x_line = np.sin(z_line)                                                     # Tg from z
y_line = np.sin(z_line)                                                     # Sin from z
ax.plot3D(x_line, y_line, z_line, 'gray')                                   # Add data to plot
fig.savefig('Lab1/graphic2.png')                                                  # Save graphic
plt.show()                                                                  # Show the plot

# Another 3D graphic
fig = plt.figure()
ax = fig.gca(projection='3d')
# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = np.power(X, 3) + np.power(Y, 3)
# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.savefig('Lab1/graphic3.png')
plt.show()
"""Graphs area end"""
