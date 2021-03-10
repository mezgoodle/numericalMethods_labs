import numpy as np


# Defining our function as seidel which takes 3 arguments
# as A matrix, Solution and B matrix

def seidel(a, x, b):
    x_ = x.copy()
    n = len(a)
    for j in range(0, n):
        d = b[j]

        for i in range(0, n):
            if j != i:
                d -= a[j][i] * x_[i]
        x_[j] = d / a[j][j]
    return x_


x = [0, 0, 0, 0]
a = [[2.12, 0.42, 1.34, 0.88],
     [0.42, 3.95, 1.87, 0.43],
     [1.34, 1.87, 2.98, 0.46],
     [0.88, 0.43, 0.46, 4.44]]
b = [11.172, 0.115, 0.009, 9.349]
eps = 10 ** (-6)
iterations = 0

while True:
    errors = []
    new_x = seidel(a, x, b)
    for i in range(len(x)):
        errors.append(abs(new_x[i] - x[i]))
    x = new_x
    if max(errors) < eps:
        break
    else:
        iterations += 1
    print(x)
print(f'Iterations: ', iterations)
print(np.linalg.solve(a, b))
