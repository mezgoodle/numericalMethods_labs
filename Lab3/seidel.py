import numpy as np


# Defining our function as seidel which takes 3 arguments
# as A matrix, Solution and B matrix

def seidel(a, x, b):
    x_ = x.copy()
    # Finding length of a(3)
    n = len(a)
    # for loop for 3 times as to calculate x, y , z
    for j in range(0, n):
        # temp variable d to store b[j]
        d = b[j]

        # to calculate respective xi, yi, zi
        for i in range(0, n):
            if j != i:
                d -= a[j][i] * x_[i]
            # updating the value of our solution
        x_[j] = d / a[j][j]
    # returning our updated solution
    return x_


# initial solution depending on n(here n=3)
x = [0, 0, 0, 0]
a = [[2.12, 0.42, 1.34, 0.88],
     [0.42, 3.95, 1.87, 0.43],
     [1.34, 1.87, 2.98, 0.46],
     [0.88, 0.43, 0.46, 4.44]]
b = [11.172, 0.115, 0.009, 9.349]
eps = 10 ** (-6)
iterations = 0

# loop run for m times depending on m the error value
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
    # print each time the updated solution
    print(x)
print(f'Iterations: ', iterations)
print(np.linalg.solve(a, b))
