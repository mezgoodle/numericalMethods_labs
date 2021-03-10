import numpy as np


# Defining our function as seidel which takes 3 arguments
# as A matrix, Solution and B matrix

def seidel(matrix_a: list, vector_b: list, vector_x: list) -> list:
    """
    Seidel algorithm part
    :param matrix_a: start matrix
    :param vector_b: start vector
    :param vector_x: solution vector
    :return:
    """
    x_ = vector_x.copy()
    n = len(matrix_a)
    for j in range(0, n):
        d = vector_b[j]

        for i in range(0, n):
            if j != i:
                d -= matrix_a[j][i] * x_[i]
        x_[j] = d / matrix_a[j][j]
    return x_


def solve(matrix_a: list, vector_b: list, vector_x: list, eps=10 ** (-6)) -> list:
    """
    Main funtion
    :param matrix_a: start matrix
    :param vector_b: start vector
    :param vector_x: solution vector
    :param eps: epsilon for comparing
    :return:
    """
    iterations = 0
    while True:
        errors = []
        new_x = seidel(matrix_a, vector_b, vector_x)
        for i in range(len(vector_x)):
            errors.append(abs(new_x[i] - vector_x[i]))
        vector_x = new_x
        if max(errors) < eps:
            break
        else:
            iterations += 1
    print(f'Iterations: ', iterations)
    return vector_x


a = [[2.12, 0.42, 1.34, 0.88],
     [0.42, 3.95, 1.87, 0.43],
     [1.34, 1.87, 2.98, 0.46],
     [0.88, 0.43, 0.46, 4.44]]
b = [11.172, 0.115, 0.009, 9.349]
x = [0 for _ in range(len(a[0]))]
x = solve(a.copy(), b.copy(), x.copy())
print(x)

print(np.linalg.solve(a, b))
