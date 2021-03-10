import numpy as np
from checker.fault import get_fault


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
    tmp = 0
    while True:
        errors = []
        new_x = seidel(matrix_a, vector_b, vector_x)
        for i in range(len(vector_x)):
            errors.append(abs(new_x[i] - vector_x[i]))
        vector_x = new_x
        if max(errors) < eps:
            print(f'Last result: {vector_x}')
            break
        else:
            if tmp < 3:
                print(f'Temporary result: {vector_x}')
                tmp += 1
            print(f'Residual vector: {np.matrix(np.subtract(vector_b, np.dot(matrix_a, vector_x)), float)}')
            iterations += 1
    print(f'Iterations: {iterations}')
    return vector_x


a = [[2.12, 0.42, 1.34, 0.88],
     [0.42, 3.95, 1.87, 0.43],
     [1.34, 1.87, 2.98, 0.46],
     [0.88, 0.43, 0.46, 4.44]]
b = [11.172, 0.115, 0.009, 9.349]
x = [0 for _ in range(len(a[0]))]
x = solve(a.copy(), b.copy(), x.copy())
print(f'Our solution: {x}')
x_np = np.linalg.solve(a, b)
print(f'NumPy solution: {x_np}')
print(f'Residual vector: {np.matrix(np.subtract(b, np.dot(a, x)), int)}')
print(f'Residual vector for NumPy: {np.matrix(np.subtract(b, np.dot(a, x_np)), int)}')
print('Fault:', round(get_fault(x, x_np), 6))
