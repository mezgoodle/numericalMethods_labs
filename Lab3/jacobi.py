import numpy as np
from math import sqrt


def solve_jacobi(matrix_a: list, vector_b: list, epsilon=10 ** (-6)) -> list:
    """
    Solve by Jacobi method(simple iteration)
    :param matrix_a: start matrix
    :param vector_b: start vector
    :param epsilon: number for comparing
    :return: solution vector
    """
    solution_vector = [0 for _ in range(len(matrix_a[0]))]

    matrix_c = []
    vector_d = []
    # Create matrix C and vector D
    for i in range(len(matrix_a)):
        element_d = vector_b[i] / matrix_a[i][i]
        vector_d.append(element_d)

        element_c = []
        for j in range(len(matrix_a[0])):
            if i == j:
                element_c.append(0)
            else:
                element_c.append((-1) * matrix_a[i][j] / matrix_a[i][i])
        matrix_c.append(element_c)

    iterations = 0
    tmp = 0
    # Start the main algorithm
    while True:
        divs = []
        left_part = np.dot(matrix_c, solution_vector)
        for i in range(len(solution_vector)):
            x_next = left_part[i] + vector_d[i]
            divs.append(abs(x_next - solution_vector[i]))
            solution_vector[i] = x_next
        # Check if we need to stop
        if max(divs) < epsilon:
            print(f'Last result: {solution_vector}')
            # It's time to stop!
            break
        else:
            if tmp < 3:
                print(f'Temporary result: {solution_vector}')
                tmp += 1
            print(f'Residual vector: {np.matrix(np.subtract(b, np.dot(a, solution_vector)), float)}')
            iterations += 1
    print(f'Iterations: {iterations}')
    return solution_vector


def get_fault(x: list, xm: list) -> float:
    """
    Function for finding get_fault
    :param x: my solution vector
    :param xm: NumPy solution vector
    :return: fault
    """
    sum_k = 0
    n = len(x)
    for k in range(1, n):
        sum_k += (x[k] - xm[k]) ** 2
    result = sqrt(sum_k / n)
    return result


a = [[2.12, 0.42, 1.34, 0.88],
     [0.42, 3.95, 1.87, 0.43],
     [1.34, 1.87, 2.98, 0.46],
     [0.88, 0.43, 0.46, 4.44]]
b = [11.172, 0.115, 0.009, 9.349]
print(f'Matrix A:', np.matrix(a))
print(f'Vector b:', b)
sol = solve_jacobi(a.copy(), b.copy())
sol_np = np.linalg.solve(a, b)
print(f'Our solution: {np.matrix(sol)}')
print(f'NumPy solution: {sol_np}')
print(f'Residual vector: {np.matrix(np.subtract(b, np.dot(a, sol)), int)}')
print(f'Residual vector for NumPy: {np.matrix(np.subtract(b, np.dot(a, sol_np)), int)}')
print('Fault:', round(get_fault(sol, sol_np), 6))
