import numpy as np
from checker.fault import get_fault

np.set_printoptions(suppress=True)


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
            print(f'Last result: {np.matrix(solution_vector).round(6)}')
            # It's time to stop!
            break
        else:
            if tmp < 3:
                print(f'Temporary result: {np.matrix(solution_vector).round(6)}')
                tmp += 1
            print(f'Residual vector: {np.matrix(np.subtract(vector_b, np.dot(matrix_a, solution_vector)), float).round(6)}')
            iterations += 1
    print(f'Iterations: {iterations}')
    return solution_vector


a = [[4.4944, 0.1764, 1.7956, 0.7744],
     [0.1764, 15.6025, 3.4969, 0.1849],
     [1.7956, 3.4969, 8.8804, 0.2116],
     [0.7744, 0.1849, 0.2116, 19.7136]]
b = [31.97212, 9.18339, 19.51289, 51.39451]
print(f'Matrix A:', np.matrix(a).round(6))
print(f'Vector b:', b)
sol = solve_jacobi(a.copy(), b.copy())
sol_np = np.linalg.solve(a, b)
print(f'Our solution: {np.matrix(sol).round(6)}')
print(f'Residual vector: {np.matrix(np.subtract(b, np.dot(a, sol))).round(6)}')
print(f'NumPy solution: {sol_np}')
print(f'Residual vector for NumPy: {np.matrix(np.subtract(b, np.dot(a, sol_np))).round(6)}')
print('Fault:', round(get_fault(sol, sol_np), 6))
