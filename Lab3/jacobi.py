import numpy as np


def jacobi(a: list, b: list, eps=10 ** (-6)) -> list:
    x = [0 for _ in range(len(a[0]))]

    c = []
    d = []
    for i in range(len(a)):
        element_d = b[i] / a[i][i]
        d.append(element_d)

        element_c = []
        for j in range(len(a[0])):
            if i == j:
                element_c.append(0)
            else:
                element_c.append((-1) * a[i][j] / a[i][i])
        c.append(element_c)

    iterations = 0
    while True:
        errors = []
        rightP = np.dot(c, x)
        for i in range(len(x)):
            x_next = rightP[i] + d[i]
            errors.append(abs(x_next - x[i]))
            x[i] = x_next
        if max(errors) < eps:
            break
        else:
            iterations += 1
    print(f'Iterations: {iterations}')
    return x


a = [[2.12, 0.42, 1.34, 0.88],
     [0.42, 3.95, 1.87, 0.43],
     [1.34, 1.87, 2.98, 0.46],
     [0.88, 0.43, 0.46, 4.44]]
b = [11.172, 0.115, 0.009, 9.349]

sol = jacobi(a, b)
print(np.matrix(sol))
print(np.linalg.solve(a, b))
