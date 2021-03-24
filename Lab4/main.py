import numpy as np

np.set_printoptions(suppress=True)

a = [
    [6.26, 1.1, 0.97, 1.24],
    [1.1, 4.16, 1.3, 0.16],
    [0.97, 1.3, 5.44, 2.1],
    [1.24, 0.16, 2.1, 6.1]
]

print(np.matrix(a))

b = np.zeros((4, 4))
np.fill_diagonal(b, 1)
b_minus1 = b.copy()
print(b)
print(b_minus1)

b[2][0] = a[3][0] / a[3][2] * (-1)
b[2][1] = a[3][1] / a[3][2] * (-1)
b[2][2] = 1 / a[3][2]
b[2][3] = a[3][3] / a[3][2] * (-1)
print(b)

b_minus1[2][0] = a[3][0]
b_minus1[2][1] = a[3][1]
b_minus1[2][2] = a[3][2]
b_minus1[2][3] = a[3][3]
print(b_minus1)

temp = np.dot(b_minus1, np.dot(a, b))
print(temp)
