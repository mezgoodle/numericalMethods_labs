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
b1 = b.copy()
print(b)
print(b1)
b[2][0] = a[0][3] / a[2][3] * (-1)
b[2][1] = a[1][3] / a[2][3] * (-1)
b[2][2] = 1 / a[2][3]
b[2][3] = a[3][3] / a[2][3] * (-1)
print(b)
b1[2][0] = a[0][3]
b1[2][1] = a[1][3]
b1[2][2] = a[2][3]
b1[2][3] = a[3][3]
print(b1)
temp = np.dot(b1,np.dot(a, b))
print(temp)
