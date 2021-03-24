import numpy as np

np.set_printoptions(suppress=True)

a = [
    [6.26, 1.1, 0.97, 1.24],
    [1.1, 4.16, 1.3, 0.16],
    [0.97, 1.3, 5.44, 2.1],
    [1.24, 0.16, 2.1, 6.1]
]

print('Matrix A:')
print(np.matrix(a))

b = np.zeros((4, 4))
np.fill_diagonal(b, 1)
b1_minus = b.copy()
b2 = b.copy()
b2_minus = b.copy()
b3 = b.copy()
b3_minus = b.copy()

b[2][0] = a[3][0] / a[3][2] * (-1)
b[2][1] = a[3][1] / a[3][2] * (-1)
b[2][2] = 1 / a[3][2]
b[2][3] = a[3][3] / a[3][2] * (-1)
print('Matrix B1:')
print(b)

b1_minus[2][0] = a[3][0]
b1_minus[2][1] = a[3][1]
b1_minus[2][2] = a[3][2]
b1_minus[2][3] = a[3][3]
print('Matrix B1-1:')
print(b1_minus)

temp = np.dot(b1_minus, np.dot(a, b))
print('Matrix temp:')
print(temp)

b2[1][0] = temp[2][0] / temp[2][1] * (-1)
b2[1][1] = 1 / temp[2][1]
b2[1][2] = temp[2][2] / temp[2][1] * (-1)
b2[1][3] = temp[2][3] / temp[2][1] * (-1)
print('Matrix B2:')
print(b2)

b2_minus[1][0] = temp[2][0]
b2_minus[1][1] = temp[2][1]
b2_minus[1][2] = temp[2][2]
b2_minus[1][3] = temp[2][3]
print('Matrix B2-1:')
print(b2_minus)

temp = np.dot(b2_minus, np.dot(temp, b2))
print('Matrix temp:')
print(temp)

b3[0][0] = 1 / temp[1][0]
b3[0][1] = temp[1][1] / temp[1][0] * (-1)
b3[0][2] = temp[1][2] / temp[1][0] * (-1)
b3[0][3] = temp[1][3] / temp[1][0] * (-1)
print('Matrix B3:')
print(b3)

b3_minus[0][0] = temp[1][0]
b3_minus[0][1] = temp[1][1]
b3_minus[0][2] = temp[1][2]
b3_minus[0][3] = temp[1][3]
print('Matrix B3-1:')
print(b3_minus)

temp = np.dot(b3_minus, np.dot(temp, b3))
print('Matrix temp:')
print(temp)
