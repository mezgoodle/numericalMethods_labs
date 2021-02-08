import numpy as np


def methodGaussian(start_matrix: list) -> list:
    """
    The methodGaussian function.
    This function get matrix like this:
    [[A11X1 + A12X2 + A13X3+...+A1nXn=B1],
    [A21X1 + A22X2 + A23X3+...+a2nxn=B2],
    [A31X1 + A32X2 + A33X3+...+A3nXn=B3]
    ...
    [an1x1 + an2x2 + an3x3+...+annxn=bn]]

    Parameters:
        start_matrix (list): The start matrix.

    Returns:
        list: The result matrix.
    """
    final_matrix = start_matrix.copy()  # Create copy.
    rows = len(final_matrix)  # Get number or rows
    columns = len(final_matrix[0])  # Get number of columns
    results = [None] * rows  # Create list of results

    # Main work through matrix
    for row in range(rows):
        for i in range(row+1, rows):
            value = final_matrix[i][row] / final_matrix[row][row]
            for j in range(row, columns):
                final_matrix[i][j] = final_matrix[i][j] + final_matrix[row][j] * (-1) * value

    print('Triangular matrix:', np.matrix(final_matrix))  # Print triangular matrix

    # Search solutions
    for row in range(rows-1, -1, -1):
        b = final_matrix[row][columns-1]
        for row_ in range(row+1, columns-1):
            b -= final_matrix[row][row_] * results[row_]
        x = b / final_matrix[row][row]
        results[row] = round(x)
    return results


matrix = [[3, 2, 1, 1, -2], [1, -1, 4, -1, -1], [-2, -2, -3, 1, 9], [1, 5, -1, 2, 4]]
a = [[3, 2, 1, 1], [1, -1, 4, -1], [-2, -2, -3, 1], [1, 5, -1, 2]]
b = [-2, -1, 9, 4]
print('Start matrix:', np.matrix(matrix))
print('Solutions:', methodGaussian(matrix))
print('NumPy solution:', np.linalg.solve(a, b))
