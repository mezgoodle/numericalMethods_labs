# importing the modules
from tabulate import tabulate
import pandas as pd
import numpy as np
from math import e

n = 5
a = b = 1 + 0.4 * n
interval = [0, 4]
h = 0.1


def dfunction(x: float, y: float) -> float:
    """
    Main diff function
    :param x: x-argument
    :param y: y-argument
    :return: result
    """
    return e ** (-a * x) * (y ** 2 + b)


def prepare_table(length):
    table = []
    for _ in range(length):
        table.append(['' for _ in range(8)])
    return table


# creating a DataFrame
table = prepare_table(7)
table[1][0] = 1
print(np.matrix(table))
df = pd.DataFrame(table)

# # displaying the DataFrame
print(tabulate(df, headers=('k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k'), tablefmt='github'))
