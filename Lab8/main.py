# importing the modules
from tabulate import tabulate
import pandas as pd
import numpy as np


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
