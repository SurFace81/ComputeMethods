import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_file():
    file = pd.read_csv("11.csv")
    file = file.iloc[:, :-1]
    print(file, "\n.............................................................................................\n\n")
    data = {}
    for index, row in file.iterrows():
        year = int(row.iloc[0])
        values = row.iloc[1:].tolist()
        data[year] = values
    return data

def graph(x, y, x_interp, y_interp):
    plt.plot(x, y, "o")
    plt.plot(x_interp, y_interp, color='red')
    plt.grid(True)
    plt.show()