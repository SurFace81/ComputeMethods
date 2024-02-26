import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tools

def lagrange_interpolation(x, y, x_interp):
    n = len(x)
    result = 0
    for i in range(n):
        term = y[i]
        for j in range(n):
            if j != i:
                term *= (x_interp - x[j]) / (x[i] - x[j])
        result += term
    return result

def newthon_interpolation():
    pass

def calc_diff(y):
    pass

data = tools.read_file()


# lagrange interp.
y = []
x = np.linspace(0, 12, 12)
start = 1889
for index in range(start, start + 12):
    y += [data[index][5]]
print(y)

x_interp = np.arange(0, 12, 0.1)
y_interp = [lagrange_interpolation(x, y, xi) for xi in x_interp]
tools.graph(x, y, x_interp, y_interp)

# newthon interp.
y = y[:6]
x = np.linspace(0, 6, 6)
print(y)

x_interp = np.arange(0, 6, 0.1)
y_interp = [lagrange_interpolation(x, y, xi) for xi in x_interp]
tools.graph(x, y, x_interp, y_interp)