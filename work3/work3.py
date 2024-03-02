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

def newton_interpolation(x, y, x_new):
    n = len(y)
    F = divided_diff(x, y)[0]
    
    interpolated_value = F[0]
    for i in range(1, n):
        term = F[i]
        for j in range(i):
            term *= (x_new - x[j])
        interpolated_value += term
    return interpolated_value

def newton_interpolation2(x, y, x_new):
    n = len(y)
    F = divided_diff(x, y)
    
    interpolated_value = F[n - 1][0]
    for i in range(1, n):
        term = F[n - 1 - i][i]
        for j in range(i):
            term *= (x_new - x[n - 1 - j])
        interpolated_value += term
    return interpolated_value

# def poly_approx(x, y):
#     n = len(x)
#     X = np.array([[x[i]**j for j in range(6)] for i in range(n)])
#     coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
#     return coefficients

# def poly_val(coefficients, x):
#     n = len(coefficients)
#     y = sum(coefficients[i] * x**i for i in range(n))
#     return y


def divided_diff(x, y):
    n = len(y)
    F = [[None] * n for _ in range(n)]
    for i in range(n):
        F[i][0] = y[i]
    for j in range(1, n):
        for i in range(n - j):
            F[i][j] = (F[i + 1][j - 1] - F[i][j - 1]) / (x[i + j] - x[i])
    return F


data = tools.read_file()


# lagrange interp.
start = 1889
y = [data[i][5] for i in range(start, start + 12)]
x = np.linspace(0, 12, 12)
print(y)

x_interp = np.arange(min(x), max(x) + 0.1, 0.1)
y_interp = [lagrange_interpolation(x, y, xi) for xi in x_interp]
tools.graph(x, y, x_interp, y_interp)


# newthon interp. ver. 1
y1 = y[:6]
x1 = np.linspace(0, 6, 6)
print(y1)

x_interp = np.arange(min(x1), max(x1) + 0.1, 0.1)
y_interp = [newton_interpolation(x1, y1, xi) for xi in x_interp]
tools.graph(x1, y1, x_interp, y_interp)


# newthon interp. ver. 2
y2 = y[6:]
x2 = np.linspace(0, 6, 6)
print(y2)

x_interp = np.arange(min(x2), max(x2) + 0.1, 0.1)
y_interp = [newton_interpolation2(x2, y2, xi) for xi in x_interp]
tools.graph(x2, y2, x_interp, y_interp)

# approx. with 5th degree polynome
#
# y_all = [data[1875 + i][5] for i in range(len(data)) if data[1875 + i][5] != 999.9]
# n = len(y_all)
# x3 = np.linspace(0, n, n)
# coefficients = poly_approx(x3, y_all)

# x_interp = np.arange(min(x3), max(x3) + 0.1, 0.1)
# y_interp = [poly_val(coefficients, xi) for xi in x_interp]
# tools.graph(x3,  y_all, x_interp, y_interp)