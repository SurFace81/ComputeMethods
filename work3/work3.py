import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tools

# =========
# Functions
# =========
def lagrange_interp(x, y, x_interp):
    n = len(x)
    result = 0
    for i in range(n):
        term = y[i]
        for j in range(n):
            if j != i:
                term *= (x_interp - x[j]) / (x[i] - x[j])
        result += term
    return result

def newton_interp(x, y, x_new):
    n = len(y)
    F = divided_diff(x, y)[0]
    
    interpolated_value = F[0]
    for i in range(1, n):
        term = F[i]
        for j in range(i):
            term *= (x_new - x[j])
        interpolated_value += term
    return interpolated_value

def newton_interp2(x, y, x_new):
    n = len(y)
    F = divided_diff(x, y)
    
    interpolated_value = F[n - 1][0]
    for i in range(1, n):
        term = F[n - 1 - i][i]
        for j in range(i):
            term *= (x_new - x[n - 1 - j])
        interpolated_value += term
    return interpolated_value

def divided_diff(x, y):
    n = len(y)
    F = [[None] * n for _ in range(n)]
    for i in range(n):
        F[i][0] = y[i]
    for j in range(1, n):
        for i in range(n - j):
            F[i][j] = (F[i + 1][j - 1] - F[i][j - 1]) / (x[i + j] - x[i])
    return F


def poly_approx(x, y, degree):
    X = np.column_stack([x**i for i in range(degree + 1)])
    coeffs = np.linalg.solve(X.T @ X, X.T @ y)
    return coeffs

def poly_val(coeffs, x):
    n = len(coeffs)
    y = sum(coeffs[i] * x**i for i in range(n))
    return y


def trig_interp(x, y, func):
    n = len(x)
    X = np.column_stack([func(k * x) for k in range(n)])
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    return coeffs

def trig_val(coeffs, xi, func):
    n = len(coeffs)
    y = sum(coeffs[k] * func(k * xi) for k in range(n))
    return y


def orth_exp_coeffs(x_values, y_values):
    n = len(x_values)
    V = np.zeros((n, n), dtype=np.complex_)
    for i in range(n):
        for j in range(n):
            V[i, j] = np.exp(1j * i * x_values[j])
    return np.linalg.solve(V, y_values)

def orth_exp_approx(x_interp, coeffs):
    n = len(coeffs)
    y_interp = np.zeros_like(x_interp, dtype=np.complex_)
    for i in range(len(x_interp)):
        for j in range(n):
            y_interp[i] += coeffs[j] * np.exp(1j * j * x_interp[i])
    return y_interp


def cubic_spline(x, y):
    n = len(x)
    h = np.diff(x)  # Compute the differences between consecutive x values
    alpha = np.zeros(n)
    for i in range(1, n - 1):
        alpha[i] = 3 / h[i] * (y[i + 1] - y[i]) - 3 / h[i - 1] * (y[i] - y[i - 1])

    # Initialize arrays for tridiagonal matrix algorithm
    l = np.zeros(n)
    mu = np.zeros(n)
    z = np.zeros(n)
    l[0] = 1

    # Forward pass
    for i in range(1, n - 1):
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    l[n - 1] = 1

    c = np.zeros(n)
    b = np.zeros(n)
    d = np.zeros(n)

    # Backward pass to find coefficients c
    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    return b, c, d

def evaluate_spline(x, y, b, c, d, x_new):
    spline_values = []
    for x_val in x_new:
        for i in range(len(x) - 1):
            if x[i] <= x_val <= x[i + 1]:
                dx = x_val - x[i]
                spline_val = y[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3
                spline_values.append(spline_val)
                break
    return np.array(spline_values)


# ===========================
# Examples of Using Functions
# ===========================
data = tools.read_file()

# # lagrange interp.
# start = 1889
# y = [data[i][5] for i in range(start, start + 12)]
# x = np.linspace(0, 12, 12)
# x_interp = np.arange(min(x), max(x) + 0.1, 0.1)
# y_interp = [lagrange_interp(x, y, xi) for xi in x_interp]
# tools.plot(x, y, x_interp, y_interp, "(Lagrange) y: [" + '; '.join(str(_) for _ in y) + "]")


# # newthon interp. ver. 1
# y1 = y[:6]
# x1 = np.linspace(0, 6, 6)
# x_interp = np.arange(min(x1), max(x1) + 0.1, 0.1)
# y_interp = [newton_interp(x1, y1, xi) for xi in x_interp]
# tools.plot(x1, y1, x_interp, y_interp, "(Newthon ver. 1) y: [" + '; '.join(str(_) for _ in y1) + "]")


# # newthon interp. ver. 2
# y2 = y[6:]
# x2 = np.linspace(0, 6, 6)
# x_interp = np.arange(min(x2), max(x2) + 0.1, 0.1)
# y_interp = [newton_interp2(x2, y2, xi) for xi in x_interp]
# tools.plot(x2, y2, x_interp, y_interp, "(Newthon ver. 2) y: [" + '; '.join(str(_) for _ in y2) + "]")


# # approx. with 5th degree polynome
# y_all = [data[1875 + i][5] for i in range(len(data)) if data[1875 + i][5] != 999.9]
# n = len(y_all)
# x3 = np.linspace(0, n, n)

# fig, axs = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True)
# for d, ax in zip(range(1, 3), axs.flat):
#     coefficients = poly_approx(x3, y_all, d)
#     x_interp = np.arange(min(x3), max(x3) + 0.1, 0.1)
#     y_interp = [poly_val(coefficients, xi) for xi in x_interp]
#     ax.plot(x3, y_all, 'o')
#     ax.plot(x_interp, y_interp, 'red')
#     ax.set_title('Degree {}'.format(d))
# plt.show()

# plt.plot(x3, y_all, "o")
# plt.plot(x_interp, y_interp, "red")
# plt.title("Approximation ({} degree)".format(3))
# plt.show()


# # cubic spline interp.
# y3 = y
# x3 = np.linspace(0, 12, 12)
# b, c, d = cubic_spline(x3, y3)
# x_interp = np.arange(min(x3), max(x3) + 0.1, 0.1)
# y_interp = evaluate_spline(x3, y3, b, c, d, x_interp)
# tools.plot(x3, y3, x_interp, y_interp, "(Cubic spline) y: [" + '; '.join(str(_) for _ in y3) + "]")


# # trig. interp.
# y_all = [data[1875 + i][5] for i in range(len(data)) if data[1875 + i][5] != 999.9]
# n = len(y_all)
# x4 = np.linspace(0, n, n)

# x_interp = np.arange(min(x4), max(x4) + 0.1, 0.1)
# coeffs = trig_interp(x4, y_all, np.sin)
# y_interp = [trig_val(coeffs, xi, np.sin) for xi in x_interp]
# tools.plot(x4, y_all, x_interp, y_interp, "(Trig. interp.)")


# orth. approx.
y_all = [data[1875 + i][5] for i in range(len(data)) if data[1875 + i][5] != 999.9]
n = len(y_all)
x5 = np.linspace(0, n, n)

y = [y_all[i] for i in range(0, len(y_all), 20)]
x6 = [x5[i] for i in range(0, len(x5), 20)]

coeffs = orth_exp_coeffs(x6, y)
x_interp = np.arange(min(x6), max(x6) + 0.1, 0.1)
y_interp = orth_exp_approx(x_interp, coeffs)
tools.plot(x5, y_all, x_interp, y_interp, "(Orth. approx.)")