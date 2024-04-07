import numpy as np
import pandas as pd
from linalg import *

def gauss(X, fi):
    n = len(X)

    for i in range(n):
        max_index = i
        for j in range(i + 1, n):
            if abs(X[j][i]) > abs(X[max_index][i]):
                max_index = j

        X[i], X[max_index] = X[max_index], X[i]
        fi[i], fi[max_index] = fi[max_index], fi[i]

        for j in range(i + 1, n):
            factor = X[j][i] / X[i][i]
            for k in range(i, n):
                X[j][k] -= factor * X[i][k]
            fi[j] -= factor * fi[i]

    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = fi[i]
        for j in range(i + 1, n):
            x[i] -= X[i][j] * x[j]
        x[i] /= X[i][i]

    return x

eps = 0.0001

def gauss_seidel_ldu(X, fi):
    L, D, U = LDU(X)
    
    x = np.zeros_like(fi)
    for it in range(1, 1000):
        L_D_inv = np.linalg.inv(L + D)
        x_new = np.dot(np.dot(L_D_inv, -1 * U), x) + np.dot(L_D_inv, fi)

        if norm(x_new_i - x_i for x_new_i, x_i in zip(x_new, x)) < eps:
            break
        
        x = np.copy(x_new)
        
    return x, it
    
def gauss_seidel(X, fi):
    n = len(fi)
    x, x_new = np.ones(n), np.ones(n)
      
    for it in range(1, 1000):
        for i in range(n):
            s1 = sum(X[i][j] * x_new[j] for j in range(i))
            s2 = sum(X[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (fi[i] - s1 - s2) / X[i][i]
        
        if norm(x_new_i - x_i for x_new_i, x_i in zip(x_new, x)) < eps:
            break
        
        x = np.copy(x_new)
    
    return x_new, it


coeffs, fi = system()

# eigvals
eig = eigvals(coeffs)
print('l_min =', min(eig))
print('l_max =', max(eig), '\n')

# condition number
print('mu =', norm_m(coeffs) * norm_m(np.linalg.inv(coeffs)), '\n')

# x
methods = ["Gauss".rjust(20), "Gauss-Seidel 1".rjust(20), "Gauss-Seidel 2".rjust(20)]
gz1 = gauss_seidel_ldu(coeffs, fi)
gz2 = gauss_seidel(coeffs, fi)

x1 = gauss(coeffs, fi)
x2 = gz1[0]
x3 = gz2[0]
roots = [x1, x2, x3]

print(pd.DataFrame(np.array(roots).T, columns=methods, index=range(1, 21)))
print(f'Iters:\t\t {gz1[1]} '.rjust(22))
print(f'Iters:\t\t\t\t {gz2[1]}\n'.rjust(24))

# residual vectors
print('r1 =', norm(fi - np.dot(coeffs, x1)))
print('r2 =', norm(fi - np.dot(coeffs, x2)))
print('r3 =', norm(fi - np.dot(coeffs, x3)))