import numpy as np
import matplotlib.pyplot as plt

def Func(X):
    return np.array([
        1 / (X[0]**1.5 + 1) - X[1],
        X[0]**2 + X[1]**2 - 9
    ])

def Jacobian(X):
    return np.array([
        [- (3/2) * (X[0]**(1/2)) / ((X[0]**(3/2) + 1)**2), -1],
        [2 * X[0], 2 * X[1]]
    ])

eps = 0.0001

def newthon_method(F, X, cntr):
    J = Jacobian(X)
    Xn = X - np.linalg.inv(J) @ F(X)
    Fxn = F(Xn)

    if np.sqrt(Fxn[0]**2 + Fxn[1]**2) < eps:
        return X[0], X[1], Fxn, cntr
    else:
        cntr += 1
        return newthon_method(F, Xn, cntr)
    

x, y, func, cntr = newthon_method(Func, np.array([1, 1]), 1)
print("x1 =", x)
print("x2 =", y)
print("\nf1 =", func[0])
print("f2 =", func[1])
print("\ncount =", cntr)