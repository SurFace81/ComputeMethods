import numpy as np
import matplotlib.pyplot as plt

def func1(x, y):
    # return 1 / (x**1.5 + 1) - y
    return x - y - 1

def func2(x, y):
    # return x**2 + y **2 - 9
    return x + 3 * y - 9

def Func(x, y):
    return np.array([func1(x, y),
                    func2(x, y)])

def Jacobian(x, y):
    # J = np.array([[(-3 * np.sqrt(x) / (2 * x**3 + 4 * x**1.5 + 2)), 2 * x],
    #               [-1, 2 * y]])
    # return np.linalg.inv(J)
    J = np.array([[1, 1],
                  [-1, 3]])
    return J

eps = 0.0001

def newthon_method(F, X):
    Xn = X - np.linalg.inv(Jacobian(X[0], X[1])) @ F(X[0], X[1])
    Fxn = F(Xn[0], Xn[1])

    if np.sqrt(Fxn[0]**2 + Fxn[1]**2) < eps:
        return X[0], X[1], J
    else:
        return newthon_method(F, Xn)
    

x, y, j = newthon_method(Func, np.array([1, 0]))
print("x1 =", x)
print("x2 =", y)
print("\nf1 =", func1(x, y))
print("f2 =", func2(x, y))
print("\nJ =", j)