import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return 2 * np.exp(x) + 5 * x - 6

def func_deriv(x):
    return 2 * np.exp(x) + 5

def iter_func(x):
    return (6 - 2 * np.exp(x)) / 5


eps = 0.0001

def simple_iterations(f, x, n):
    xn = f(x)
    if abs((xn - x) / xn) < eps:
        return xn, n + 1
    else:
        n += 1
        return simple_iterations(f, xn, n)
    
def newthon_method(f, fd, x, n):
    xn = x - f(x) / fd(x)
    if abs(xn - x) < eps:
        return xn, n + 1
    else:
        n += 1
        return newthon_method(f, fd, xn, n)
    pass


# simple iterations
x1, count = simple_iterations(iter_func, 0.5, 0)     # 0.524 - root
print(count)
print("x1 =", x1)
print("fx1 =", func(x1))

# newthon
x1, count = newthon_method(func, func_deriv, 0.5, 0)
print("\n", count, sep="")
print("x1 =", x1)
print("fx1 =", func(x1))

x = np.arange(-2, 2, 0.001)
y = func(x)
plt.plot(x, y)
plt.grid(True)
plt.title('F(x)')
plt.show()