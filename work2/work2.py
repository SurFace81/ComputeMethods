import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return 3 * x**4 - 4 * x**3 - 12 * x**2 + 1

eps = 0.0001
        
def dichotomy(f, cntr, a, b):
    c = (a + b) / 2
    if f(a) * f(c) < 0:
        b = c
    elif f(c) * f(b) < 0:
        a = c
    
    if abs(b - a) < eps:
        return c, f(c), cntr
    else:
        cntr += 1
        return dichotomy(f, cntr, a, b)  
    
def chords(f, cntr, a, b):
    c = a - f(a) * (b - a) / (f(b) - f(a))

    if f(a) * f(c) < 0:
        b = c
    elif f(c) * f(b) < 0:
        a = c
        
    if abs(b - a) < eps:
        return c, f(c), cntr
    else:
        cntr += 1
        return chords(f, cntr, a, b)
        

# dichotomy method
x1, fx1, count = dichotomy(func, 1, -1.5, 1.5)
print("x1 =", x1)
print("fx1 =", fx1)
print("cnt =", count)

# chord method
x1, fx1, count = chords(func, 1, -1.5, 1.5)
print("\nx1 =", x1)
print("fx1 =", fx1)
print("cnt =", count)