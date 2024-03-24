import matplotlib.pyplot as plt
import numpy as np

func_x = [0., 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2]
func_y = [1., 1.007568, 1.031121, 1.073456, 1.140228, 1.242129, 1.400176, 1.6603, 2.143460]

n = len(func_y)
# trapeze method
I_tr = 0.15 / 2 * sum([(func_y[i] + func_y[i + 1]) for i in range(n - 1)])
print(I_tr)

# Simpson method
I_Sm = 0.3 / 6 * (func_y[0] + func_y[-1] + sum([4 * func_y[i] for i in range(1, n, 2)]) + sum([2 * func_y[i] for i in range(2, n - 1, 2)]))
print(I_Sm)

# Runge method
func_x1 = [func_x[i] for i in range(0, n, 2)]
func_y1 = [func_y[i] for i in range(0, n, 2)]
# using trapeze method
I = (func_x1[1] - func_x1[0]) / 2 * sum([(func_y1[i] + func_y1[i + 1]) for i in range(len(func_y1) - 1)])
print("eps = ", abs(I_tr - I))
print("delta = ", abs((I_tr - I) / I))