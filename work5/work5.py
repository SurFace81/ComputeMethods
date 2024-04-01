import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

func_x = [0., 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2]
func_y = [1., 1.007568, 1.031121, 1.073456, 1.140228, 1.242129, 1.400176, 1.6603, 2.143460]

x_interp = np.linspace(min(func_x), max(func_x), 1000)
y_interp = interp1d(func_x, func_y, kind='cubic')(x_interp)


def trapeze(x, y):
    return (x[1] - x[0]) / 2 * sum([(y[i] + y[i + 1]) for i in range(len(y) - 1)])

def simpson(x, y):
    return (x[1] - x[0]) / 3 * sum((y[i] + 4 * y[i + 1] + y[i + 2]) for i in range(0, len(y) - 2, 2))

def runge(x, y):
    x1 = [x[i] for i in range(0, len(x), 2)]
    y1 = [y[i] for i in range(0, len(y), 2)]

    # using trapeze method
    eps = abs(trapeze(x, y) - trapeze(x1, y1))
    delta = abs((trapeze(x, y) - trapeze(x1, y1)) / trapeze(x1, y1))
    return eps, delta


# trapeze method
print(trapeze(x_interp, y_interp))

# Simpson method
print(simpson(x_interp, y_interp))

# Runge method
eps, delta = runge(x_interp, y_interp)
print("eps =", eps)
print("delta =", delta)

# graph
plt.plot(x_interp, y_interp, color="red")
plt.plot(func_x, func_y, "o", markersize=5)
plt.grid(True)
plt.show()