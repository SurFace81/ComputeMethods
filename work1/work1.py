import matplotlib.pyplot as plt
import numpy as np

def func(x):
    return (x - 1) * (x - 2)**2 * (x - 3)**3

def derivative_a(x):
    return 6 * x**5 - 70 * x**4 + 320 * x**3 - 714 * x**2 + 774 * x - 324

def deriv_2nd(x):
    return 30 * x**4 - 280 * x**3 + 960 * x**2 - 1428 * x + 774

def derivative_n(x, h):
    return (func(x + h) - func(x - h)) / (2 * h)


eps = 1/2;
while 1 + eps != 1:
    eps /= 2  
print("eps =", eps)

x = np.arange(0.3, 3, 0.01)
derivs = deriv_2nd(x)
functs = func(x)
M0 = max(functs)
M2 = max(derivs)
h = 2 * np.sqrt(abs(M0 * eps / M2))
print("h =", h)
err = 2 * np.sqrt(abs(M0 * M2 * eps))
print("err =", err)
print("\nM0 =", M0)
print("M2 =", M2)


fig, ax = plt.subplots(nrows=2, ncols=2)
fig.subplots_adjust(hspace=0.4)

y = [func(_) for _ in x]
ax[0, 0].plot(x, y)
ax[0, 0].grid(True)
ax[0, 0].set_title("F(x)")

y = [derivative_a(_) for _ in x]
ax[0, 1].plot(x, y)
ax[0, 1].grid(True)
ax[0, 1].set_title("F'ₐ(x)")

y1 = [derivative_n(_, h) for _ in x]
ax[1, 0].plot(x, y1)
ax[1, 0].grid(True)
ax[1, 0].set_title("F'ₙ(x)")

y = [abs(y1[i] - y[i]) for i in range(len(y))]
ax[1, 1].plot(x, y)
ax[1, 1].grid(True)
ax[1, 1].set_title("|F'ₙ(x) - F'ₐ(x)|")

plt.show()