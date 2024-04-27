import minimization as mn
import numpy as np
import method as m
import matplotlib.pyplot as plt

# function
def psi(x0):
    x = current_point[0] + p[0] * abs(x0)
    y = current_point[1] + p[1] * abs(x0)
    return func(x, y)

def func(x, y):
    return x**4 + 20 * x**3 + 2 * x**2 * y**2 + 36 * x**2 * y + 312 * x**2 + 20 * x * y**2 + 360 * x * y + 2121 * x + y**4 + 36 * y**3 + 537 * y**2 + 3834 * y + 11308

def f(vec):
    return func(*vec)

def grad(x, y):
    dudx = 4 * x**3 + 60 * x**2 + 4 * x * y**2 + 72 * x * y + 624 * x + 20 * y**2 + 360 * y + 2121
    dudy = 4 * x**2 * y + 36 * x**2 + 40 * x * y + 360 * x + 4 * y**3 + 108 * y**2 + 1074 * y + 3834
    return np.array([dudx, dudy], dtype="float64")


eps = 0.0001
start = np.array([6, 7], dtype="float64")

points_seq = [start]
function_vals = [func(*start)]

iters = 1
gradient = grad(*start)
p = -gradient
while np.linalg.norm(gradient) > eps:
    current_point = points_seq[iters-1]
    
    alpha = m.quad_interp(0., 0.02, eps, psi)

    new_point = np.array([current_point[0] + p[0] * alpha, current_point[1] + p[1] * alpha], dtype="float64")

    beta = np.linalg.norm(grad(*new_point))**2 / np.linalg.norm(gradient)**2
    gradient = grad(*new_point)
    p = beta * p - gradient
    
    iters += 1
    points_seq.append(new_point)
    function_vals.append(func(*new_point))


# Output
points_seq = np.array(points_seq)
print("Func val:", func(*points_seq[-1]))
print("Min point:", points_seq[-1][0], points_seq[-1][1])
print("Iters:", iters)

x_axis = np.arange(-50, 50, 0.1)
y_axis = np.arange(-60, 60, 0.1)
X, Y = np.meshgrid(x_axis, y_axis)
Z = func(X, Y)

plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=30, cmap="plasma")
plt.plot(points_seq[:, 0], points_seq[:, 1], '.-', color='r', label="Conj. gradients")

# Hook-Jeevs
seq, iters = mn.hooke_jeeves(func, start, 0.1, eps)
seq = np.array(seq)
print("\nFunc val:", func(*seq[-1]))
print("Min point:", seq[-1][0], seq[-1][1])
print("Iters:", iters)
plt.plot(seq[:, 0], seq[:, 1], '.-', color='g', label="Hook-Jeevs")

# Nelder-Mead
seq, iters = mn.nelder_mead(func, start, eps)
seq = np.array(seq)
print("\nFunc val:", func(*seq[-1]))
print("Min point:", seq[-1][0], seq[-1][1])
print("Iters:", iters)
plt.plot(seq[:, 0], seq[:, 1], '.-', color='b', label="Nelder-Mead")

# Powell
seq, iters = mn.powell_method(func, start, eps)
seq = np.array(seq)
print("\nFunc val:", func(*seq[-1]))
print("Min point:", seq[-1][0], seq[-1][1])
print("Iters:", iters)
plt.plot(seq[:, 0], seq[:, 1], '.-', color='m', label="Powel")

# Random search
seq, iters = mn.random_search(func, start, 0.001)
seq = np.array(seq)
print("\nFunc val:", func(*seq[-1]))
print("Min point:", seq[-1][0], seq[-1][1])
print("Iters:", iters)
plt.plot(seq[:, 0], seq[:, 1], '.-', color='black', label="Random search")

plt.grid(True)
plt.colorbar()
plt.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='plasma')
plt.show()