import numpy as np
from numpy import sin, cos, exp
from matplotlib import pyplot as plt
from numpy.linalg import norm
from scipy.integrate import odeint

x = [0, 2]
ypq = [1, 0, 0]

def func(args, x):
    y, p, q = args
    return [p, q, 4 * y + sin(x) * exp(-x)]

def ode():
    sol = odeint(func, ypq, X)    
    return sol[:,0], sol[:,1], sol[:,2]

def Teilor(cntr, xrange, init):
    xmin, xmax = xrange
    y, p, q = init
    
    h = (xmax - xmin) / (2 ** cntr)
    x = xmin
    N = 0
    
    X = np.array([x])
    Y = np.array([y])
    P = np.array([p])
    Q = np.array([q])
    
    while x < xmax:
        x += h
        N += 1
        
        y2 = y + h * p + \
             h ** 2 / 2 * q + \
             h ** 3 / 6 * (4 * y + exp(-x) * sin(x)) + \
             h ** 4 / 24 * (4 * p - exp(-x) * sin(x) + exp(-x) * cos(x))
        
        p2 = p + h * q + \
             h ** 2 / 2 * (4 * y + exp(-x) * sin(x)) + \
             h ** 3 / 6 * (4 * p - exp(-x) * sin(x) + exp(-x) * cos(x)) + \
             h ** 4 / 24 * (4 * q - 2 * exp(-x) * cos(x))
        
        q2 = q + h * (4 * y + exp(-x) * sin(x)) + \
             h ** 2 / 2 * (4 * p - exp(-x) * sin(x) + exp(-x) * cos(x)) + \
             h ** 3 / 6 * (4 * q - 2 * exp(-x) * cos(x)) + \
             h ** 4 / 24 * (16 * y + 6 * exp(-x) * sin(x) + 2 * exp(-x) * cos(x))
        
        y, p, q = y2, p2, q2
        X = np.append(X, x)
        Y = np.append(Y, y)
        P = np.append(P, p)
        Q = np.append(Q, q)
        
    return X, Y, P, Q, h

eps = 0.0001
N = 0

while True:
    T = Teilor(N, x, ypq)
    T2 = Teilor(N + 1, x, ypq)

    N += 1
    
    if abs(T2[1][1] - T[1][1]) < eps:
        break


print("Iters: ", N)
X, Y, P, Q, h = T2
YY, PP, QQ = ode()
print("h: ", h)

# Solution graphs
plt.plot(X, Y, '.-', color='black')
plt.plot(X, YY, '.', color='r')
plt.xlabel('x')
plt.ylabel('y')
plt.title('y(x)')
plt.grid(True)
plt.show()

plt.plot(X, P, '.-', color='black')
plt.plot(X, PP, '.', color='r')
plt.xlabel('x')
plt.ylabel("y'")
plt.title("y'(x)")
plt.grid(True)
plt.show()

# Trajectories graphs
plt.plot(Y, P, '.-', color='black')
plt.plot(Y, PP, '.', color='r')
plt.xlabel('y')
plt.ylabel("y'")
plt.title("y'(y)")
plt.grid(True)
plt.show()

# Difference
plt.plot(X, abs(YY - Y), '.-', color='black')
plt.xlabel('x')
plt.ylabel("y")
plt.title("Difference")
plt.grid(True)
plt.show()

plt.plot(X, abs(PP - P), '.-', color='black')
plt.xlabel('x')
plt.ylabel("y'")
plt.title("Difference")
plt.grid(True)
plt.show()