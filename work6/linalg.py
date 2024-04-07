import numpy as np

def system():
    n = 20
    coeffs = []
    vals = [0.1, 1, 10, 1, 0.1]
    
    for i in range(n):
        t = i - 2
        x_vals = np.zeros(n)
        for j in range(5):
            if t >= 0 and t < 20:
                x_vals[t] = vals[j]
            t += 1
        coeffs += [x_vals]
    
    return coeffs, [i for i in range(1, n + 1)]   # coeffs, fi


def norm(x):
    return sum(x_i**2 for x_i in x)**0.5

def norm_m(coeffs):
    n = len(coeffs)
    return sum([coeffs[i][j]**2 for i in range(n) for j in range(n)])**0.5


def LDU(coeffs):
    n = len(coeffs)
    L, D, U = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i < j:
                U[i][j] = coeffs[i][j]
            elif i > j:
                L[i][j] = coeffs[i][j]
            else:
                D[i][j] = coeffs[i][j]

    return L, D, U


def eigvals(A):  
    def iters(coeffs, n):
        b = np.ones(n)
        for _ in range(1000):
            b_new = np.dot(A, b)
            b = b_new / norm(b_new)
    
        eigenvalue = np.dot(np.dot(b.T, A), b)
        return eigenvalue, b
    
    n = len(A)
    eigvals = []
    eigvecs = []
    for _ in range(n):
        eigval, eigvec = iters(A, n)
        eigvals += [eigval]
        eigvecs += [eigvec]
        
        A -= eigval * np.outer(eigvec, eigvec)
        
    return eigvals