def quad_interp(a, b, eps, f):   
    RATIO = ((5**0.5) - 1) / 2

    while abs(a - b) > eps:
        d = RATIO * (b - a)
        if f(a + d) > f(b - d):
            b = a + d
        else:
            a = b - d

    return (a + b) / 2
