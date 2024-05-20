import numpy as np

def unim(x0, h, f):
    vec_h = np.array([h, h], dtype="float64")
    
    while True:
        if np.all(f(x0 - vec_h) > f(x0)) and np.all(f(x0) < f(x0 + vec_h)):
            return x0 + vec_h, x0 - vec_h
        
        if np.all(f(x0 - vec_h) > f(x0 + vec_h)):
            x0 += vec_h / 2
        else:
            x0 -= vec_h / 2

def quad_interp(x0, h, eps, f):
    a, b = unim(x0, h, f)
    x1 = (a + b) / 2
    xs = [x1, x1 + h]
    if np.all(f(xs[0]) > f(xs[1])):
        xs.append(xs[0] + 2 * h)
    else:
        xs.append(xs[0] - h)
        
    while True:
        f1, f2, f3 = f(xs[0]), f(xs[1]), f(xs[2])
        denom = (xs[1] - xs[2]) * f1 + (xs[2] - xs[0]) * f2 + (xs[0] - xs[1]) * f3
        x_min = min((xs[0], f1), (xs[1], f2), (xs[2], f3), key=lambda p: p[1])[0]
    
        if denom[0] == 0 and denom[1] == 0:
            xs[0] = x_min
            continue
        else:
            x_ = 0.5 * ((xs[1]**2 - xs[2]**2) * f1 + (xs[2]**2 - xs[0]**2) * f2 + (xs[0]**2 - xs[1]**2) * f3) / denom
        
        if abs(f(x_min) - f(x_)) < eps:
            return x_

        if  np.linalg.norm(xs[0]) <= np.linalg.norm(x_) <= np.linalg.norm(xs[2]):
            xs[0] = min((x_min, f(x_min)), (x_, f(x_)), key=lambda p: p[1])[0]
        else:
            xs[0] = x_
            
def hooke_jeeves(f, x0, h, eps):
    def explore_search(f, x, h):
        x_new = x.copy()
        for i in range(len(x)):
            x_pos = x.copy()
            x_pos[i] += h
            x_neg = x.copy()
            x_neg[i] -= h
            if f(*x_pos) < f(*x_new):
                x_new = x_pos
            elif f(*x_neg) < f(*x_new):
                x_new = x_neg
        return x_new
    
    def pattern_search(f, x, pk, alpha, eps):
        x_new = x + alpha * pk
        while f(*x_new) < f(*x):
            x = x_new
            x_new = x + alpha * pk
        return x
    
    x_current = np.array(x0)
    points_sequence = [x_current.tolist()]
    iterations = 0
    
    while True:
        iterations += 1
        x_explored = explore_search(f, x_current, h)
        if np.allclose(x_explored, x_current):
            h /= 2
        else:
            x_pattern = pattern_search(f, x_explored, x_explored - x_current, 0.5, eps)
            points_sequence.append(x_pattern.tolist())
            x_current = x_pattern
            h *= 2
        
        if h < eps:
            break
    
    return points_sequence, iterations

def nelder_mead(f, x0, eps, alpha=1, beta=0.5, gamma=2):
    n = len(x0)
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    for i in range(n):
        x = np.array(x0, dtype=float)
        x[i] = x[i] + 0.1
        simplex[i + 1] = x

    f_values = np.array([f(*x) for x in simplex])

    n_iter = 0
    trace = [simplex[0]]
    while np.max(np.abs(simplex[0] - simplex[1:])) > eps:
        n_iter += 1

        # Step 1: Sort vertices by function values
        idx = np.argsort(f_values)
        simplex = simplex[idx]
        f_values = f_values[idx]

        # Compute the centroid of all points except the farthest one
        x_c = np.mean(simplex[:-1], axis=0)

        # Reflection
        x_r = x_c + alpha * (x_c - simplex[-1])
        f_r = f(*x_r)

        if f_values[0] <= f_r < f_values[-2]:
            simplex[-1], f_values[-1] = x_r, f_r
        elif f_r < f_values[0]:
            # Expansion
            x_e = x_c + gamma * (x_r - x_c)
            f_e = f(*x_e)
            if f_e < f_r:
                simplex[-1], f_values[-1] = x_e, f_e
            else:
                simplex[-1], f_values[-1] = x_r, f_r
        else:
            # Contraction
            x_s = x_c + beta * (simplex[-1] - x_c)
            f_s = f(*x_s)
            if f_s < f_values[-1]:
                simplex[-1], f_values[-1] = x_s, f_s
            else:
                # Reduction of the simplex
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    f_values[i] = f(*simplex[i])

        trace.append(simplex[0])

    return trace, n_iter

def random_search(f, x_start, eps):
    best_solution = None
    best_score = float('inf')
    f_min = -14.4724
    trajectory = [x_start]
    iters = 0
    while abs(f_min - best_score) > eps:
        iters += 1
        solution = np.array([np.random.uniform(-15, 0), np.random.uniform(-15, 0)])
        score = f(*solution)
        if score < best_score:
            best_solution = solution
            best_score = score
            trajectory.append(best_solution)
    return np.array(trajectory), iters


def golden(f, a, b, eps):
    golden_ratio = (1 + np.sqrt(5)) / 2

    x1 = b - (b - a) / golden_ratio
    x2 = a + (b - a) / golden_ratio

    while abs(b - a) > eps:
        if f(x1) < f(x2):
            b = x2
        else:
            a = x1

        x1 = b - (b - a) / golden_ratio
        x2 = a + (b - a) / golden_ratio

    return (a + b) / 2

def powell(f, x0, eps):
    def f_wrapper(x):
        return f(x[0], x[1])

    x = np.array(x0)
    directions = np.identity(len(x))
    iter_count = 0
    points = [x.copy()]

    while iter_count < 1000:
        delta = 0
        for i in range(len(x)):
            direction = directions[i]
            alpha = golden(lambda alpha: f_wrapper(x + alpha * direction), -10, 10, eps)
            x = x + alpha * direction
            delta = max(delta, np.abs(alpha * direction).max())
            points.append(x.copy())

        if delta < eps:
            break

        for i in range(len(x) - 1):
            directions[i] = directions[i + 1]
        directions[-1] = np.random.randn(len(x))

        iter_count += 1

    return points, iter_count