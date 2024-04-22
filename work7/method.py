from matplotlib import pyplot as plt
import numpy as np

def output_method_data(min_value: float, min_point: np.ndarray, iters: int) -> None:
    print(f"Minimum value: {min_value}")
    print(f"Minimum point: {min_point[0]}; {min_point[1]}")
    print(f"Iterations: {iters - 1}")


def plot_relaxation_sequence(X: np.ndarray, Y: np.ndarray , Z: np.ndarray, points_seq: list) -> None:
    relaxation_sequence_x = []
    relaxation_sequence_y = []
    for point in points_seq:
        relaxation_sequence_x.append(point[0])
        relaxation_sequence_y.append(point[1])

    # Setting the values of the target function for drawing level lines
    level_lines_min = np.arange(0, 2, 0.5)
    level_lines_med = np.arange(2, 102, 20)
    level_lines_max = np.arange(103, 1503, 100)
    level_lines = np.concatenate([level_lines_min, level_lines_med, level_lines_max])

    plt.grid()
    plt.rcParams["contour.negative_linestyle"] = "solid"
    plt.contour(X, Y, Z, levels=level_lines, colors="blue", linewidths=1)
    plt.plot(relaxation_sequence_x, relaxation_sequence_y, color="black", linewidth=3)
    plt.plot(relaxation_sequence_x[0], relaxation_sequence_y[0], "ro")
    plt.plot(relaxation_sequence_x[len(relaxation_sequence_x)-1], relaxation_sequence_y[len(relaxation_sequence_y)-1], "ro")
    plt.show()

def quad_interp(a, b, eps, f):   
    RATIO = ((5**0.5) - 1) / 2

    while abs(a - b) > eps:
        d = RATIO * (b - a)
        if f(a + d) > f(b - d):
            b = a + d
        else:
            a = b - d

    return (a + b) / 2
