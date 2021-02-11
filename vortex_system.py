from ulti import *


def grad_euler_method(system):
    x1, y1, s1 = system[0]
    x2, y2, s2 = system[1]
    dx1 = -1 * s2 * np.divide((y1 - y2), calculate_distance_sq(x1, y1, x2, y2)) \
          + (-1 * (-1 * s1) * np.divide(1, 2 * y1)) \
          + (-1 * (-1 * s2) * np.divide((y1 + y2), calculate_distance_sq(x1, y1, x2, -1 * y2)))

    dy1 = 1 * s2 * np.divide((x1 - x2), calculate_distance_sq(x1, y1, x2, y2)) \
          + ((-1 * s2) * np.divide((x1 - x2), calculate_distance_sq(x1, y1, x2, -1 * y2)))

    dx2 = -1 * s1 * np.divide((y2 - y1), calculate_distance_sq(x1, y1, x2, y2)) \
          + (-1 * (-1 * s2) * np.divide(1, 2 * y2)) \
          + (-1 * (-1 * s1) * np.divide((y2 + y1), calculate_distance_sq(x1, -1 * y1, x2, y2)))

    dy2 = 1 * s1 * np.divide((x2 - x1), calculate_distance_sq(x1, y1, x2, y2)) \
          + ((-1 * s1) * np.divide((x2 - x1), calculate_distance_sq(x1, -1 * y1, x2, y2)))
    ds1 = 0
    ds2 = 0
    grad = np.array([[dx1, dy1, ds1], [dx2, dy2, ds2]])

    return grad


def grad_euler_method_improved(system, step=0.0001):
    x1, y1, s1 = system[0]
    x2, y2, s2 = system[1]
    dx1 = -1 * s2 * np.divide((y1 - y2), calculate_distance_sq(x1, y1, x2, y2)) \
          + (-1 * (-1 * s1) * np.divide(1, 2 * y1)) \
          + (-1 * (-1 * s2) * np.divide((y1 + y2), calculate_distance_sq(x1, y1, x2, -1 * y2)))

    dy1 = 1 * s2 * np.divide((x1 - x2), calculate_distance_sq(x1, y1, x2, y2)) \
          + ((-1 * s2) * np.divide((x1 - x2), calculate_distance_sq(x1, y1, x2, -1 * y2)))

    dx2 = -1 * s1 * np.divide((y2 - y1), calculate_distance_sq(x1, y1, x2, y2)) \
          + (-1 * (-1 * s2) * np.divide(1, 2 * y2)) \
          + (-1 * (-1 * s1) * np.divide((y2 + y1), calculate_distance_sq(x1, -1 * y1, x2, y2)))

    dy2 = 1 * s1 * np.divide((x2 - x1), calculate_distance_sq(x1, y1, x2, y2)) \
          + ((-1 * s1) * np.divide((x2 - x1), calculate_distance_sq(x1, -1 * y1, x2, y2)))
    ds1 = 0
    ds2 = 0
    grad_p = np.array([[dx1, dy1, ds1], [dx2, dy2, ds2]])

    x1t = x1 + step * dx1
    y1t = y1 + step * dy1
    x2t = x2 + step * dx2
    y2t = y2 + step * dy2

    dx1t = -1 * s2 * np.divide((y1t - y2t), calculate_distance_sq(x1t, y1t, x2t, y2t)) \
           + (-1 * (-1 * s1) * np.divide(1, 2 * y1t)) \
           + (-1 * (-1 * s2) * np.divide((y1t - y2t), calculate_distance_sq(x1t, y1t, x2t, -1 * y2t)))
    dy1t = 1 * s2 * np.divide((x1t - x2t), calculate_distance_sq(x1t, y1t, x2t, y2t)) \
           + (-1 * (-1 * s2) * np.divide((x1t - x2t), calculate_distance_sq(x1t, y1t, x2t, -1 * y2t)))
    dx2t = -1 * s1 * np.divide((y2t - y1t), calculate_distance_sq(x1t, y1t, x2t, y2t)) \
           + (-1 * (-1 * s2) * np.divide(1, 2 * y2t)) \
           + (-1 * (-1 * s1) * np.divide((y2t - y1t), calculate_distance_sq(x1t, -1 * y1t, x2t, y2t)))
    dy2t = 1 * s1 * np.divide((x2t - x1t), calculate_distance_sq(x1t, y1t, x2t, y2t)) \
           + (-1 * (-1 * s1) * np.divide((x2t - x1t), calculate_distance_sq(x1t, -1 * y1t, x2t, y2t)))

    ds1t = 0
    ds2t = 0

    grad_t = np.array([[dx1t, dy1t, ds1t], [dx2t, dy2t, ds2t]])

    grad = 1 / 2 * (grad_p + grad_t)

    return grad


def update(system, grad, step=0.0001):
    system = system + step * grad
    return system


def hit_the_boundary(system):
    y_coord = system[:, 1]
    for y in y_coord:
        if y <= 0: return True
    return False


def run_vortex_system(init_system, step=0.0001, epoch=10000, method='euler'):
    system_history = []
    system = init_system
    for i in range(epoch):
        if method == 'euler_improved':
            grad = grad_euler_method_improved(system, step)
        else:  # euler method
            grad = grad_euler_method(system)
        system = update(system, grad, step)
        if hit_the_boundary(system):
            break
        system_history.append(system)
    system_history = np.array(system_history)
    return np.array(system_history)


def solve_system(x1=-0.5, y1=0.2, x2=0.5, y2=1.2, s1=1, lambd=-1, step=0.0003, epoch=8000, x_r='3',
                 x_l='-3', y_t='3', y_b='0', method='euler'):
    s2 = lambd * s1
    init_system = np.array([[x1, y1, s1], [x2, y2, s2]])
    history_forward = run_vortex_system(init_system, step, epoch, method)
    gen_plot_forward_vec(history_forward)
    history_backward = run_vortex_system(init_system, -1 * step, np.int(epoch / 4), method)
    gen_plot_backward_vec(history_backward)
    bifu_value = calculate_expression(init_system)
    w = calculate_w(init_system)
    show_fig_vec(lambd, w, bifu_value, x_r, x_l, y_t, y_b)
    print("lambda : ", lambd)
    print("Energy : ", calculate_energy_vec(init_system))
    print("P : ", calculate_p(init_system))
    print("W : ", w)
    print("T: ", calculate_t_vec(init_system))
    print("bifurcation value: ", bifu_value)


if __name__ == "__main__":
    solve_system()
