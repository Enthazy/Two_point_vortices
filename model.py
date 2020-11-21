from ulti import *


def euler_method(x1=-2, y1=0.5, x2=2, y2=0.5, s1=1, s2=-1, step=0.0001, epoch=100000):
    x1_history = []
    y1_history = []
    x2_history = []
    y2_history = []

    for i in range(epoch):
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

        if y1 < 0 or y2 < 0:
            print("the simulation is interrupted. Hit the boundary")
            break

        x1_history.append(x1)
        y1_history.append(y1)
        x2_history.append(x2)
        y2_history.append(y2)

        # check_difference(y2-y1, y_r)

        x1 += step * dx1  # * 1/(2*np.pi)
        y1 += step * dy1  # * 1/(2*np.pi)
        x2 += step * dx2  # * 1/(2*np.pi)
        y2 += step * dy2  # * 1/(2*np.pi)

    return x1_history, y1_history, x2_history, y2_history


def euler_method_improved(x1, y1, x2, y2, s1, s2, step, r):
    x1_history = []
    y1_history = []
    x2_history = []
    y2_history = []

    for i in range(r):
        dx1 = -1 * s2 * np.divide((y1 - y2), calculate_distance_sq(x1, y1, x2, y2)) \
              + (-1 * (-1 * s1) * np.divide(1, 2 * y1)) \
              + (-1 * (-1 * s2) * np.divide((y1 - y2), calculate_distance_sq(x1, y1, x2, -1 * y2)))
        dy1 = 1 * s2 * np.divide((x1 - x2), calculate_distance_sq(x1, y1, x2, y2)) \
              + (-1 * (-1 * s2) * np.divide((x1 - x2), calculate_distance_sq(x1, y1, x2, -1 * y2)))
        dx2 = -1 * s1 * np.divide((y2 - y1), calculate_distance_sq(x1, y1, x2, y2)) \
              + (-1 * (-1 * s2) * np.divide(1, 2 * y2)) \
              + (-1 * (-1 * s1) * np.divide((y2 - y1), calculate_distance_sq(x1, -1 * y1, x2, y2)))
        dy2 = 1 * s1 * np.divide((x2 - x1), calculate_distance_sq(x1, y1, x2, y2)) \
              + (-1 * (-1 * s1) * np.divide((x2 - x1), calculate_distance_sq(x1, -1 * y1, x2, y2)))

        if y1 < 0 or y2 < 0: break

        x1_history.append(x1)
        y1_history.append(y1)
        x2_history.append(x2)
        y2_history.append(y2)

        # check_difference(y2 - y1, y_r)

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

        x1 += step * (dx1 + dx1t) / 2
        y1 += step * (dy1 + dy1t) / 2
        x2 += step * (dx2 + dx2t) / 2
        y2 += step * (dy2 + dy2t) / 2

    return x1_history, y1_history, x2_history, y2_history
