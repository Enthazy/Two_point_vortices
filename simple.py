import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import assert_almost_equal


def cal_energy(x1, y1, x2, y2, s1, s2):
    t1 = np.square(s1) * np.log(2 * y1)
    t2 = np.square(s2) * np.log(2 * y2)
    t3 = np.divide((x1 - x2) ** 2 + (y1 + y2) ** 2, (x1 - x2) ** 2 + (y1 - y2) ** 2)
    t4 = s1 * s2 * np.log(t3)
    return 1 / (2 * np.pi) * (t1 + t2 + t4)


def cal_nu_inverse_sqr(nu):
    return np.divide(1, (nu ** 2))


def distance_sq(x1, y1, x2, y2):
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


def epoch_euler(x1=-2, y1=0.5, x2=2, y2=0.5, s1=1, s2=-1, step=0.0001, round=100000, tt=0):
    x1_history = []
    y1_history = []
    x2_history = []
    y2_history = []

    for i in range(round):
        dx1 = -1 * s2 * np.divide((y1 - y2), distance_sq(x1, y1, x2, y2)) \
              + (-1 * (-1 * s1) * np.divide(1, 2 * y1)) \
              + (-1 * (-1 * s2) * np.divide((y1 + y2), distance_sq(x1, y1, x2, -1 * y2)))

        dy1 = 1 * s2 * np.divide((x1 - x2), distance_sq(x1, y1, x2, y2)) \
              + ((-1 * s2) * np.divide((x1 - x2), distance_sq(x1, y1, x2, -1 * y2)))

        dx2 = -1 * s1 * np.divide((y2 - y1), distance_sq(x1, y1, x2, y2)) \
              + (-1 * (-1 * s2) * np.divide(1, 2 * y2)) \
              + (-1 * (-1 * s1) * np.divide((y2 + y1), distance_sq(x1, -1 * y1, x2, y2)))

        dy2 = 1 * s1 * np.divide((x2 - x1), distance_sq(x1, y1, x2, y2)) \
              + ((-1 * s1) * np.divide((x2 - x1), distance_sq(x1, -1 * y1, x2, y2)))
        if (y1 < 0 or y2 < 0): break

        x1_history.append(x1)
        y1_history.append(y1)
        x2_history.append(x2)
        y2_history.append(y2)

        # check_difference(y2-y1, y_r)

        x1 += step * dx1
        y1 += step * dy1
        x2 += step * dx2
        y2 += step * dy2

    return x1_history, y1_history, x2_history, y2_history

    # print(y2-y1)
    # print(x1_history[-1], y1_history[-1], x2_history[-1], y2_history[-1])
    # plt.plot(x1_history, y1_history, label="left")
    # plt.plot(x2_history, y2_history, label="right")
    # plt.plot(np.arange(-5, 6), np.arange(-5, 6) * 0, label="0")
    # plt.xlim(-10,10)
    # # plt.ylim(-5,5)
    # plt.legend(loc=1)
    # plt.title(str(np.round(tt, 3)))
    # plt.savefig("trail3_"+str(np.round(tt,3))+".png")
    # # plt.show()
    # plt.show(block = False)
    # plt.pause(2)
    # plt.close()


def genplot(x1_history, y1_history, x2_history, y2_history, xl=-6, xr=6, y=5):
    plt.plot(x1_history, y1_history, label="left")
    plt.plot(x2_history, y2_history, label="right")
    plt.plot(np.arange(-5, 6), np.arange(-5, 6) * 0, label="0")
    plt.xlim(xl, xr)
    plt.ylim(0, y)
    plt.legend(loc=1)
    # plt.title(str(np.round(tt, 3)))
    # plt.savefig("trail3_" + str(np.round(tt, 3)) + ".png")
    plt.show()
    # plt.show(block=False)
    # plt.pause(2)
    # plt.close()

def epoch_euler_improved(x1, y1, x2, y2, s1, s2, step, R, tt=0):
    x1_history = []
    y1_history = []
    x2_history = []
    y2_history = []

    for i in range(R):
        dx1 = -1 * s2 * np.divide((y1 - y2), distance_sq(x1, y1, x2, y2)) \
              + (-1 * (-1 * s1) * np.divide(1, 2 * y1)) \
              + (-1 * (-1 * s2) * np.divide((y1 - y2), distance_sq(x1, y1, x2, -1 * y2)))
        dy1 = 1 * s2 * np.divide((x1 - x2), distance_sq(x1, y1, x2, y2)) \
              + (-1 * (-1 * s2) * np.divide((x1 - x2), distance_sq(x1, y1, x2, -1 * y2)))
        dx2 = -1 * s1 * np.divide((y2 - y1), distance_sq(x1, y1, x2, y2)) \
              + (-1 * (-1 * s2) * np.divide(1, 2 * y2)) \
              + (-1 * (-1 * s1) * np.divide((y2 - y1), distance_sq(x1, -1 * y1, x2, y2)))
        dy2 = 1 * s1 * np.divide((x2 - x1), distance_sq(x1, y1, x2, y2)) \
              + (-1 * (-1 * s1) * np.divide((x2 - x1), distance_sq(x1, -1 * y1, x2, y2)))
        if (y1 < 0 or y2 < 0): break

        x1_history.append(x1)
        y1_history.append(y1)
        x2_history.append(x2)
        y2_history.append(y2)

        check_difference(y2 - y1, y_r)
        x1t = x1 + step * dx1
        y1t = y1 + step * dy1
        x2t = x2 + step * dx2
        y2t = y2 + step * dy2
        dx1t = -1 * s2 * np.divide((y1t - y2t), distance_sq(x1t, y1t, x2t, y2t)) \
               + (-1 * (-1 * s1) * np.divide(1, 2 * y1t)) \
               + (-1 * (-1 * s2) * np.divide((y1t - y2t), distance_sq(x1t, y1t, x2t, -1 * y2t)))
        dy1t = 1 * s2 * np.divide((x1t - x2t), distance_sq(x1t, y1t, x2t, y2t)) \
               + (-1 * (-1 * s2) * np.divide((x1t - x2t), distance_sq(x1t, y1t, x2t, -1 * y2t)))
        dx2t = -1 * s1 * np.divide((y2t - y1t), distance_sq(x1t, y1t, x2t, y2t)) \
               + (-1 * (-1 * s2) * np.divide(1, 2 * y2t)) \
               + (-1 * (-1 * s1) * np.divide((y2t - y1t), distance_sq(x1t, -1 * y1t, x2t, y2t)))
        dy2t = 1 * s1 * np.divide((x2t - x1t), distance_sq(x1t, y1t, x2t, y2t)) \
               + (-1 * (-1 * s1) * np.divide((x2t - x1t), distance_sq(x1t, -1 * y1t, x2t, y2t)))

        x1 += step * (dx1 + dx1t) / 2
        y1 += step * (dy1 + dy1t) / 2
        x2 += step * (dx2 + dx2t) / 2
        y2 += step * (dy2 + dy2t) / 2

    # print(y2-y1)
    # print(x1_history[-1], y1_history[-1], x2_history[-1], y2_history[-1])
    plt.plot(x1_history, y1_history, label="left")
    plt.plot(x2_history, y2_history, label="right")
    plt.plot(np.arange(-5, 6), np.arange(-5, 6) * 0, label="0")
    plt.xlim(-5,5)
    # plt.ylim(-5,5)
    plt.legend(loc=1)
    plt.title(str(np.round(tt, 1)))
    # plt.savefig("trail2_"+str(np.round(tt,1))+".png")
    plt.show()
    # plt.show(block = False)
    # plt.pause(2)
    # plt.close()


def check_difference(x1, x2):
    assert_almost_equal(x1, x2, decimal=5)


# if __name__ == "__main__":
#     # for tt in np.arange(0.25,0.34,0.01):
#
#     x1 = -4
#     y1 = 2
#     s1 = 1
#
#     x2 = 4
#     y2 = 3
#     s2 = -1
#     xxcc = cal_energy(x1, y1, x2, y2, s1, s2)
#     # print(xxcc)
#     print(np.exp(-1 * 2 * np.pi * xxcc))
#
#     y_r = y2 - y1
#     print(cal_nu_inverse_sqr(y_r))
#
#     step = 0.0003
#     R = 100000
#
#     a,b,c,d =epoch_euler(x1, y1, x2, y2, s1, s2, step, R, 0)
#     genplot(a,b,c,d)
#     # epoch_euler_improved(x1, y1, x2, y2, s1, s2, step, R)
