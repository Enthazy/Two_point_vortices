import numpy as np
from numpy.testing import assert_almost_equal
import matplotlib.pyplot as plt


def calculate_energy(x1, y1, x2, y2, s1, s2):
    t1 = np.square(s1) * np.log(2 * y1)
    t2 = np.square(s2) * np.log(2 * y2)
    t3 = np.divide((x1 - x2) ** 2 + (y1 + y2) ** 2, (x1 - x2) ** 2 + (y1 - y2) ** 2)
    t4 = s1 * s2 * np.log(t3)
    return 1 / (4 * np.pi) * (t1 + t2 + t4)


def calculate_c(x1, y1, x2, y2, s1, s2):
    e = calculate_energy(x1, y1, x2, y2, s1, s2)
    return np.exp(np.divide(-1 * 4 * np.pi * e, s1**2))


def calculate_nu(x1, y1, x2, y2, s1, s2):
    lambd = np.divide(s2, s1)
    return y1 + lambd * y2


def calculate_t(x1, y1, x2, y2, s1, s2):
    c = calculate_c(x1, y1, x2, y2, s1, s2)
    v = calculate_nu(x1, y1, x2, y2, s1, s2)
    lambd = np.divide(s2, s1)
    return c * np.power(v, (1 + np.square(lambd)))


def calculate_distance_sq(x1, y1, x2, y2):
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


def show_message(x1, y1, x2, y2, s1, s2):
    print("energy: ", calculate_energy(x1, y1, x2, y2, s1, s2))
    print("Nu: ", calculate_nu(x1, y1, x2, y2, s1, s2))
    print("T: ", calculate_t(x1, y1, x2, y2, s1, s2))



def check_difference(x1, x2):
    assert_almost_equal(x1, x2, decimal=5)


def gen_plot(x1_history, y1_history, x2_history, y2_history, xl=-6, xr=6, y=5):
    plt.plot(x1_history, y1_history, label="left")
    plt.plot(x2_history, y2_history, label="right")
    plt.plot(np.arange(-5, 6), np.arange(-5, 6) * 0, label="0")
    # plt.xlim(xl, xr)
    # plt.ylim(0, y)
    plt.legend(loc=1)
    # plt.title(str(np.round(tt, 3)))
    # plt.savefig("trail3_" + str(np.round(tt, 3)) + ".png")
    plt.show()
    # plt.show(block=False)
    # plt.pause(2)
    # plt.close()


def save_fig(x1_history, y1_history, x2_history, y2_history, name, xl=-6, xr=6, y=5):
    plt.plot(x1_history, y1_history, label="left")
    plt.plot(x2_history, y2_history, label="right")
    plt.plot(np.arange(-5, 6), np.arange(-5, 6) * 0, label="0")
    # plt.xlim(xl, xr)
    # plt.ylim(0, y)
    plt.legend(loc=1)
    plt.title(name)
    plt.savefig(name + ".png")
    plt.show()


def save_data():
    return