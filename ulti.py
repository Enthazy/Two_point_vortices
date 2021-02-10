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

def calculate_lambd(x1, y1, x2, y2, s1, s2):
    return np.divide(s2, s1)


def calculate_nu(x1, y1, x2, y2, s1, s2):
    lambd = np.divide(s2, s1)
    return y1 + lambd * y2


def calculate_t(x1, y1, x2, y2, s1, s2):
    c = calculate_c(x1, y1, x2, y2, s1, s2)
    v = calculate_nu(x1, y1, x2, y2, s1, s2)
    lambd = np.divide(s2, s1)
    return c * v * np.power(np.abs(v), (np.square(lambd)))


def calculate_distance_sq(x1, y1, x2, y2):
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


def show_message(x1, y1, x2, y2, s1, s2):
    print("energy: ", calculate_energy(x1, y1, x2, y2, s1, s2))
    print("C: ", calculate_c(x1, y1, x2, y2, s1, s2))
    print("Nu: ", calculate_nu(x1, y1, x2, y2, s1, s2))
    print("T: ", calculate_t(x1, y1, x2, y2, s1, s2))



def check_difference(x1, x2):
    assert_almost_equal(x1, x2, decimal=5)


def gen_plot(x1_history, y1_history, x2_history, y2_history, xl=-6, xr=6, y=5):
    plt.plot(x1_history, y1_history, label="left")
    plt.plot(x2_history, y2_history, label="right")
    # plt.plot(np.arange(-5, 6), np.arange(-5, 6) * 0, label="0")
    # plt.xlim(xl, xr)
    plt.ylim(bottom=0)
    plt.legend(loc=1)
    # plt.title(str(np.round(tt, 3)))
    # plt.savefig("trail3_" + str(np.round(tt, 3)) + ".png")
    plt.show()
    # plt.show(block=False)
    # plt.pause(2)
    # plt.close()


def save_fig_old(x1_history, y1_history, x2_history, y2_history, name, xl=-6, xr=6, y=5):
    plt.plot(x1_history, y1_history, label="left")
    plt.plot(x2_history, y2_history, label="right")
    plt.plot(np.arange(-3, 6), np.arange(-3, 6) * 0, label="0")
    # plt.xlim(xl, xr)
    # plt.ylim(0, y)
    plt.legend(loc=1)
    plt.title(name)
    plt.savefig(name + ".png")
    # plt.show()
    plt.show(block=False)
    plt.pause(2)
    plt.close()

def gen_plot_forward(x1_history, y1_history, x2_history, y2_history, xl=-6, xr=6, y=5):
    plt.figure(figsize=(8, 5))
    plt.plot(x1_history, y1_history, c='coral', label="vortex 1, forward")
    plt.plot(x2_history, y2_history, c='teal', label="vortex 2, forward")
    # plt.plot(np.arange(-5, 6), np.arange(-5, 6) * 0, label="0")
    plt.legend(loc=1)


def gen_plot_backward(x1_history, y1_history, x2_history, y2_history, xl=-6, xr=6, y=5):
    plt.plot(x1_history, y1_history, c='coral', linestyle="--", label="vortex 1, past")
    plt.plot(x2_history, y2_history, c='teal', linestyle="--",label="vortex 2, past")
    # plt.plot(np.arange(-5, 6), np.arange(-5, 6) * 0)
    plt.legend(loc=1)


def show_fig(lambd, value):
    lambd = str(round(lambd, 2))
    K = str(round(value, 3))
    name = lambd + "_" + K
    plt.title(r'$\lambda$'+" = "+ lambd + ", K = " + K, y=-0.13)
    plt.ylim(0)
    plt.show()

def save_fig(lambd, value, left = 1, right = -1):
    lambd = str(round(lambd, 2))
    K = str(round(value, 3))
    name = lambd + "_" + K
    plt.title(r'$\lambda$'+" = "+ lambd + ", K = " + K, y=-0.13)
    plt.ylim(0)
    if right != -1: plt.xlim(left, right)
    plt.savefig(name + ".png")
    plt.show()
    # plt.show(block=False)
    # plt.pause(1)
    # plt.close()


# Refactor in vector

def calculate_energy_vec(pos):
    x1, y1, s1 = pos[0]
    x2, y2, s2 = pos[1]
    return calculate_energy(x1, y1, x2, y2, s1, s2)


def calculate_p(pos):
    y_coord = pos[:, 1]
    s_lst = pos[:, 2]
    return y_coord @ s_lst


def calculate_c_vec(pos):
    x1, y1, s1 = pos[0]
    x2, y2, s2 = pos[1]
    e = calculate_energy(x1, y1, x2, y2, s1, s2)
    return np.exp(np.divide(-1 * 4 * np.pi * e, s1 ** 2))

def calculate_lambd_vec(pos):
    x1, y1, s1 = pos[0]
    x2, y2, s2 = pos[1]
    return s2/s1


def calculate_w(pos):
    c = calculate_c_vec(pos)
    p = calculate_p(pos)
    lambd = calculate_lambd_vec(pos)
    s = pos[0][2]
    w = np.power(p/s, 1+lambd*lambd)*c
    return w

def calculate_expression(pos):
    x = calculate_lambd_vec(pos)
    sq = 2*x+np.sqrt(4*x*x+1)
    num = np.power(x+sq, 1+x*x)*np.power(1-sq,2*x)
    dom = np.power(2,1+x*x)*(sq)*np.power(1+sq,2*x)
    return num/dom

def calculate_t_vec(pos):
    x1, y1, s1 = pos[0]
    x2, y2, s2 = pos[1]
    c = calculate_c(x1, y1, x2, y2, s1, s2)
    v = calculate_nu(x1, y1, x2, y2, s1, s2)
    lambd = np.divide(s2, s1)
    return c * v * np.power(np.abs(v), (np.square(lambd)))



def gen_plot_forward_vec(history):
    x1_history = history[:, 0, 0]
    y1_history = history[:, 0, 1]
    x2_history = history[:, 1, 0]
    y2_history = history[:, 1, 1]
    gen_plot_forward(x1_history, y1_history, x2_history, y2_history)


def gen_plot_backward_vec(history):
    x1_history = history[:, 0, 0]
    y1_history = history[:, 0, 1]
    x2_history = history[:, 1, 0]
    y2_history = history[:, 1, 1]
    gen_plot_backward(x1_history, y1_history, x2_history, y2_history)

def show_fig_vec(lambd, w, value):
    lambd = str(round(lambd, 2))
    k = str(round(value, 3))
    w = str(round(w, 3))
    name = lambd + "_" + k
    plt.title(r'$\lambda$'+" = " + lambd + ", W = " + w + ", bifur. = " + k, y=-0.13)
    plt.ylim(0)
    plt.show()
