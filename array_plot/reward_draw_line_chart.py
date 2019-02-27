import os
import sys
import random
import matplotlib.pyplot as plt
import numpy as np
def main():
    x = np.load('array_plot/time_reward_t_plot_fix.npy')
    y = np.load('array_plot/reward_t_plot_fix.npy')

    x2 = np.load('array_plot/time_reward_t_plot.npy')
    y2 = np.load('array_plot/reward_t_plot.npy')

    plt.plot(x, y)
    plt.plot(x2, y2)
    plt.show()


if __name__ == '__main__':
    main()