import os
import sys
import random
import matplotlib.pyplot as plt
import numpy as np
def main():
    key = '_10000'
    x = np.load('array_plot/time_reward_t_plot_fix'+key+'.npy')
    y = np.load('array_plot/reward_t_plot_fix'+key+'.npy')

    x2 = np.load('array_plot/time_reward_t_plot'+key+'.npy')
    y2 = np.load('array_plot/reward_t_plot'+key+'.npy')

    plt.plot(x, y)
    plt.plot(x2, y2)
    plt.show()


if __name__ == '__main__':
    main()