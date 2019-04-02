import os
import sys
import random
import matplotlib.pyplot as plt
import numpy as np


def f(t):
    s1 = np.cos(2 * np.pi * t)
    e1 = np.exp(-t)
    return s1 * e1

def main():
    # key = '_10000'
    # t1 = np.load('array_plot/array_time_fix'+key+'.npy')
    # t2 = np.load('array_plot/array_waiting_time_fix'+key+'.npy')
    # t3 = np.arange(0.0, 2.0, 0.01)
    #
    # fig, axs = plt.subplots(2, 1)
    # axs[0].plot(t1,t2)
    # axs[0].set_title('subplot 1')
    # axs[0].set_xlabel('distance (m)')
    # axs[0].set_ylabel('Damped oscillation')
    # fig.suptitle('This is a somewhat long figure title', fontsize=16)
    #
    # axs[1].plot(t3, np.cos(2 * np.pi * t3), '--')
    # axs[1].set_xlabel('time (s)')
    # axs[1].set_title('subplot 2')
    # axs[1].set_ylabel('Undamped')
    #
    # plt.show()
    key = '_10000'
    x = np.load('array_plot/array_time_fix'+key+'.npy')
    y = np.load('array_plot/array_waiting_time_fix'+key+'.npy')

    x2 = np.load('array_plot/array_time'+key+'.npy')
    y2 = np.load('array_plot/array_waiting_time'+key+'.npy')

    fig, axs = plt.subplots(2, 1, tight_layout=True)
    axs[0].plot(x,y)
    axs[0].plot(x2, y2)
    axs[0].set_title('Waiting time')
    axs[0].set_xlabel('time (s)')
    axs[0].set_ylabel('waiting time (s)')
    # fig.suptitle('Line chart', fontsize=15)

    x3 = np.load('array_plot/time_reward_t_plot_fix'+key+'.npy')
    y3 = np.load('array_plot/reward_t_plot_fix'+key+'.npy')

    x4 = np.load('array_plot/time_reward_t_plot'+key+'.npy')
    y4 = np.load('array_plot/reward_t_plot'+key+'.npy')


    axs[1].plot(x3,y3)
    axs[1].plot(x4, y4)
    axs[1].set_xlabel('time (s)')
    axs[1].set_title('Reward')
    axs[1].set_ylabel('reward')

    plt.show()


if __name__ == '__main__':
    main()