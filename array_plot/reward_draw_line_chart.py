import os
import sys
import random
import matplotlib.pyplot as plt
import numpy as np

def main():
    key = "_EW"
    y = np.load('array_plot/array_total_reward'+key+'.npy')
    x = range(0, len(y))
    # x2 = np.load('array_plot/time_reward_t_plot'+key+'.npy')
    y2 = np.load('array_plot/array_total_reward_fix_40'+key+'.npy')
    y2 = np.full(len(x), y2[0])
    # print y2, len(x)
    y3 = np.load('array_plot/array_total_reward_fix_33' + key + '.npy')
    y3 = np.full(len(x), y3[0])

    fig, axs = plt.subplots(2, 1, tight_layout=True)
    axs[0].plot(x, y3, label="fix time 33")
    axs[0].plot(x, y2, label="fix time 40")
    axs[0].plot(x,y, label="DQN model")
    axs[0].set_title('Reward every eposide')
    axs[0].set_xlabel('eposides')
    axs[0].set_ylabel('reward')
    axs[0].legend(loc='best')

    # draw average waiting time
    y = np.load('array_plot/array_waiting_time_average'+key+'.npy')
    y = [i*14870/2920 for i in y]
    x = range(0, len(y))
    # x2 = np.load('array_plot/time_reward_t_plot'+key+'.npy')
    y2 = np.load('array_plot/array_waiting_time_average_fix_40' + key + '.npy')
    y2 = np.full(len(x), y2[0])
    # print y2, len(x)
    y3 = np.load('array_plot/array_waiting_time_average_fix_33' + key + '.npy')
    y3 = np.full(len(x), y3[0])

    axs[1].plot(x, y3, label="fix time 33")
    axs[1].plot(x, y2, label="fix time 40")
    axs[1].plot(x, y, label="DQN model")
    axs[1].set_title('Average waiting time every eposide')
    axs[1].set_xlabel('eposides')
    axs[1].set_ylabel('time (s)')
    axs[1].legend(loc='best')

    plt.show()


if __name__ == '__main__':
    main()