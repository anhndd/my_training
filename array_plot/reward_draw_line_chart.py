import os
import sys
import random
import matplotlib.pyplot as plt
import numpy as np
def main():
    key = "_10000"
    y = np.load('array_plot/array_total_reward.npy')
    x = np.load('array_plot/array_episode.npy')
    # x2 = np.load('array_plot/time_reward_t_plot'+key+'.npy')
    y2 = np.load('array_plot/array_total_reward_fix'+key+'_40.npy')
    y2 = np.full(len(x), y2[0])
    # print y2, len(x)
    y3 = np.load('array_plot/array_total_reward_fix' + key + '_33.npy')
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
    y = np.load('array_plot/array_waiting_time_average.npy')
    x = np.load('array_plot/array_episode.npy')
    # x2 = np.load('array_plot/time_reward_t_plot'+key+'.npy')
    y2 = np.load('array_plot/array_waiting_time_average_fix' + key + '_40.npy')
    y2 = np.full(len(x), y2[0])
    # print y2, len(x)
    y3 = np.load('array_plot/array_waiting_time_average_fix' + key + '_33.npy')
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