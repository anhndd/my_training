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

    fig, axs = plt.subplots(1, 1, tight_layout=True)
    axs.plot(x, y3, label="fix time 33")
    axs.plot(x, y2, label="fix time 40")
    axs.plot(x,y, label="DQN model")
    axs.set_title('Reward')
    axs.set_xlabel('eposides')
    axs.set_ylabel('reward')
    axs.legend(loc='best')

    plt.show()


if __name__ == '__main__':
    main()