import os
import sys
import random
import matplotlib.pyplot as plt
import numpy as np

def main():
    key = '_10000'
    x = np.load('array_plot/array_time_fix'+key+'_33'+'.npy')
    y = np.load('array_plot/array_waiting_time_fix'+key+'_33'+'.npy')

    x5 = np.load('array_plot/array_time_fix'+key+'_40'+'.npy')
    y5 = np.load('array_plot/array_waiting_time_fix'+key+'_40'+'.npy')

    x2 = np.load('array_plot/array_time'+key+'.npy')
    y2 = np.load('array_plot/array_waiting_time'+key+'.npy')

    fig, axs = plt.subplots(2, 1, tight_layout=True)
    axs[0].plot(x,y, label="fix time 33")
    # axs[0].plot(x5, y5, label="fix time 40")
    axs[0].plot(x2, y2, label="DQN model")
    axs[0].set_title('Average Waiting Time')
    axs[0].set_xlabel('time (s)')
    axs[0].set_ylabel('waiting time (s)')
    axs[0].legend(loc='best')

    x3 = np.load('array_plot/time_reward_t_plot_fix'+key+'_33'+'.npy')
    y3 = np.load('array_plot/reward_t_plot_fix'+key+'_33'+'.npy')

    x6 = np.load('array_plot/time_reward_t_plot_fix' + key+'_40' + '.npy')
    y6 = np.load('array_plot/reward_t_plot_fix' + key+'_40' + '.npy')

    x4 = np.load('array_plot/time_reward_t_plot'+key+'.npy')
    y4 = np.load('array_plot/reward_t_plot'+key+'.npy')


    axs[1].plot(x3,y3, label="fix time 33")
    axs[1].plot(x6, y6, label="fix time 40")
    axs[1].plot(x4, y4, label="DQN model")
    axs[1].set_xlabel('time (s)')
    axs[1].set_title('Reward')
    axs[1].set_ylabel('reward')
    axs[1].legend(loc='best')

    mean = 0
    for i in range(0,len(y5)):
        mean += y5[i]
    mean = mean/len(y5)

    mean2 = 0
    for i in range(0,len(y2)):
        mean2 += y2[i]
    mean2 = mean2/len(y2)

    print (mean-mean2)/mean * 100

    plt.show()


if __name__ == '__main__':
    main()