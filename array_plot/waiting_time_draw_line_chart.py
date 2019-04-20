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

    # axs[1].plot(x,y, label="fix time 33")
    axs[1].plot(x5, y5, label="fix time 40")
    axs[1].plot(x2, y2, label="DQN model")
    axs[1].set_title('Average Waiting Time')
    axs[1].set_xlabel('time (s)')
    axs[1].set_ylabel('waiting time (s)')
    axs[1].legend(loc='best')

    average_waiting_time_fix33 = np.load('array_plot/array_waiting_time_average_fix' + key + '_33'+'.npy')[0]
    average_waiting_time_fix40 = np.load('array_plot/array_waiting_time_average_fix' + key + '_40' + '.npy')[0]
    average_waiting_time_model = np.load('array_plot/array_waiting_time_average' + key + '.npy')[0]
    print sum(y), sum(y2), (sum(y) - sum(y2)) / sum(y), average_waiting_time_model,average_waiting_time_fix33,((average_waiting_time_fix33- average_waiting_time_model) / float(average_waiting_time_fix33)) * 100
    print sum(y5),sum(y2),(sum(y5)-sum(y2))/sum(y5), average_waiting_time_model,average_waiting_time_fix40, ((average_waiting_time_fix40 - average_waiting_time_model) / float(average_waiting_time_fix40)) * 100

    plt.show()


if __name__ == '__main__':
    main()