import matplotlib.pyplot as plt
import numpy as np

def main():
    x = np.load('array_plot/array_step_random_sample.npy')
    y = np.load('array_plot/array_loss_random_sample.npy')

    x2 = np.load('array_plot/array_step.npy')
    y2 = np.load('array_plot/array_loss.npy')

    # x= x[11000:12000]
    # y = y[11000:12000]
    #
    # x2= x2[11000:12000]
    # y2 = y2[11000:12000]

    fig, axs = plt.subplots(1, 1, tight_layout=True)
    axs.plot(x,y, label="random sample")
    # axs.plot(x2, y2, label="priority")
    axs.set_title('Loss')
    axs.set_xlabel('time of training (times)')
    axs.set_ylabel('loss')
    axs.legend(loc='best')

    plt.show()

    plt.show()

if __name__ == '__main__':
    main()