import matplotlib.pyplot as plt
import numpy as np

def main():
    x = np.load('array_plot/array_step.npy')
    y = np.load('array_plot/array_loss.npy')

    x2 = np.load('array_plot/array_step_random_sample.npy')
    y2 = np.load('array_plot/array_loss_random_sample.npy')

    fig, axs = plt.subplots(1, 1, tight_layout=True)
    axs.plot(x2,y2, label="random sample")
    axs.plot(x,y, label="priority experience")
    axs.set_title('Loss')
    axs.set_xlabel('time of training (times)')
    axs.set_ylabel('loss')
    axs.legend(loc='best')

    plt.show()


if __name__ == '__main__':
    main()