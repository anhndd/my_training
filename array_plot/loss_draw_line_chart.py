import matplotlib.pyplot as plt
import numpy as np

def main():
    key = '_10000'
    x = np.load('array_plot/array_step.npy')
    y = np.load('array_plot/array_loss.npy')

    fig, axs = plt.subplots(1, 1, tight_layout=True)
    axs.plot(x,y)
    axs.set_title('Loss')
    axs.set_xlabel('time of training (times)')
    axs.set_ylabel('loss')

    plt.show()


if __name__ == '__main__':
    main()