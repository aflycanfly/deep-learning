import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 这里设函数为y=3x+2
x_data = [1.0, 2.0, 3.0]
y_data = [5.0, 8.0, 11.0]


def forward(x):
    return x * w + b


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


W = np.arange(0.0, 6.1, 0.1)
B = np.arange(0.0, 4.1, 0.1)
[w, b] = np.meshgrid(W, B)


if __name__ == '__main__':
    mse_list = []

    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        print(y_pred_val)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(w, b, l_sum / 3)
    plt.show()
