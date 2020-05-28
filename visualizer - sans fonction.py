import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from 


def z_function(x,y):
    return 20+x*x+y*y-10*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))


def visualize2D(data):
    fig, ax = plt.subplots()

    # same as X, Y = np.array([data[:, :, 0], data[:, :, 1]])
    X, Y = np.moveaxis(data, 2, 0)

    def z_function(x, y):
        return 20 + x * x + y * y - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))

    x = np.linspace(-5, 5, 500)
    y = np.linspace(-5, 5, 500)
    X, Y = np.meshgrid(x, y)
    Z = z_function(X, Y)
    i = 0
    cp =plt.contourf(X, Y, Z)

    plt.colorbar(cp)
    xmin, xmax = np.min(X), np.max(X)
    ymin, ymax = np.min(Y), np.max(Y)
    xpadding = .1 * (xmax - xmin)
    ypadding = .1 * (ymax - ymin)
    ax.set_xlim(xmin - xpadding, xmax + xpadding)
    ax.set_ylim(ymin - ypadding, ymax + ypadding)

    scat = ax.scatter(X[0], Y[0], c="r", s=10, marker="x")

    def animate(step):
        scat.set_offsets(step)
        return (scat,)

    ani = FuncAnimation(fig, animate, frames=data, interval=10, repeat_delay=2000)
    plt.show()



visualize2D(data)