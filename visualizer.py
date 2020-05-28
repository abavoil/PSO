import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


def progress_bar(t):
    length = 20
    filled = int(length * t)
    bar = "|" + filled * "#" + (length - filled) * "-" + "|"
    return bar


def visualize2D(data, f=None):
    """
    data[step][particle][dim]
    """
    fig, ax = plt.subplots()

    X, Y = np.moveaxis(data, 2, 0)
    N = len(data)

    xmin, xmax = np.min(X), np.max(X)
    ymin, ymax = np.min(Y), np.max(Y)
    xmin, xmax = xmin - .1 * (xmax - xmin), xmax + .1 * (xmax - xmin)
    ymin, ymax = ymin - .1 * (ymax - ymin), ymax + .1 * (ymax - ymin)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if f is not None:
        X_bg, Y_bg = np.array(np.meshgrid(np.arange(xmin, xmax, .02),
                                          np.arange(ymin, ymax, .02)))
        Z = f(np.moveaxis([X_bg, Y_bg], 0, -1))
        ax.contourf(X_bg, Y_bg, Z, 10, cmap=plt.cm.jet, zorder=0)
        ax.contour(X_bg, Y_bg, Z, 3, colors="b", zorder=1)

    scat = ax.scatter(X[0], Y[0], c="r", s=10, marker="x", zorder=2)

    def init():
        pass

    def animate(i):
        t = i / N
        scat.set_offsets(data[i])
        ax.set_title(progress_bar(t), fontdict={'family': 'monospace'})
        return (scat,)

    ani = FuncAnimation(fig, animate, init_func=init, frames=range(N), interval=1000 / 60)
    # ani.save('myAnimation.gif', writer='imagemagick', fps=30)
    plt.show()


if __name__ == '__main__':
    from algorithm import PSO
    from benchmark import rastrigin, project_onto_domain

    # f = lambda x: np.linalg.norm(x, axis=-1)
    f = rastrigin
    p = project_onto_domain
    n = 2
    w = lambda t: (1 - t**4)
    # w = lambda t: 1.01 if t < .8 else 0
    # w = .9

    nb_particles = 20
    x0 = np.random.uniform(-5.12, 5.12, (nb_particles, n))
    max_iter = 200
    phi1 = lambda t: .07 if t < .5 else .01
    phi2 = .07

    gb, data = PSO(f, np.copy(x0), max_iter, phi1, phi2, w, p, record_pos=True)
    visualize2D(data, f=f)
