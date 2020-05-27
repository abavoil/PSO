import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def progress_bar(t):
    length = 20
    filled = int(length * t)
    bar = "|" + filled * "#" + (length - filled) * "-" + "|"
    return bar


def visualize2D(data):
    """
    data[step][particle][dim]
    """
    fig, ax = plt.subplots()

    # same as X, Y = np.array([data[:, :, 0], data[:, :, 1]])
    X, Y = np.moveaxis(data, 2, 0)
    N = len(data)

    xmin, xmax = np.min(X), np.max(X)
    ymin, ymax = np.min(Y), np.max(Y)
    xpadding = .1 * (xmax - xmin)
    ypadding = .1 * (ymax - ymin)
    ax.set_xlim(xmin - xpadding, xmax + xpadding)
    ax.set_ylim(ymin - ypadding, ymax + ypadding)

    scat = ax.scatter(X[0], Y[0], c="r", s=10, marker="x")

    def animate(i):
        scat.set_offsets(data[i])
        ax.set_title(progress_bar(i / N))
        return (scat,)

    ani = FuncAnimation(fig, animate, frames=range(N), interval=1000 / 120)
    plt.show()


if __name__ == '__main__':
    from algorithm import PSO
    from benchmark import rastrigin as r, project_onto_domain as p

    n = 2
    w = lambda t: 1 - t**4

    nb_particles = 50
    x0 = np.random.uniform(-5.12, 5.12, (nb_particles, n))
    max_iter = 1000
    phi1 = lambda t: .1 if t > .8 else .03
    phi2 = .07

    gb, data = PSO(r, np.copy(x0), max_iter, phi1, phi2, w, p, record_pos=True)
    visualize2D(data)
