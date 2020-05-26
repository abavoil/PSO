import numpy as np
import time


def timeit(method):
    def timed(*args, **kw):
        t0 = time.time()
        result = method(*args, **kw)
        tf = time.time()
        print(f"{len(x0)} points, {max_iter} itérations: {(tf - t0) * 1000} ms")

        return result

    return timed


@timeit
def PSO(f, x0, max_iter, phi1, phi2, w, project_onto_domain=None, stable_tol=1e-3, stable_iter=5):
    """Minimize a function using Particle Swamp Optimization

    Parameters
    ----------
    f: function to optimize
    x0: array of initial positions
    max_iter: number of iterations
    phi1, phi2: coefficients in teh PSO equation
    w: function of time with values between 0 (no inertia) and 1 (full inertia)
    project_onto_domain: vectorized function that maps an array of points from
                         R^n onto their projections on the search domain
                         None if domain is R^n to save time
    static_bary_th: tolerance to consider the barycenter stable from iteration to the following


    Example call
    ----------
    PSO(lambda x, y: x**2 + y**2,
        np.array.uniform(-5, 5, (20, 2)),
        100, 1000, 1, 1,
        lambda t: 1 - .006*t if t < 1000 else .4)

    Passing the domain through a function allows any shape of domain (cuboid, sphere,...)
    Points is the domain are left unchanged while points outside the domain are mapped to points inside.
    """

    nparticles = x0.shape[0]
    ndim = x0.shape[1]

    # positions, velocitys and pb's and gb's are stored in these arrays
    pos = np.empty((max_iter + 1, nparticles, ndim))
    velocity = np.empty((max_iter + 1, nparticles, ndim))
    pb_val = np.empty((max_iter + 1, nparticles))
    pb_pos = np.empty((max_iter + 1, nparticles, ndim))
    gb_val = np.empty((max_iter + 1,))
    gb_pos = np.empty((max_iter + 1, ndim))
    barycenter = np.empty((max_iter + 1, ndim))
    pos[:], velocity[:], pb_val[:], pb_pos[:], gb_val[:], gb_pos[:], barycenter[:] = [np.NaN] * 7

    # initialization
    pos[0] = x0
    velocity[0] = np.zeros(x0.shape)
    pb_val[0] = np.apply_along_axis(f, 1, x0)
    pb_pos[0] = x0

    gb_index = np.argmin(pb_val[0])
    gb_val[0] = pb_val[0][gb_index]
    gb_pos[0] = pb_pos[0][gb_index]

    barycenter[0] = np.sum(x0, 0) / nparticles
    stable_count = 0

    for iter in range(max_iter):
        # update velocity
        velocity[iter + 1] = w(iter / max_iter) * velocity[iter] \
            + phi1 * np.random.rand(nparticles, 1) * (pb_pos[iter] - pos[iter]) \
            + phi2 * np.random.rand(nparticles, 1) * (gb_pos[iter] - pos[iter])

        # update position
        pos[iter + 1] = pos[iter] + velocity[iter + 1]
        if project_onto_domain is not None:
            pos[iter] = project_onto_domain(pos[iter])

        # evaluate f at each point of pos
        val = np.apply_along_axis(f, 1, pos[iter + 1])

        # update pb's
        # new_pb[i] true if val[i] is a new pb for particle i
        new_pb = val < pb_val[iter]
        # update only new pb's, and keep the rest
        pb_val[iter + 1][new_pb] = val[new_pb]
        pb_pos[iter + 1][new_pb] = pos[iter][new_pb]
        pb_val[iter + 1][np.logical_not(new_pb)] = pb_val[iter][np.logical_not(new_pb)]
        pb_pos[iter + 1][np.logical_not(new_pb)] = pb_pos[iter][np.logical_not(new_pb)]

        # update gb
        min_pb_ind = np.argmin(pb_val[iter + 1])
        min_pb_val = pb_val[iter + 1][min_pb_ind]
        if np.any(new_pb) and min_pb_val < gb_val[iter]:
            gb_val[iter + 1] = min_pb_val
            gb_pos[iter + 1] = pos[iter + 1][min_pb_ind]
        else:
            gb_val[iter + 1] = gb_val[iter]
            gb_pos[iter + 1] = gb_pos[iter]

        # update barycenter
        barycenter[iter + 1] = np.sum(pos[iter + 1], 0) / nparticles
        if np.linalg.norm(barycenter[iter + 1] - barycenter[iter]) < stable_tol:
            stable_count += 1
            if stable_count >= stable_iter:
                break
        else:
            stable_count = 0

    final_iter = iter + 1
    # if early break because of stable barycenter, don't return nan's
    data_hist = {
        "position": pos[:final_iter],
        "velocity": velocity[:final_iter],
        "personal best": pb_pos[:final_iter],
        "global best": gb_pos[:final_iter],
        "barycenter": barycenter[:final_iter]
    }

    return gb_pos[final_iter], data_hist


"""
2 points 3 dimensions 4 étapes

gb :
[0, 0, 0]

position :
[
    [[1, 2, 0], [5, 2, 3]],
    [[1, 5, 0], [5, 2, 1]],
    [[1, 7, 0], [5, 2, -1]],
    [[1, 9, 0], [5, 2, -3]],
]

velocity :
[
    [[0, 0, 0], [0, 0, 0]]
    [[0, 3, 0], [0, 0, -2]],
    [[0, 2, 0], [0, 0, -2]],
    [[0, 2, 0], [0, 0, -2]],
    [[0, 0, 0], [0, 0, 0]],
]
"""

if __name__ == '__main__':
    import pickle

    n = 1
    M = 5.12

    def rastrigin(x):
        return 10 * n + np.linalg.norm(x)**2 - 10 * np.sum(np.cos(2 * np.pi * x))

    def norm(x):
        return np.linalg.norm(x)

    def w(t):
        return 1 - .06 * t if t < 10 else .4

    def project_onto_domain(x):
        # np.clip(x, [x0min, x1min], [x0max, x1max]) would work too for n = 2
        return np.clip(x, -M, M)

    f = rastrigin
    x0 = np.random.uniform(-M, M, (20, n))
    max_iter = 20
    phi1, phi2 = .5, .1
    # gb, data_hist = PSO(rastrigin, x0, max_iter, phi1, phi2, w, project_onto_domain)
    gb, data_hist = PSO(rastrigin, x0, max_iter, phi1, phi2, w)
    with open("data_hist.pickle", "wb") as pickle_out:
        pickle.dump(data_hist, pickle_out)
    print(len(data_hist["position"]), f(gb))
