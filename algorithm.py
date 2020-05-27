import numpy as np
import numbers
import time

# np.random.seed(1)


def timeit(method):
    def timed(*args, **kw):
        t0 = time.time()
        result = method(*args, **kw)
        tf = time.time()
        if __name__ == '__main__':
            print(f"{method.__name__} {len(args[1])} points, {args[2]} itérations: {(tf - t0) * 1000} ms")

        return result

    return timed


@timeit
def PSO(f, x0, max_iter, phi1, phi2, w=None, project_onto_domain=None, stable_tol=1e-6, stable_iter=50, record_pos=False):
    """Minimize a function using Particle Swamp Optimization

    Parameters
    ----------
    f: function to optimize
       needs to be appliable to both individual particles (arrays of shape n)
       and arrays of particles (arrays of shape nb_particles * n)
       np.apply_along_axis can be used if no alternative is found
    x0: array of initial positions
    max_iter: number of iterations
    phi1, phi2, w: coefficients in the PSO equation
                   can either be numbers or functions of time
                   w defaults to lambda t: 1 - t if t > .6 else .4
    project_onto_domain: vectorized function that maps an array of points from
                         R^n onto their projections on the search domain
                         None if domain is R^n to save time
    stable_tol: tolerance to consider the barycenter stable from iteration to the following
    stable_iter: after stable_iter iterations of stable barycenter, the algorithm stops
    record_pos: if True, positions over iterations are kept and memory and
                     in the second argument (high memory usage)

    Returns
    ----------
    gb: the best position found
    pos_hist: the positions of particles over time (None if record_pos is False)

    Example call
    ----------
    PSO(lambda x, y: x**2 + y**2,
        np.array.uniform(-5, 5, (20, 2)),
        100, 1000, 1, 1,
        lambda t: 1 - .006*t if t < 1000 else .4)

    Notes
    Passing the domain through a function allows any shape of domain (cuboid, sphere,...)
    Points is the domain are left unchanged while points outside the domain are mapped to points inside.
    """

    # make all coefficients functions of time
    if isinstance(phi1, numbers.Number):
        phi1_val = phi1
        phi1 = lambda t: phi1_val

    if isinstance(phi2, numbers.Number):
        phi2_val = phi2
        phi2 = lambda t: phi2_val

    if w is None:
        w = lambda t: 1 - t if t > .6 else .4
    elif isinstance(phi1, numbers.Number):
        w_val = w
        phi1 = lambda t: w_val

    nparticles = x0.shape[0]

    # initialization
    vel = np.zeros(x0.shape)
    pos = x0
    # pb_val = np.apply_along_axis(f, 1, x0)
    pb_val = f(x0)
    pb_pos = x0

    gb_index = np.argmin(pb_val)
    gb_val = pb_val[gb_index]
    gb_pos = pb_pos[gb_index]

    prev_barycenter = None
    barycenter = np.sum(x0, 0) / nparticles
    stable_count = 0

    # record positions ?
    if record_pos:
        pos_hist = np.empty((max_iter + 1, nparticles, x0.shape[1]))
        pos_hist[:] = np.NaN
        pos_hist[0] = pos
    else:
        pos_hist = None

    # iterations of the algorithm
    for iter in range(max_iter):
        # # update velocity
        vel = w(iter / max_iter) * vel \
            + phi1(iter / max_iter) * np.random.rand(nparticles, 1) * (pb_pos - pos) \
            + phi2(iter / max_iter) * np.random.rand(nparticles, 1) * (gb_pos - pos)

        # update position
        pos = pos + vel
        if project_onto_domain is not None:
            pos = project_onto_domain(pos)

        # record positions ?
        if record_pos:
            pos_hist[iter + 1] = pos

        # evaluate f at each point of pos
        # val = np.apply_along_axis(f, 1, pos)
        val = f(pos)

        # update pb's
        # new_pb[i] true if val[i] is a new pb for particle i
        new_pb = val < pb_val
        # update only new pb's, and keep the rest
        pb_val[new_pb] = val[new_pb]
        pb_pos[new_pb] = pos[new_pb]

        # update gb
        min_pb_ind = np.argmin(pb_val)
        min_pb_val = pb_val[min_pb_ind]
        if np.any(new_pb) and min_pb_val < gb_val:
            gb_val = min_pb_val
            gb_pos = pos[min_pb_ind]

        # update barycenter, and potential early stop
        barycenter = np.sum(pos, 0) / nparticles
        if prev_barycenter is not None and np.linalg.norm(barycenter - prev_barycenter) < stable_tol:
            stable_count += 1
            if stable_count >= stable_iter:
                break
        else:
            stable_count = 0
        prev_barycenter = barycenter

    if record_pos:
        pos_hist = pos_hist[:iter + 1]

    return gb_pos, pos_hist


if __name__ == '__main__':
    import pickle

    n = 2
    M = 5.12

    def rastrigin(x):
        return 10 * n + np.sum(x * x - 10 * np.cos(2 * np.pi * x), axis=-1)

    def booth(x):
        return (x[0] + 2 * x[1] - 7)**2 + (2 * x[0] + x[1] - 5)**2

    def w(t):
        # t in [0, 1]
        return -t**2 + 1

    def project_onto_domain(x):
        # np.clip(x, [x0min, x1min], [x0max, x1max]) would do the same for n = 2
        return np.clip(x, -M, M)

    f = rastrigin
    nb_particles = 10
    x0 = np.random.uniform(-M, M, (nb_particles, n))
    max_iter = 100
    phi1, phi2 = .01, .05

    gb, data = PSO(f, np.copy(x0), max_iter, phi1, phi2, w, project_onto_domain, record_pos=True)
    print(data.shape)
