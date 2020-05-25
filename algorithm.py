import numpy as np
from copy import copy


def PSO(f, x0, boundaries, nsteps, phi1, phi2, w):
    """
    Finds the global minimum of f over a cuboid using a particle swamp optimisation algorithm


    f: function to optimize
    x0: array of initial positions
    boundaries: numpy array of mins and maxs of each dimension [[min1, min2, min3], [max1, max2, max3]]
    nsteps: number of steps
    phi1, phi2: coefficients in teh PSO equation
    w: function of time with values between 0 (no inertia) and 1 (full inertia)

    example call:
    PSO(lambda x, y: x**2 + y**2, np.array([(-5, -5), (5, 5)]), 100, 1000, 1, 1, lambda t: 1 - .006*t if t < 1000 else .4)
    """

    nparticles = x0.shape[0]
    ndim = x0.shape[1]

    # positions, speeds and pb's are stored in these arrays
    pos = np.empty((nsteps + 1, nparticles, ndim))
    speed = np.empty((nsteps + 1, nparticles, ndim))
    gb_val = np.empty((nsteps + 1, nparticles))
    gb_pos = np.empty((nsteps + 1, nparticles, ndim))
    pos[:], speed[:], gb_val[:], gb_pos[:] = [np.NaN] * 4

    # initialization
    pos[0] = x0
    speed[0] = np.zeros(x0.shape)
    gb_val[0] = np.min(np.apply_along_axis(f, 1, pos))
    gb_pos[0] = pos[np.argwhere(gb_val == gb_val)]

    for step in range(nsteps):

        speed[step + 1] = w(step / nsteps) * speed[step] \
            + phi1 * np.random.rand(nparticles) * (pb_pos[step] - pos[step]) \
            + phi2 * np.random.rand(nparticles) * (gb_pos[step] - pos[step])

        pos[step + 1] = pos[step] + speed[step + 1]

        val = np.apply_along_axis(f, 1, pos)

        # new_pb[i] true if val[i] is a new pb for particle i
        new_pb = val < pb_val
        pb_val[new_pb] = val[new_pb]
        pb_pos[new_pb] = pos[new_pb]

        if np.any(new_pb):
            gb_index_candidate = np.argmin(pb_val)
            if pb_val[i] < gb_val:
                gb_val = pb_val[i]
                gb_pos = pb_pos[i]


        # constraint
        # for i, coord in enumerate(pos):
        #     pos[step] = np.min(np.max(coord, boundaries[0, i]), boundaries[1, i])

        # stop barycentre (stable_iter)

    return gb_pos, pos, speed


"""
2 points 3 dimensions 4 étapes

gb :
[0, 0, 0]

positions :
[
    [[1, 2, 0], [5, 2, 3]],
    [[1, 5, 0], [5, 2, 1]],
    [[1, 7, 0], [5, 2, -1]],
    [[1, 9, 0], [5, 2, -3]],
]

vitesses :
[
    [[0, 0, 0], [0, 0, 0]]
    [[0, 3, 0], [0, 0, -2]],
    [[0, 2, 0], [0, 0, -2]],
    [[0, 2, 0], [0, 0, -2]],
    [[0, 0, 0], [0, 0, 0]],
]
"""

if __name__ == '__main__':
    def f(x): return x[0]**2 + x[1]**2
    x0 = np.random.uniform(-1, 1, (3, 2))
    def w(t): return 1 - .006 * t if t < 1000 else .4
    boundaries = np.array([(-1, -1), (1, 1)])
    nsteps = 10
    phi1, phi2 = 1, 1
    gb, positions, speeds = PSO(f, x0, boundaries, nsteps, phi1, phi2, w)
    print(gb)
