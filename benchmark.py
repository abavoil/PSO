from algorithm import PSO
from visualizer import visualize2D
import matplotlib.pyplot as plt
import numpy as np
import time

"""
plot the mean of 20? results with 50? particles for phi1 in (0, .1, .2, .5, 1, 2, 5) (x log scale ?)
test for 10d ?
test for
"""

# boundaries for rastrigin
M = 5.12


def rastrigin(x):
    # can be applied to all arrays with previous last axis being
    n = x.shape[-1]
    return 10 * n + np.sum(x * x - 10 * np.cos(2 * np.pi * x), axis=-1)


def w(t):
    return 1 - t**3


def project_onto_domain(x):
    return np.clip(x, -M, M)


def test_phi1():
    nb_phi1 = 20
    tries_per_value = 50

    nb_particles = 10
    x0 = np.random.uniform(-M, M, (nb_particles, 2))
    max_iter = 100
    phi2 = .5
    # value relative to phi2
    Phi1 = phi2 * np.logspace(-2, 1, nb_phi1)

    # PSO(rastrigin, x0.copy(), max_iter, phi1, phi2, w, project_onto_domain)

    t0 = time.time()
    solverate = np.array([np.mean([np.linalg.norm(PSO(rastrigin, x0.copy(), max_iter, phi1, phi2, w, project_onto_domain)) < 1e-3
                                   for i in range(tries_per_value)])
                          for phi1 in Phi1])
    print(tries_per_value * nb_phi1, time.time() - t0)
    print(f"{solverate=}")
    # phi2 = .1
    # mean_error = [0.5721208610882622, 0.35777208268564437, 0.4847634175635628, 0.4083726533555915, 0.5875673254412085, 0.5493573398762917, 0.7211545688734958, 0.5697007585600631, 0.470307557184262, 0.5453901278926019, 0.772572392051146, 0.5086087308315459, 0.6498122941274377, 0.8190197648607243, 0.5499401544225342, 0.8722568927213057, 0.6718204112268401, 0.7906751032411645, 1.739928826388622, 3.2071398156482025]

    plt.plot(Phi1, solverate)
    plt.xscale("log")
    plt.suptitle("Erreur moyenne en fonction de phi1")
    plt.title("err = |rastrigin(gb)| avec gb le résultat de PSO")
    plt.show()


def time_benchmark():
    phi1 = .02
    phi2 = .1
    w = lambda t: 1 - t**4

    dim_list = (2, 3, 5, 10)
    nb_part_list = (10, 30, 50, 100, 1000)
    nb_iter_list = (100, 200, 500, 1000)
    nb_runs = 10

    for dim in dim_list:
        print(f"{dim=}")
        with open(f"time_benchmark_{dim}D.csv", "a") as output_file:
            output_file.write("nb_part\\nb_iter," + ",".join(map(str, nb_iter_list)))
            for nb_part in nb_part_list:
                print(f"    {nb_part=}")
                times = []
                x0 = np.random.uniform(-M, M, (nb_part, dim))
                for nb_iter in nb_iter_list:
                    print(f"        {nb_iter=}")
                    t0 = time.monotonic_ns()
                    for run in range(nb_runs):
                        PSO(rastrigin, x0.copy(), nb_iter, phi1, phi2, w, project_onto_domain, stable_tol=0, stable_iter=nb_iter)
                    times.append(round((time.monotonic_ns() - t0) / (1e6 * nb_runs), 1))
                line = f"{nb_part}," + ", ".join(map(str, times))
                output_file.write("\n" + line)


def time_benchmark_dimensions():
    phi1 = .02
    phi2 = .1
    w = lambda t: 1 - t**4

    dim_list = (10, 30, 100, 300, 1000, 3000, 10000)
    nb_iter = 100
    nb_part = 100
    nb_runs = 10
    times = []
    for dim in dim_list:
        x0 = np.random.uniform(-M, M, (nb_part, dim))
        t0 = time.monotonic_ns()
        for run in range(nb_runs):
            PSO(rastrigin, x0.copy(), nb_iter, phi1, phi2, w, project_onto_domain, stable_tol=0, stable_iter=nb_iter)
        t = round((time.monotonic_ns() - t0) / (1e6 * nb_runs), 1)
        times.append(t)
        print(f"{dim=}, {t=}")

    with open("time_benchmark_for_dimensions.csv", "a") as output_file:
        output_file.write("nb_part,nb_iter\\dim," + ",".join(map(str, dim_list)))
        output_file.write(f"\n{nb_part},{nb_iter}," + ",".join(map(str, times)))


def phi1_constant_benchmark():
    np.random.seed(0)

    phi2 = .1
    w = lambda t: 1 - t**4

    phi1_list = (0, .01, .05, .08, .09, .095, .1, .105, .11, .12, .15, .2, .3, .5, .8, 1, 2, 4)
    dim = 2
    nb_iter = 50
    nb_part = 200
    nb_runs = 100

    correct_counts = dict((phi1, 0) for phi1 in phi1_list)
    for phi1 in phi1_list:
        for run in range(nb_runs):
            x0 = np.random.uniform(-M, M, (nb_part, dim))
            best_found = PSO(rastrigin, x0, nb_iter, phi1, phi2, w, project_onto_domain)[0]
            if rastrigin(best_found) < 1e-3:
                correct_counts[phi1] += 1
        print(f"{phi1=}, correct={correct_counts[phi1]/nb_runs}")

    with open("phi1_cst_benchmark.csv", "a") as output_file:
        output_file.write("phi1,correctness\n")
        for phi1 in phi1_list:
            output_file.write(f"{phi1},{correct_counts[phi1]/nb_runs}\n")


if __name__ == '__main__':
    # test_phi1()
    # time_benchmark()
    # time_benchmark_dimensions()
    phi1_constant_benchmark()
