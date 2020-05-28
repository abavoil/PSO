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
    nb_iter = 100
    nb_part = 50
    nb_runs = 100

    with open("phi1_cst_benchmark.csv", "a") as output_file:
        output_file.write("phi1,correctness\n")
        for phi1 in phi1_list:
            correct_counts = 0
            for run in range(nb_runs):
                x0 = np.random.uniform(-M, M, (nb_part, dim))
                best_found = PSO(rastrigin, x0, nb_iter, phi1, phi2, w, project_onto_domain)[0]
                if rastrigin(best_found) < 1e-3:
                    correct_counts += 1
            print(f"{phi1=}, correct={correct_counts/nb_runs}")
            output_file.write(f"{phi1},{correct_counts/nb_runs}\n")


def size_nb_iter_benchmark():
    np.random.seed(0)

    phi1 = .1
    phi2 = .1
    w = lambda t: 1 - t**4

    nb_part_list = (10, 20, 30, 50, 70, 100, 300, 1000)
    nb_iter_list = (50, 100, 300, 1000, 3000)
    dim = 2
    nb_runs = 100

    with open(f"swarm_size_nb_iterations_benchmark.csv", "a") as output_file:
        output_file.write("nb_part\\nb_iter," + ",".join(map(str, nb_iter_list)))
        for nb_part in nb_part_list:
            print(f"{nb_part=}")
            success_rates = []
            for nb_iter in nb_iter_list:
                print(f"    {nb_iter=}")
                success_count = 0
                for run in range(nb_runs):
                    x0 = np.random.uniform(-M, M, (nb_part, dim))
                    best_found = PSO(rastrigin, x0, nb_iter, phi1, phi2, w, project_onto_domain)[0]
                    if rastrigin(best_found) < 1e-3:
                        success_count += 1
                success_rates.append(success_count / nb_runs)
            line = f"{nb_part}," + ", ".join(map(str, success_rates))
            output_file.write("\n" + line)


def validation():
    np.random.seed(0)

    def booth(x):
        x_, y_ = np.moveaxis(x, -1, 0)
        return (x_ + 2 * x_ - 7)**2 + (2 * x_ + y_ - 5)**2

    def himmelblau(x):
        x_, y_ = np.moveaxis(x, -1, 0)
        return (x_ * x_ + y_ - 11)**2 + (x_ + y_**2 - 7)**2

    def holder_table(x):
        x_, y_ = np.moveaxis(x, -1, 0)
        return -np.cos(x_) * np.cos(y_) * np.exp(-((x_ - np.pi)**2 + (y_ - np.pi)**2))

    phi1 = .1
    phi2 = .1
    w = lambda t: 1 - t**4
    dim = 2
    nb_part = 30
    nb_iter = 100
    nb_runs = 1000

    functions = [rastrigin, booth, himmelblau, holder_table]
    boundary = [5.12, 5, 10, 100]
    pod = [lambda x: np.clip(x, -M, M) for M in boundary]

    # min_positions = [np.array([[1, 3]]),
    #                  np.array([[3, 2],
    #                            [-2.805118, 3.131312],
    #                            [-3.77931, -3.283186],
    #                            [3.584428, -1.848126]]),
    #                  np.array([np.pi, np.pi])]

    min_values = [0, 0, 0, -1]

    with open("validation.csv", "a") as output_file:
        output_file.write("function,correctness\n")
        for i in range(len(functions)):
            correct_count = 0
            for run in range(nb_runs):
                x0 = np.random.uniform(-boundary[i], boundary[i], (nb_part, 2))
                best_found, _ = PSO(functions[i], x0, nb_iter, phi1, phi2, w, pod[i])
                if np.abs(functions[i](best_found) - min_values[i]) < 1e-3:
                    correct_count += 1
            print(f"{functions[i].__name__},{correct_count/nb_runs}")
            output_file.write(f"{functions[i].__name__},{correct_count/nb_runs}\n")


if __name__ == '__main__':
    # time_benchmark()
    # time_benchmark_dimensions()
    # phi1_constant_benchmark()
    # size_nb_iter_benchmark()
    validation()
    pass
