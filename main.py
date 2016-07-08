#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pymzn
import stfb

PROBLEMS = {
    "rand-constr-bool":
        lambda args, rng:
            stfb.RandConstrBoolProblem(args.num_attributes, rng=rng),
    "pc":
        lambda args, rng: stfb.PCProblem(rng=rng),
    "travel":
        lambda arg, rng: stfb.TravelProblem(rng=rng),
}

METHODS = {
    "pp-attributes":
        lambda args, problem:
            stfb.pp(problem, args.max_iters, "attributes", update=args.update,
                    debug=args.debug),
    "pp-all":
        lambda args, problem:
            stfb.pp(problem, args.max_iters, "all", update=args.update,
                    debug=args.debug),
    "critique-pp":
        lambda args, problem:
            stfb.critique_pp(problem, args.max_iters, debug=args.debug),
}

def _get_experiment_name(args):
    return "_".join(map(str, [
        args.problem, args.method, args.num_users, args.max_iters,
        args.num_attributes, args.update, args.seed]))

def _to_matrix(l, rows=None, cols=None):
    if rows is None:
        rows = len(l)
    if cols is None:
        cols = max(map(len, l))
    m = np.zeros((rows, cols))
    for i, x in enumerate(l):
        m[i,:len(x)] = x
    return m

def _get_ticks(x):
    return x / 10 # XXX
    offset = np.floor(x / 10.0)
    decimals = -np.log10(offset) + 1
    return int(np.round(offset, decimals))

def _draw_matrices(ax, matrices, cumulative=False):
    max_x, max_y = None, None
    for i, matrix in enumerate(matrices):

        current_max_x = matrix.shape[1]
        if max_x is None or current_max_x > max_x:
            max_x = current_max_x

        x = np.arange(current_max_x)

        if cumulative:
            matrix = matrix.cumsum(axis=1)

        y = np.median(matrix, axis=0)
        yerr = np.std(matrix, axis=0) / np.sqrt(matrix.shape[0])

        current_max_y = max(y + yerr)
        if max_y is None or current_max_y > max_y:
            max_y = current_max_y

        ax.plot(x, y, "o-", linewidth=2.5)
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.35, linewidth=0)

    ax.set_xlim([0, max_x])
    ax.set_ylim([0, max_y])

    ax.set_xticks(np.arange(0, max_x, _get_ticks(max_x)))
    ax.set_yticks(np.arange(0, max_y, _get_ticks(max_y)))

def draw(basename, loss_matrix, time_matrix):
    assert loss_matrix.shape == time_matrix.shape

    fig, ax = plt.subplots(1, 1)
    _draw_matrices(ax, [loss_matrix], cumulative=False)
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Utility loss")
    fig.savefig(basename + "_loss.png", bbox_inches="tight")

    fig, ax = plt.subplots(1, 1)
    _draw_matrices(ax, [loss_matrix], cumulative=True)
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Cumulative time (seconds)")
    fig.savefig(basename + "_time.png", bbox_inches="tight")

def main():
    import argparse

    np.seterr(all="raise")
    np.set_printoptions(threshold=np.nan, linewidth=180)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("problem", type=str, help="any of {}".format(PROBLEMS.keys()))
    parser.add_argument("method", type=str, help="any of {}".format(METHODS.keys()))
    parser.add_argument("-U", "--num-users", type=int, default=10,
                        help="number of users to average over")
    parser.add_argument("-T", "--max-iters", type=int, default=100,
                        help="maximum number of iterations")
    parser.add_argument("-n", "--num-attributes", type=int, default=10,
                        help="number of attributes, for problems that support it")
    parser.add_argument("-u", "--update", type=str, default="perceptron",
                        help="pp update type")
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="RNG seed")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="let structured feedback be verbose")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="let PyMzn be verbose")
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    pymzn.verbose(args.verbose)

    # Run the main loop
    all_losses, all_times = [], []
    for i in range(args.num_users):
        print("==== USER {}/{} ====".format(i, args.num_users))
        problem = PROBLEMS[args.problem](args, rng)
        _, trace = METHODS[args.method](args, problem)
        _, _, losses, times = zip(*trace)
        all_losses.append(losses)
        all_times.append(times)
        print("\n\n")

    draw(_get_experiment_name(args),
         _to_matrix(all_losses),
         _to_matrix(all_times))

if __name__ == "__main__":
    main()
