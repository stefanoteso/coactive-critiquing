#!/usr/bin/env python3

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

_METHOD_TO_PLOTCONFIG = {
    "pp-attr":  ("#555753", "#2E3436", "D-"), # yellow
    "pp-all":   ("#A40000", "#CC0000", "s-"), # red
    "cpp":      ("#204A87", "#3465A4", "*-"), # blue
    "drone-cpp":("#4E9A06", "#73D216", "o-"), # green
}

def _get_ticks(x):
    return np.ceil(x / 10)

def _draw_matrices(ax, matrices, args, mean=False, cumulative=False, regret=False):
    max_x, max_y = None, None
    for i, (matrix, arg) in enumerate(zip(matrices, args)):
        fg, bg, marker = _METHOD_TO_PLOTCONFIG[arg.method]

        current_max_x = matrix.shape[1]
        if max_x is None or current_max_x > max_x:
            max_x = current_max_x

        x = np.arange(current_max_x)

        if cumulative or regret:
            matrix = matrix.cumsum(axis=1)

        if mean:
            y = np.mean(matrix, axis=0)
        else:
            y = np.median(matrix, axis=0)

        if regret:
            y /= np.arange(matrix.shape[1]) + 1

        yerr = np.std(matrix, axis=0) / np.sqrt(matrix.shape[0])

        current_max_y = max(y + yerr)
        if max_y is None or current_max_y > max_y:
            max_y = current_max_y

        ax.plot(x, y, marker, linewidth=2.5, color=fg)
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.35, linewidth=0, color=bg)

    ax.set_xlim([0, max_x + 0.1])
    ax.set_ylim([0, max_y + 0.1])

    ax.set_xticks(np.arange(0, max_x, _get_ticks(max_x)))
    ax.set_yticks(np.arange(0, max_y, _get_ticks(max_y)))

def main():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("png_basename", type=str,
                        help="plot path")
    parser.add_argument("results_path", type=str, nargs="+",
                        help="list of result files to plot")
    args = parser.parse_args()

    loss_fig, loss_ax = plt.subplots(1, 1)
    loss_ax.set_xlabel("Number of iterations")
    loss_ax.set_ylabel("Utility loss")

    regret_fig, regret_ax = plt.subplots(1, 1)
    regret_ax.set_xlabel("Number of iterations")
    regret_ax.set_ylabel("Regret")

    time_fig, time_ax = plt.subplots(1, 1)
    time_ax.set_xlabel("Number of iterations")
    time_ax.set_ylabel("Cumulative time (seconds)")

    query_fig, query_ax = plt.subplots(1, 1)
    query_ax.set_xlabel("Number of iterations")
    query_ax.set_ylabel("Number of critique queries")

    experiment_args, loss_matrices, time_matrices, query_matrices = [], [], [], []
    for path in args.results_path:
        with open(path, "rb") as fp:
            data = pickle.load(fp)
        experiment_args.append(data["experiment_args"])
        loss_matrices.append(data["loss_matrix"])
        time_matrices.append(data["time_matrix"])
        query_matrices.append(data["is_critiques"])
        assert loss_matrices[-1].shape == \
               time_matrices[-1].shape == \
               query_matrices[-1].shape
    _draw_matrices(loss_ax, loss_matrices, experiment_args)
    _draw_matrices(regret_ax, loss_matrices, experiment_args, regret=True)
    _draw_matrices(time_ax, time_matrices, experiment_args, cumulative=True)
    _draw_matrices(query_ax, query_matrices, experiment_args, mean=True)

    loss_fig.savefig(args.png_basename + "_loss.png", bbox_inches="tight")
    regret_fig.savefig(args.png_basename + "_regret.png", bbox_inches="tight")
    time_fig.savefig(args.png_basename + "_time.png", bbox_inches="tight")
    query_fig.savefig(args.png_basename + "_query.png", bbox_inches="tight")

if __name__ == "__main__":
    main()
