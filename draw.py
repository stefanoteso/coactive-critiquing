#!/usr/bin/env python3

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

_METHOD_TO_PLOTCONFIG = {
    "pp-attr":  ("#555753", "#2E3436", "D-"),
    "pp-all":   ("#CC0000", "#EF2929", "o-"),
    "cpp":      ("#73D216", "#8AE234", "s-"),
}

def _get_ticks(x):
    return np.ceil(x / 10)

def _draw_matrices(ax, matrices, args, mean=False, cumulative=False,
                   num_features=None):

    new_matrices, new_args = [], []

    pf_to_matrices_args = defaultdict(list)
    for matrix, arg in zip(matrices, args):
        if arg.method == "pp-attr":
            pf_to_matrices_args[arg.perc_feat].append((matrix, arg))
        else:
            new_matrices.append(matrix)
            new_args.append(arg)

    for pf, matrices_args in pf_to_matrices_args.items():
        pf_matrices, pf_args = zip(*matrices_args)
        avg_matrix = sum(pf_matrices) / len(pf_matrices)
        new_matrices.append(avg_matrix)
        new_args.append(pf_args[0])

    matrices, args = new_matrices, new_args

    max_x, max_y = None, None
    for i, (matrix, arg) in enumerate(zip(matrices, args)):
        fg, bg, marker = _METHOD_TO_PLOTCONFIG[arg.method]
        if arg.method == "pp-attr":
            fg = "#{:02x}00FF".format(int(arg.perc_feat * 255))
            bg = "#{:02x}00FF".format(int(arg.perc_feat * 255))

        current_max_x = matrix.shape[1]
        if max_x is None or current_max_x > max_x:
            max_x = current_max_x

        x = np.arange(current_max_x)

        if cumulative:
            matrix = matrix.cumsum(axis=1)

        if mean:
            y = np.mean(matrix, axis=0)
        else:
            y = np.median(matrix, axis=0)

        yerr = np.std(matrix, axis=0) / np.sqrt(matrix.shape[0])

        if num_features is not None:
            if arg.method == "pp-attr":
                y = np.ones_like(y) * arg.perc_feat * num_features
                yerr = np.zeros_like(y)
            elif arg.method == "pp-all":
                y = np.ones_like(y) * num_features
                yerr = np.zeros_like(y)

        current_max_y = max(y + yerr)
        if max_y is None or current_max_y > max_y:
            max_y = current_max_y

        ax.plot(x, y, marker, linewidth=2.5, color=fg)
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.5, linewidth=0, color=bg)

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
    parser.add_argument("-m", "--num-features", type=int, default=100,
                        help="total number of features")
    args = parser.parse_args()

    loss_fig, loss_ax = plt.subplots(1, 1)
    loss_ax.set_xlabel("Iterations")
    loss_ax.set_ylabel("Utility loss")

    time_fig, time_ax = plt.subplots(1, 1)
    time_ax.set_xlabel("Iterations")
    time_ax.set_ylabel("Cumulative time (seconds)")

    query_fig, query_ax = plt.subplots(1, 1)
    query_ax.set_xlabel("Iterations")
    query_ax.set_ylabel("Number of features")

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
    loss_fig.savefig(args.png_basename + "_loss.png", bbox_inches="tight")

    _draw_matrices(time_ax, time_matrices, experiment_args, cumulative=True)
    time_fig.savefig(args.png_basename + "_time.png", bbox_inches="tight")

    _draw_matrices(query_ax, query_matrices, experiment_args, mean=True,
                   cumulative=True, num_features=args.num_features)
    query_fig.savefig(args.png_basename + "_query.png", bbox_inches="tight")

if __name__ == "__main__":
    main()
