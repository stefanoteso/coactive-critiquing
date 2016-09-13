#!/usr/bin/env python3

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from collections import defaultdict

CMAP = cm.ScalarMappable(cmap=plt.get_cmap("winter"),
                         norm=colors.Normalize(vmin=0, vmax=1))

def _get_ticks(x):
    return np.ceil(x / 10)

def _pad(m, nrows):
    new_m = []
    for row in m:
        assert row.shape[0] <= nrows
        new_m.append(np.hstack((row, np.zeros(nrows - row.shape[0]))))
    return np.array(new_m, dtype=m.dtype)

def _draw_matrices(ax, matrices, args, real_max_y, mean=False, cumulative=False,
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
        pf_matrices = [_pad(matrix, 100) for matrix in pf_matrices]
        avg_matrix = sum(pf_matrices) / len(pf_matrices)
        new_matrices.append(avg_matrix)
        new_args.append(pf_args[0])

    def key(matrix_arg):
        arg = matrix_arg[1]
        method = arg.method
        pf = 1 - arg.perc_feat
        pc = 1 - arg.p_critique if hasattr(arg, "p_critique") else -1.0
        return method, pf, pc

    matrices, args = zip(*sorted(zip(new_matrices, new_args), key=key,
                         reverse=True))

    max_x, max_y = None, None
    for matrix, arg in zip(matrices, args):
        if matrix.max() >= 1000:
            matrix /= 1000

        if arg.method == "cpp":
            if hasattr(arg, "p_critique"):
                fg = bg = CMAP.to_rgba(1.0 - arg.p_critique)
                label = "CC p={:3.2f}".format(arg.p_critique)
            else:
                fg = bg = "#EF2929"
                label = "CC"
            marker = "s-"
        else:
            perc_feat = 1.0 if arg.method == "pp-all" else arg.perc_feat
            fg = bg = CMAP.to_rgba(1.0 - (perc_feat - 0.2) / 0.8)
            marker = {0.0: "x-", 0.2: "v-", 0.4: "^-", 0.6: "<-", 0.8: ">-", 1.0: "D-"}[perc_feat]
            label = "CL {}%".format(int(perc_feat*100))


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

        plot = ax.plot(x, y, marker, linewidth=2.5, color=fg, label=label)
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.25, linewidth=0, color=bg)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)

    max_y += 5

    ax.set_xlim([0, 80])
    ax.set_ylim([0, real_max_y])

    ax.set_xticks(np.arange(0, 80, 10))
    ax.set_yticks(np.arange(0, real_max_y, _get_ticks(real_max_y)))

def main():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("png_basename", type=str,
                        help="plot path")
    parser.add_argument("results_path", type=str, nargs="+",
                        help="list of result files to plot")
    parser.add_argument("-l", "--max-loss", type=int, default=100,
                        help="I DONT GIVE A CRAP ANYMORE")
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

    _draw_matrices(loss_ax, loss_matrices, experiment_args, args.max_loss)
    loss_fig.savefig(args.png_basename + "_loss.png", bbox_inches="tight")

    _draw_matrices(query_ax, query_matrices, experiment_args, args.num_features+5, mean=True,
                   cumulative=True, num_features=args.num_features)
    query_fig.savefig(args.png_basename + "_query.png", bbox_inches="tight")

if __name__ == "__main__":
    main()
