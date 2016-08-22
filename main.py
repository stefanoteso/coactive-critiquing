#!/usr/bin/env python3

import pickle
import numpy as np
import pymzn
import stfb

PROBLEMS = {
    "canvas":
        lambda args, rng:
            stfb.CanvasProblem(noise=args.noise, sparsity=args.sparsity,
                               rng=rng),
    "pc":
        lambda args, rng:
            stfb.PCProblem(noise=args.noise, sparsity=args.sparsity, rng=rng),
    "travel":
        lambda args, rng:
            stfb.TravelProblem(noise=args.noise, sparsity=args.sparsity,
                               rng=rng),
}

LEARNERS = {
    "perceptron": stfb.Perceptron,
    "exp-perceptron": stfb.ExpPerceptron,
}

METHODS = {
    "pp-attr":
        lambda args, problem, rng:
            stfb.pp(problem, args.max_iters, "attributes",
                    Learner=LEARNERS[args.update], rng=rng, debug=args.debug),
    "pp-all":
        lambda args, problem, rng:
            stfb.pp(problem, args.max_iters, "all",
                    Learner=LEARNERS[args.update], rng=rng, debug=args.debug),
    "cpp":
        lambda args, problem, rng:
            stfb.pp(problem, args.max_iters, "attributes", can_critique=True,
                    Learner=LEARNERS[args.update], rng=rng, debug=args.debug),
}

def _get_experiment_name(args):
    return "_".join(map(str, [
        args.problem, args.method, args.num_users, args.max_iters,
        args.noise, args.sparsity, args.num_attributes, args.update,
        args.seed]))

def _to_matrix(l, rows=None, cols=None):
    if rows is None:
        rows = len(l)
    if cols is None:
        cols = max(map(len, l))
    m = np.zeros((rows, cols))
    for i, x in enumerate(l):
        m[i,:len(x)] = x
    return m

def main():
    import argparse

    np.seterr(all="raise")
    np.set_printoptions(precision=3, threshold=np.nan)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("problem", type=str, help="any of {}".format(list(PROBLEMS.keys())))
    parser.add_argument("method", type=str, help="any of {}".format(list(METHODS.keys())))
    parser.add_argument("-U", "--num-users", type=int, default=10,
                        help="number of users to average over")
    parser.add_argument("-T", "--max-iters", type=int, default=100,
                        help="maximum number of iterations")
    parser.add_argument("-u", "--update", type=str, default="perceptron",
                        help="pp update type")
    parser.add_argument("-S", "--sparsity", type=float, default=0.2,
                        help="percentage of non-zero weights")
    parser.add_argument("-E", "--noise", type=float, default=0.1,
                        help="amplitude of noise for improvement query")
    parser.add_argument("-n", "--num-attributes", type=int, default=10,
                        help="number of attributes, for problems that support it")
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="RNG seed")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="let structured feedback be verbose")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="let PyMzn be verbose")
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    pymzn.debug(args.verbose)

    SEP = "=" * 80

    # Run the main loop
    all_losses, all_times, all_is_critiques = [], [], []
    for i in range(args.num_users):
        print("{}\nUSER {}/{}\n{}".format(SEP, i, args.num_users, SEP))
        problem = PROBLEMS[args.problem](args, rng)
        num_iters, trace = METHODS[args.method](args, problem, rng)
        losses, times, is_critiques = zip(*trace)
        all_losses.append(losses)
        all_times.append(times)
        all_is_critiques.append(is_critiques)
        print("\n" * 5)

    # Dump the results on disk
    name = _get_experiment_name(args)
    data = {
        "experiment_args": args,
        "loss_matrix": _to_matrix(all_losses),
        "time_matrix": _to_matrix(all_times),
        "is_critiques": _to_matrix(all_is_critiques),
    }
    with open("results_" + name + ".pickle", "wb") as fp:
        pickle.dump(data, fp)

if __name__ == "__main__":
    main()
