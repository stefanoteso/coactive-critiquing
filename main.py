#!/usr/bin/env python3

import pickle
import numpy as np
import pymzn
import stfb

PROBLEMS = {
    "canvas":
        lambda args, rng, w_star:
            stfb.CanvasProblem(noise=args.noise, sparsity=args.sparsity,
                               rng=rng, w_star=w_star),
    "pc":
        lambda args, rng, w_star:
            stfb.PCProblem(noise=args.noise, sparsity=args.sparsity, rng=rng,
                           w_star=w_star),
    "travel":
        lambda args, rng, w_star:
            stfb.TravelProblem(noise=args.noise, sparsity=args.sparsity,
                               rng=rng, w_star=w_star),
}

LEARNERS = {
    "perceptron": stfb.Perceptron,
}

METHODS = {
    "pp-attr":
        lambda args, problem, num_critiques, rng:
            stfb.pp(problem, args.max_iters, "attributes",
                    Learner=LEARNERS[args.update],
                    perturbation=args.perturbation,
                    rng=rng, debug=args.debug, gamma=args.gamma),
    "pp-all":
        lambda args, problem, num_critiques, rng:
            stfb.pp(problem, args.max_iters, "all",
                    Learner=LEARNERS[args.update],
                    perturbation=args.perturbation,
                    rng=rng, debug=args.debug, gamma=args.gamma),
    "cpp":
        lambda args, problem, num_critiques, rng:
            stfb.pp(problem, args.max_iters, "attributes", can_critique=True,
                    Learner=LEARNERS[args.update],
                    perturbation=args.perturbation,
                    rng=rng, debug=args.debug, gamma=args.gamma),
    "drone-cpp":
        lambda args, problem, num_critiques, rng:
            stfb.pp(problem, args.max_iters, "attributes", can_critique=True,
                    num_critiques=num_critiques,
                    Learner=LEARNERS[args.update],
                    perturbation=args.perturbation,
                    rng=rng, debug=args.debug, gamma=args.gamma),
}

def _get_experiment_path(args, method=None):
    method = args.method if method is None else method
    name = "_".join(map(str, [
        args.problem, method, args.num_users, args.max_iters,
        args.noise, args.sparsity, args.update, args.perturbation,
        args.seed]))
    return "results_" + name + ".pickle"

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
    parser.add_argument("-S", "--sparsity", type=float, default=0.2,
                        help="percentage of non-zero weights")
    parser.add_argument("-E", "--noise", type=float, default=0.1,
                        help="amplitude of noise for improvement query")
    parser.add_argument("-u", "--update", type=str, default="perceptron",
                        help="pp update type")
    parser.add_argument("-p", "--perturbation", type=float, default=0.0,
                        help="amount of inference perturbation")
    parser.add_argument("-g", "--gamma", type=float, default=1.0,
                        help="coefficient for probabilistic query critique "
                             "selection")
    parser.add_argument("-W", "--weights", type=str, default=None,
                        help="path to pickle file with user weights")
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="RNG seed")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="let structured feedback be verbose")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="let PyMzn be verbose")
    args = parser.parse_args()

    pymzn.debug(args.verbose)

    SEP = "=" * 80

    old_num_critiques = None
    if args.method == "drone-cpp":
        drone_path = _get_experiment_path(args, method="cpp")
        with open(drone_path, "rb") as fp:
            old_is_critiques = pickle.load(fp)["is_critiques"]
        assert old_is_critiques.shape[0] == args.num_users
        assert old_is_critiques.shape[1] <= args.max_iters
        old_num_critiques = np.sum(old_is_critiques, axis=1).astype(int)

    weights = None
    if args.weights is not None:
        weights = pickle.load(open(args.weights, 'rb'))

    # Run the main loop
    all_losses, all_times, all_is_critiques = [], [], []
    for i in range(args.num_users):
        print("{}\nUSER {}/{}\n{}".format(SEP, i, args.num_users, SEP))
        rng = np.random.RandomState(args.seed + i)
        w_star = None
        if weights is not None:
            w_star = weights[i]
        problem = PROBLEMS[args.problem](args, rng, w_star)
        num_critiques_for_user = None
        if args.method == "drone-cpp":
            num_critiques_for_user = old_num_critiques[i]
        num_iters, trace = METHODS[args.method](args, problem,
                                                num_critiques_for_user, rng)
        losses, times, is_critiques = zip(*trace)
        all_losses.append(losses)
        all_times.append(times)
        all_is_critiques.append(is_critiques)
        print("\n" * 5)

    # Dump the results on disk
    data = {
        "experiment_args": args,
        "loss_matrix": _to_matrix(all_losses),
        "time_matrix": _to_matrix(all_times),
        "is_critiques": _to_matrix(all_is_critiques),
    }
    with open(_get_experiment_path(args), "wb") as fp:
        pickle.dump(data, fp)

if __name__ == "__main__":
    main()
