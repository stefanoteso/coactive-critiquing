#!/usr/bin/env python3

import numpy as np
import pymzn
import stfb

PROBLEMS = {
    "unconstr-bool":
        lambda args, rng:
            stfb.UnconstrBoolProblem(args.num_attributes, rng=rng),
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
            stfb.pp(problem, args.max_iters, "attributes", update=args.update),
    "pp-all":
        lambda args, problem:
            stfb.pp(problem, args.max_iters, "all", update=args.update),
    "critique-pp":
        lambda args, problem:
            stfb.critique_pp(problem, args.max_iters),
}

def main():
    import argparse

    np.seterr(all="raise")
    np.set_printoptions(linewidth=180)

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
        _, xs, losses, times = zip(*trace)
        all_losses.append(losses)
        all_times.append(times)
        print("\n\n")

    # XXX the drawing code goes here

if __name__ == "__main__":
    main()
