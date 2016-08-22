# -*- encoding: utf-8 -*-

import numpy as np
import cvxpy as cvx
from textwrap import dedent
from time import time

# TODO the 'perturbed' pp algorithm is preferred for noisy users.

class Perceptron(object):
    """Implementation of the standard perceptron."""
    def __init__(self, problem, features, **kwargs):
        debug = kwargs.pop("debug", False)

        targets = problem.enumerate_features(features)

        # TODO sample form a standard normal if debug is False
        # XXX for sparse users perhaps sample from a sparse distribution
        self.w = np.zeros(problem.num_features, dtype=np.float32)
        self.w[targets] = np.ones(len(targets))

        self._debug = debug

    def default_weight(self, num_targets):
        # TODO sample from a standard normal if debug is False
        return 0.0

    def update(self, delta):
        self.w += delta


def is_separable(x, verbose=False):
    """Checks whether a dataset is separable using hard SVM."""
    n, d = x.shape
    if n < 2:
        return True

    w = cvx.Variable(d)

    norm_w = cvx.norm(w, 2)
    constraints = [cvx.sum_entries(x[i] * w) >= 1 for i in range(n)]

    problem = cvx.Problem(cvx.Minimize(norm_w), constraints)
    problem.solve(verbose=verbose)
    return w.value is not None


def pp(problem, max_iters, targets, Learner=Perceptron, can_critique=False,
       rng=None, debug=False):
    """The (Critiquing) Preference Perceptron [1]_.

    Contrary to the original algorithm:
    - There is no support for the "context" part.
    - The weight vector is initialized to (n**-2, ..., n**-2) rather than 0,
      to avoid having to solve inference with w == 0 (which is ambiguous).

    Termination occurs when (i) the user does not modify the proposal, or (ii)
    the maximum number of iterations is reached.

    Parameters
    ----------
    problem : Problem
        The target problem.
    max_iters : positive int
        Number of iterations.
    targets : str or list of int
        Indices or description of features describing the configuration space.
        "attributes" means only attribute-level features, "all" means all
        possible features. The space may change when can_critique is True.
    can_critique : bool, defaults to False
        Whether critique queries are enabled.
    Learner : class, defaults to Perceptron
        The learner to be used.
    rng : int or None
        The RNG.
    debug : bool, defaults to False
        Whether to spew debug output.

    Returns
    -------
    num_iters : int
        Number of iterations elapsed
    trace : list of numpy.ndarray of shape (num_features,)
        List of (loss, time) pairs for all iterations.

    References
    ----------
    .. [1] Shivaswamy and Joachims, *Coactive Learning*, JAIR 53 (2015)
    """
    learner = Learner(problem, targets, max_iters=max_iters, debug=debug)

    targets = problem.enumerate_features(targets)

    def delta(xs, targets):
        if isinstance(xs, tuple):
            return problem.phi(xs[0], targets) - problem.phi(xs[1], targets)
        return [problem.phi(x_bar, targets) - problem.phi(x, targets)
                for x_bar, x in xs]

    if debug:
        print(dedent("""\
            == USER INFO ==

            w_star =
            {}

            x_star =
            {}
            phi(x_star) =
            {}
            """).format(problem.w_star, problem.x_star,
                        problem.phi(problem.x_star, "all")))
    s = 0.0
    alpha = 50.0
    last_critique = True
    trace, dataset, deltas = [], [], []
    for it in range(max_iters):
        t0 = time()
        x = problem.infer(learner.w, targets)
        t0 = time() - t0

        loss = problem.utility_loss(x, "all")
        x_bar = problem.query_improvement(x, "all")

        t1 = time()
        is_satisfied = (x == x_bar).all()
        d = delta((x_bar, x), targets)

        if can_critique and len(dataset) > 0 and \
           not is_separable(np.vstack((deltas, d))):
            if not last_critique:
                s += 1
            p = (alpha * s) / (alpha * s  + (it + 1))
            ask_critique = rng.binomial(1, p)
            last_critique = bool(ask_critique)
        else:
            p = 0.0
            ask_critique = False
            last_critique = True
        t1 = time() - t1

        rho = None
        if not is_satisfied and ask_critique:
            rho, _ = problem.query_critique(x, x_bar, targets)
            assert rho > 0

        if debug:
            w = learner.w
            phi = problem.phi(x, "all")
            phi_bar = problem.phi(x_bar, "all")
            print(dedent("""\
                == ITERATION {it:3d} ==

                w =
                {w}
                loss = {loss}

                x =
                {x}
                phi(x) =
                {phi}

                x_bar =
                {x_bar}
                phi(x_bar) =
                {phi_bar}

                phi(x_bar) - phi(x) =
                {d}
                ask_critique = {ask_critique}
                p = {p}

                rho = {rho}

                is_satisfied = {is_satisfied}
                """).format(**locals()))

        t2 = time()
        if not is_satisfied:
            dataset.append((x_bar, x))
            if rho is not None:
                targets.append(rho)
                learner.w[rho] = learner.default_weight(len(targets))
                deltas = delta(dataset, targets)
            else:
                deltas.append(d)
            learner.update(d)
        t2 = time() - t2

        if debug:
            w = learner.w
            num_targets = len(targets)
            print(dedent("""\
                new w =
                {w}

                features = {targets}
                |features| = {num_targets}
                """).format(**locals()))

        is_critique = rho is not None
        trace.append((loss, t0 + t1, is_critique))
        if is_satisfied:
            if loss > 0:
                print("user is not satisfied, but can not improve item!")
            else:
                print("user is satisfied!")
            break
    else:
        print("user not satisfied, iterations elapsed")

    return it, trace
