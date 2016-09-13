# -*- encoding: utf-8 -*-

import numpy as np
import cvxpy as cvx
from sklearn.utils import check_random_state
from textwrap import dedent
from time import time


def is_separable(deltas, d, verbose=False):
    """Checks whether a dataset is separable using hard SVM."""
    if len(deltas) <= 1:
        return True

    x = np.vstack((deltas, d))
    n, k = x.shape

    w = cvx.Variable(k)

    norm_w = cvx.norm(w, 2)
    constraints = [cvx.sum_entries(x[i] * w) >= 1 for i in range(n)]

    problem = cvx.Problem(cvx.Minimize(norm_w), constraints)
    problem.solve(verbose=verbose)
    return w.value is not None


def pp(problem, max_iters, targets, can_critique=False, num_critiques=None,
       fill_with_ones=False, p_critique=None, rng=None, debug=False):
    """The (Critiquing) Preference Perceptron [1]_.

    Contrary to the original algorithm, there is no support for the "context"
    part.

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
    num_critiques : int, defaults to None
        How many critiques to ask the user for. Critiques are allocated
        uniformly at random in the max_iter iterations. If None, the usual
        query type selection heuristic is used.
    fill_with_ones : bool, defaults to False
        What it says.
    p_critique : float or None, defaults to None
        WRITEME
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
    rng = check_random_state(rng)

    targets = problem.enumerate_features(targets)

    w = np.zeros(problem.num_features, dtype=np.float32)
    if fill_with_ones:
        w[targets] = np.ones(len(targets))

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

    critique_iters = set()
    if num_critiques is not None:
        critique_iters = set(rng.permutation(max_iters)[:num_critiques])

    trace, dataset, deltas = [], [], []
    for it in range(max_iters):
        t0 = time()
        w = np.array(w)
        x = problem.infer(w, targets)
        t0 = time() - t0

        loss = problem.utility_loss(x, "all")
        x_bar = problem.query_improvement(x, "all")

        t1 = time()
        if x_bar == "satisfied":
            trace.append((loss, t0, False))
            print("user is satisfied after {} iterations!".format(it))
            break

        d = delta((x_bar, x), targets)
        ask_critique = False
        if can_critique:
            if p_critique is None:
                ask_critique = not is_separable(deltas, d)
            else:
                ask_critique = rng.binomial(1, p_critique)
        t1 = time() - t1

        rho = None
        if ask_critique:
            rho, *_ = problem.query_critique(x, x_bar, targets)
            ask_critique = rho is not None
            assert rho is None or rho > 0

        if debug:
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

                rho = {rho}
                """).format(**locals()))

        t2 = time()
        dataset.append((x_bar, x))
        if rho is not None:
            targets.append(rho)
            w[rho] = 1.0 if fill_with_ones else 0.0
            deltas = delta(dataset, targets)
        else:
            deltas.append(d)
        w += d
        t2 = time() - t2

        if debug:
            num_targets = len(targets)
            print(dedent("""\
                new w =
                {w}

                features = {targets}
                |features| = {num_targets}
                """).format(**locals()))
        print('\nt0={}\tt1={}\tt2={}\n'.format(t0, t1, t2))
        trace.append((loss, t0 + t1 + t2, ask_critique))
    else:
        print("user not satisfied, iterations elapsed")

    return it, trace
