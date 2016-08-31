# -*- encoding: utf-8 -*-

import numpy as np
import cvxpy as cvx
from sklearn.utils import check_random_state
from textwrap import dedent
from time import time

class Perceptron(object):
    """Implementation of the standard perceptron."""
    def __init__(self, problem, features, **kwargs):
        self._debug = kwargs.pop("debug", False)
        self._rng = check_random_state(kwargs.pop("rng", None))

        targets = problem.enumerate_features(features)

        self.w = np.zeros(problem.num_features, dtype=np.float32)
        if self._debug:
            self.w[targets] = np.ones(len(targets))
        else:
            # XXX for sparse users perhaps sample from a sparse distribution
            self.w[targets] = \
                rng.normal(0, 1, size=len(targets)).astype(np.float32)

    def default_weight(self, num_targets):
        return 1.0 if self._debug else self._rng.normal(0, 1)

    def update(self, delta):
        self.w += delta


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


def pp(problem, max_iters, targets, Learner=Perceptron, can_critique=False,
       num_critiques=None, perturbation=0, rng=None, debug=False, gamma=1.0):
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
    perturbation : float, defaults to 0
        Amount of w perturbation to use during inference. Used for
        implementing the *perturbed* preference perceptron.
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

    critique_iters = set()
    if num_critiques is not None:
        critique_iters = set(rng.permutation(max_iters)[:num_critiques])

    s = 0.0
    ask_critique = True
    trace, dataset, deltas = [], [], []
    num_real_improvements = 0
    for it in range(max_iters):
        t0 = time()
        w = np.array(learner.w)
        if perturbation > 0:
            w[targets] = rng.normal(0, perturbation, size=len(targets))
        x = problem.infer(w, targets)
        t0 = time() - t0

        loss = problem.utility_loss(x, "all")
        x_bar = problem.query_improvement(x, "all")

        t1 = time()
        if x_bar == "satisfied":
            trace.append((loss, t0, False))
            print("user is satisfied!")
            break

        u = problem.utility(x, "all")
        u_bar = problem.utility(x_bar, "all")
        is_improvement = u_bar > u
        if is_improvement:
            num_real_improvements += 1

        d = delta((x_bar, x), targets)

        p, ask_critique = 0.0, False
        if not can_critique:
            pass
        elif it in critique_iters:
            p, ask_critique = 1.0, True
        elif not is_separable(deltas, d):
            if not ask_critique:
                s += 1
            p = 1 # (gamma * s) / (gamma * s  + (it + 1))
            ask_critique = bool(rng.binomial(1, p))
        t1 = time() - t1

        rho = None
        if ask_critique:
            rho, _ = problem.query_critique(x, x_bar, targets)
            assert rho > 0

        if debug:
            w = learner.w
            phi = problem.phi(x, "all")
            phi_bar = problem.phi(x_bar, "all")
            perc_real_improvements = num_real_improvements / (it + 1)
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

                is_improvement = {is_improvement}
                % real improvements = {perc_real_improvements}

                phi(x_bar) - phi(x) =
                {d}
                ask_critique = {ask_critique}
                p = {p}

                rho = {rho}
                """).format(**locals()))

        t2 = time()
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
    else:
        print("user not satisfied, iterations elapsed")

    return it, trace
