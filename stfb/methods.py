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
        return 1.0

    def update(self, delta):
        self.w += delta

class ExpPerceptron(object):
    """Implementation of the sparse (exponentiated) perceptron [KW97]_.

    References
    ----------
    .. [KW97] Kivinen and Warmuth, *Exponentiated Gradient Versus Gradient
        Descent for Linear Predictors*, 1997, Information and Computation
    """
    def __init__(self, problem, features, **kwargs):
        max_iters = kwargs.pop("max_iters", 100)
        debug = kwargs.pop("debug", False)

        targets = problem.enumerate_features(features)

        self.w = np.zeros(problem.num_features, dtype=np.float32)
        self.w[targets] = np.ones(len(targets)) / len(targets)
        self.eta = 1 / (2 * problem.get_feature_radius() * np.sqrt(max_iters))

    def default_weight(self, num_targets):
        return 1 / num_targets

    def update(self, delta):
        v = self.w * np.exp(self.eta * delta)
        assert (v >= 0).all()
        self.w = v / np.sum(v)

def _is_separable(deltas, verbose=False, rng=None):
    """Checks whether a dataset is separable using hard SVM."""
    n_examples = len(deltas)
    if n_examples < 2:
        return True
    n_features = len(deltas[0])

    w = cvx.Variable(n_features)

    norm_w = cvx.norm(w, 2)
    constraints = [cvx.sum_entries(deltas[i] * w) >= 1
                   for i in range(n_examples)]

    problem = cvx.Problem(cvx.Minimize(norm_w), constraints)
    problem.solve(verbose=verbose)
    return w.value is not None

def _is_separable_soft(deltas, verbose=False, rng=None):
    from sklearn import svm

    n_examples = len(deltas)
    if n_examples < 2:
        return True

    max_iter = 10 * n_examples**3

    X = deltas[:-1] + [-deltas[-1]]
    y = [1] * (n_examples - 1) + [-1]

    model = svm.LinearSVC(C=1e6, max_iter=max_iter, random_state=rng)
    model.fit(X, y)

    return model.n_iter_ < max_iter

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

    trace, dataset, deltas = [], [], []
    for it in range(max_iters):
        t0 = time()
        x = problem.infer(learner.w, targets)
        t0 = time() - t0

        loss = problem.utility_loss(x, "all")
        x_bar = problem.query_improvement(x, "all")

        t1 = time()
        is_satisfied = (x == x_bar).all()
        dataset.append((x_bar, x))

        delta = problem.phi(x_bar, targets) - problem.phi(x, targets)
        deltas.append(delta)

        is_separable = _is_separable_soft(deltas, rng=rng) if can_critique else False
        t1 = time() - t1

        rho, sign = None, None
        if can_critique and not is_satisfied and not is_separable:
            rho, sign = problem.query_critique(x, x_bar, targets)
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
                {delta}
                is_separable = {is_separable}

                rho = {rho}
                sign = {sign}

                is_satisfied = {is_satisfied}
                """).format(**locals()))

        t2 = time()
        if not is_satisfied:
            if rho is None:
                #assert not can_critique or (delta != 0).any(), "phi(x) and phi(x_bar) projections are identical"
                learner.update(delta)
            else:
                targets.append(rho)
                learner.w[rho] = sign * learner.default_weight(len(targets))

                new_deltas = []
                for x_bar_1, x_1 in dataset:
                    delta_1 = (problem.phi(x_bar_1, targets) -
                               problem.phi(x_1, targets))
                    new_deltas.append(delta_1)
                deltas = new_deltas

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
