# -*- encoding: utf-8 -*-

import numpy as np
from textwrap import dedent
from time import time

# NOTE the user must be able to answer "no change", or alpha-informativity
# breaks and convergence can not occur.

# NOTE utility convergences, weights may not (especially when features are
# discrete, and so the updates are discrete as well.)

# NOTE different configurations may have the same utility, so the termination
# condition is looser than strictly required

# TODO exponentiated update for sparse weights with clamping

# TODO L1 SVM variant

# TODO the 'perturbed' pp algorithm is preferred for noisy users.

def pp(problem, max_iters, targets="attributes", can_critique=False):
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
    targets : str or list of int, defaults to "attributes"
        Indices or description of features describing the configuration space.
        "attributes" means only attribute-level features, "all" means all
        possible features. The space may change when can_critique is True.
    can_critique : bool, defaults to False
        Whether critique queries are enabled.

    Returns
    -------
    trace : list of numpy.ndarray of shape (num_features,)
        List of (loss, time) pairs for all iterations.

    References
    ----------
    .. [1] Shivaswamy and Joachims, *Coactive Learning*, JAIR 53 (2015)
    """
    targets = problem.enumerate_features(targets)
    num_targets = len(targets)

    w = np.zeros(problem.num_features, dtype=np.float32)
    w[targets] = np.ones(num_targets)

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

    trace = []
    for it in range(max_iters):
        t0 = time()
        x = problem.infer(w, targets)
        t0 = time() - t0

        x_bar = problem.query_improvement(x, "all")
        is_satisfied = (x == x_bar).all()

        if not can_critique:
            rho, sign = None, None
        else:
            rho, sign = problem.query_critique(x, x_bar, targets)

        phi = problem.phi(x, "all")
        phi_bar = problem.phi(x_bar, "all")
        print(dedent("""\
            == ITERATION {it:3d} ==

            w =
            {w}

            x =
            {x}
            phi(x) =
            {phi}

            rho = {rho}
            sign = {sign}
            """).format(**locals()))

        t1 = time()
        if rho is None:
            w += problem.phi(x_bar, targets) - problem.phi(x, targets)
        else:
            w[rho] = sign * problem.get_feature_radius()
            targets.append(rho)
            is_satisfied = False
        t1 = time() - t1

        num_targets = len(targets)
        loss = problem.utility_loss(x, "all")

        print(dedent("""\
            x_bar =
            {x_bar}
            phi(x_bar) =
            {phi_bar}

            new w =
            {w}

            features = {targets}
            |features| = {num_targets}

            loss = {loss}
            is_satisfied = {is_satisfied}
            """).format(**locals()))

        trace.append((loss, t0 + t1))
        if is_satisfied:
            if loss > 0:
                print("user is not satisfied, but can not improve item!")
            else:
                print("user is satisfied!")
            break
    else:
        print("user not satisfied, iterations elapsed")

    return trace
