# -*- encoding: utf-8 -*-

import numpy as np
from textwrap import dedent

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
    w[targets] = np.ones(num_targets) / np.sqrt(num_targets)

    w_star, x_star = problem.w_star, problem.x_star
    phi_star = problem.phi(x_star, "all")

    trace = []
    for it in range(max_iters):
        x = problem.infer(w, targets)

        if not can_critique:
            rho, sign = None, None
        else:
            rho, sign = problem.query_critique(x, targets)

        phi = problem.phi(x, "all")
        print(dedent("""\
            == ITERATION {it:3d} ==

            w_star =
            {w_star}
            w =
            {w}

            x =
            {x}
            x_star =
            {x_star}

            phi(x) =
            {phi}
            phi(x_star) =
            {phi_star}

            rho = {rho}
            sign = {sign}
            """).format(**locals()))

        if rho is None:
            x_bar = problem.query_improvement(x, targets)
            w += problem.phi(x_bar, targets) - problem.phi(x, targets)
            is_satisfied = (x == x_bar).all()
        else:
            x_bar = None
            w[rho] = -sign * problem.get_feature_radius()
            targets.append(rho)
            is_satisfied = False

        num_targets = len(targets)
        lloss = problem.utility_loss(x, targets)
        gloss = problem.utility_loss(x, "all")

        print(dedent("""\
            x_bar =
            {x_bar}

            new w =
            {w}

            features = {targets}
            |features| = {num_targets}

            lloss = {lloss}
            gloss = {gloss}
            is_satisfied = {is_satisfied}
            """).format(**locals()))

        trace.append((gloss, -1.0))
        if is_satisfied:
            print("user is satisfied!")
            break
    else:
        print("user not satisfied, iterations elapsed")

    return trace
