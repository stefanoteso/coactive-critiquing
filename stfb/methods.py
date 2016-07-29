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

def pp(problem, max_iters, targets="attributes"):
    """The Preference Perceptron [1]_.

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
    features : str or list of int
        List of feature indices to be used in the computations. "all" means all
        features (including latent ones), "attributes" means only
        attribute-level features.

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

    trace = []
    for it in range(max_iters):
        x = problem.infer(w, targets)
        x_bar = problem.query_improvement(x, "all")
        is_satisfied = (x == x_bar).all()
        w += problem.phi(x_bar, targets) - problem.phi(x, targets)

        lloss = problem.utility_loss(x, targets)
        gloss = problem.utility_loss(x, "all")

        print(dedent("""\
            {it:3d} : lloss={lloss} gloss={gloss} |phi|={num_targets}
                x     = {x}
                x_bar = {x_bar}
                w     = {w}
            """.format(**locals())))

        trace.append((gloss, -1.0))
        if is_satisfied:
            print("user is satisfied!")
            break

    return trace

def cpp(problem, max_iters):
    """The Critiquing Preference Perceptron.

    Termination occurs when WRITEME

    Parameters
    ----------
    problem : Problem
        The target problem.
    max_iters : positive int
        Number of iterations.

    Returns
    -------
    trace : list of numpy.ndarray of shape (num_features,)
        List of (loss, time) pairs for all iterations.
    """
    targets = problem.enumerate_features("attributes")
    num_targets = len(targets)

    w = np.zeros(problem.num_features, dtype=np.float32)
    w[targets] = np.ones(num_targets) / np.sqrt(num_targets)

    trace = []
    for it in range(max_iters):
        x = problem.infer(w, targets)
        rho, sign = problem.query_critique(x, targets)
        if rho is None:
            x_bar = problem.query_improvement(x, "all")
            w += problem.phi(x_bar, targets) - problem.phi(x, targets)
            is_satisfied = (x == x_bar).all()
        else:
            w[rho] = -sign * problem.get_feature_radius()
            targets.append(rho)
            is_satisfied = False

        num_targets = len(targets)
        lloss = problem.utility_loss(x, targets)
        gloss = problem.utility_loss(x, "all")
        print("{it:3d} | lloss={lloss} gloss={gloss} |phi|={num_targets}" \
                  .format(**locals()))

        trace.append((gloss, -1.0))
        if is_satisfied:
            print("user is satisfied!")
            break

    return trace
