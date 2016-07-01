# -*- encoding: utf-8 -*-

import numpy as np
from time import time
from textwrap import dedent

# FIXME the 'perturbed' pp algorithm is preferred for noisy users.

# NOTE the user must be able to answer "no change", or alpha-informativity
# breaks and convergence can not occur.

# NOTE utility convergences, weights may not (especially when features are
# discrete, and so the updates are discrete as well.)

# TODO L1 SVM variant

def pp(problem, max_iters, features, update="perceptron"):
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
    update : str, defaults to "perceptron"
        Type of update to perform: "perceptron" and "exponentiated" are
        supported, see [1]_ for details.

    Returns
    -------
    w : numpy.ndarray of shape (num_features,)
        The learned weights.
    trace : list of numpy.ndarray of shape (num_features,)
        List of (w, x, time) pairs for all iterations.

    References
    ----------
    .. [1] Shivaswamy and Joachims, *Coactive Learning*, JAIR 53 (2015)
    """
    num_features = len(problem.enumerate_features(features))

    w = np.ones(num_features) / np.sqrt(num_features)
    x = problem.infer(w, features)

    eta = 1.0 / (2 * problem.get_feature_radius() * np.sqrt(max_iters))

    def rescale(w):
        return w / np.sum(w)

    trace = []
    for it in range(max_iters):
        t = time()
        x_bar = problem.query_improvement(x, features)
        delta = problem.phi(x_bar, features) - \
                problem.phi(x, features)
        if update == "perceptron":
            w += delta
        elif update == "exponentiated":
            w = rescale(w * np.exp(eta * delta))
        t = time() - t

        is_satisfied = (x == x_bar).all()

        loss = problem.utility_loss(x, features)
        print("{it:3d} | loss={loss} {t}s".format(**locals()))

        x = problem.infer(w, features)
        trace.append((w, x, loss, t))

        if is_satisfied:
            print("user is satisfied!")
            break
    else:
        print("user not satisfied, max iters reached!")

    return w, trace

def critique_pp(problem, max_iters):
    """The Critiquing Preference Perceptron.

    Termination occurs when WRITEME

    Parameters
    ----------
    problem : Problem
        The target problem.
    max_iters : positive int
        Number of iterations.
    features : list of indices, defaults to "all"
        List of feature indices to be used in the computations. "all" means
        all features (including latent ones), "attributes" means only
        per-attribute identity features.

    Returns
    -------
    w : numpy.ndarray of shape (num_features,)
        The learned weights.
    trace : list of numpy.ndarray of shape (num_features,)
        List of (w, x, time) pairs for all iterations.
    """
    raise NotImplementedError()

#    features = list(range(problem.num_attributes))
#    w = np.ones(problem.num_attributes)
#    x = problem.infer(w, features=features)
#
#    trace = []
#    for it in range(max_iters):
#        t = time()
#        rho = problem.query_critique(x, features=features)
#        if rho is None:
#            x_bar = problem.query_improvement(x, features=features)
#            w += problem.phi(x_bar, features=features) - \
#                 problem.phi(x, features=features)
#            is_satisfied = (x == x_bar).all()
#        else:
#            features.append(rho)
#            w = np.concatenate((w, [problem.get_feature_radius()]))
#            is_satisfied = False
#        t = time() - t
#
#        num_curr_features = len(features)
#        num_features = problem.num_features
#        loss = problem.utility_loss(x)
#        print("{it:3d} | loss={loss} {t}s | m={num_curr_features}/{num_features}" \
#                  .format(**locals()))
#
#        x = problem.infer(w, features=features)
#        trace.append((w, x, t))
#
#        if is_satisfied:
#            print("user is satisfied!")
#            break
#    else:
#        print("user not satisfied, max iters reached!")
#
#    return w, trace
