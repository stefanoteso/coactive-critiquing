# -*- encoding: utf-8 -*-

import numpy as np
from time import time
from textwrap import dedent

# FIXME the 'perturbed' pp algorithm is preferred for noisy users.

# NOTE the user must be able to answer "no change", or alpha-informativity
# breaks and convergence can not occur.

# NOTE utility convergences, weights may not (especially when features are
# discrete, and so the updates are discrete as well.)

# NOTE different configurations may have the same utility, so the termination
# condition is looser than strictly required

# TODO L1 SVM variant

def _print_initial_state(problem, features):
    w_star, x_star = problem.w_star, problem.x_star
    phi_star = problem.phi(x_star, "all")
    u_star = problem.utility(x_star, "all")
    print(dedent("""\
        INITIAL STATE
        =============

        w*      = {w_star}

        x*      = {x_star}
        phi*    = {phi_star}
        gu*     = {u_star}
        """).format(**locals()))

def _print_iter_state(problem, w, x, x_bar, features):
    phi = problem.phi(x, features)
    phi_bar = problem.phi(x_bar, features)
    u = problem.utility(x, features)
    u_bar = problem.utility(x_bar, features)
    print(dedent("""\
        ITERATION
        =========

        features = {features}
        w       = {w}

        x       = {x}
        phi     = {phi}
        u       = {u}

        x_bar   = {x_bar}
        phi_bar = {phi_bar}
        u_bar   = {u_bar}
        """).format(**locals()))

def _rescale(w):
    return w / np.sum(w)

def pp(problem, max_iters, features, update="perceptron",
       debug=False):
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
    debug : bool, defaults to False
        Whether to spew debug output.

    Returns
    -------
    w : numpy.ndarray of shape (num_features,)
        The learned weights.
    trace : list of numpy.ndarray of shape (num_features,)
        List of (w, x, loss, time) pairs for all iterations.

    References
    ----------
    .. [1] Shivaswamy and Joachims, *Coactive Learning*, JAIR 53 (2015)
    """
    if debug:
        _print_initial_state(problem, features)

    if update == "perceptron":
        update = lambda w, x, x_bar, features: \
            w + problem.phi(x_bar, features) - problem.phi(x, features)
    elif update == "exp-perceptron":

        eta = 1.0 / (problem.get_feature_radius() * np.sqrt(max_iters))
        update = lambda w, x, x_bar, features: \
            w * _rescale(np.exp(eta * (problem.phi(x_bar, features) - problem.phi(x, features))))
    else:
        raise ValueError("invalid update")

    num_features = len(problem.enumerate_features(features))
    w = np.ones(num_features) / np.sqrt(num_features)

    trace = []
    for it in range(max_iters):
        t1 = time()
        x = problem.infer(w, features)
        t1 = time() - t1

        x_bar = problem.query_improvement(x, features)
        is_satisfied = (x == x_bar).all()

        if debug:
            _print_iter_state(problem, w, x, x_bar, features)

        t2 = time()
        w = update(w, x, x_bar, features)
        t2 = time() - t2

        t = t1 + t2

        local_x_star = problem.compute_best_configuration(features)
        local_local_loss = problem.utility(local_x_star, features) - \
                           problem.utility(x, features)
        local_loss = problem.utility_loss(x, features)
        global_loss = problem.utility_loss(x, "all")
        print("{it:3d} | llloss={local_local_loss} lloss={local_loss} gloss={global_loss} |phi|={num_features}  {t}s".format(**locals()))

        trace.append((w, x, global_loss, t))

        if is_satisfied:
            print("user is satisfied!")
            break
    else:
        print("user not satisfied, max iters reached!")

    return w, trace

def critique_pp(problem, max_iters, debug=False):
    """The Critiquing Preference Perceptron.

    Termination occurs when WRITEME

    Parameters
    ----------
    problem : Problem
        The target problem.
    max_iters : positive int
        Number of iterations.
    debug : bool, defaults to False
        Whether to spew debug output.

    Returns
    -------
    w : numpy.ndarray of shape (num_features,)
        The learned weights.
    trace : list of numpy.ndarray of shape (num_features,)
        List of (w, x, loss, time) pairs for all iterations.
    """
    features = list(range(problem.num_base_features))
    num_features = len(features)
    r = problem.get_feature_radius()

    if debug:
        print(dedent("""\
            PP: initialization

            w*      = {}
            x*      = {}
            phi(x*) = {}
            u(x*)   = {}
        """).format(problem.w_star, problem.x_star,
                    problem.phi(problem.x_star, features),
                    problem.utility(problem.x_star, features)))

    w = np.ones(num_features) / np.sqrt(num_features)
    x = problem.infer(w, features)

    trace = []
    for it in range(max_iters):
        t = time()
        rho = problem.query_critique(x, features)
        if rho is None:
            x_bar = problem.query_improvement(x, features)

            if debug:
                print(dedent("""\
                    PP: inference & improvement

                    w          = {}

                    x          = {}
                    phi(x)     = {}
                    u(x)       = {}

                    x_bar      = {}
                    phi(x_bar) = {}
                    u(x_bar)   = {}
                """).format(w, x, problem.phi(x, features),
                            problem.utility(x, features),
                            x_bar, problem.phi(x_bar, features),
                            problem.utility(x_bar, features)))

            delta = problem.phi(x_bar, features) - \
                    problem.phi(x, features)
            w += delta
            is_satisfied = (x == x_bar).all()
        else:
            sign, rho = np.sign(rho), int(np.abs(rho))
            w = np.concatenate((w, [-sign * r]))
            features.append(rho)
            is_satisfied = False
        t = time() - t

        num_features = len(features)
        local_loss = problem.utility_loss(x, features)
        global_loss = problem.utility_loss(x, "all")
        print("{it:3d} | lloss={local_loss} gloss={global_loss} |phi|={num_features}  {t}s".format(**locals()))

        x = problem.infer(w, features)
        trace.append((w, x, global_loss, t))

        if is_satisfied:
            print("user is satisfied!")
            break
    else:
        print("user not satisfied, max iters reached!")

    return w, trace
