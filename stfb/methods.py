# -*- encoding: utf-8 -*-

import numpy as np
from textwrap import dedent
from time import time
from scipy.spatial import Delaunay

# NOTE the user must be able to answer "no change", or alpha-informativity
# breaks and convergence can not occur.

# NOTE utility convergences, weights may not (especially when features are
# discrete, and so the updates are discrete as well.)

# NOTE different configurations may have the same utility, so the termination
# condition is looser than strictly required

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
    num_iters : int
        Number of iterations elapsed
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

    dataset = []
    triangulation = None
    trace = []
    for it in range(max_iters):
        t0 = time()
        x = problem.infer(w, targets)
        t0 = time() - t0

        loss = problem.utility_loss(x, "all")
        x_bar = problem.query_improvement(x, "all")

        t1 = time()
        is_satisfied = (x == x_bar).all()
        delta = problem.phi(x_bar, targets) - problem.phi(x, targets)
        t1 = time() - t1

        rho, sign = None, None
        if not triangulation:
            triangulation = Delaunay(np.array([delta]), incremental=True)
        elif can_critique and triangulation.find_simplex(-delta) >= 0:
            # The union of all the simplices of the Delaunay triangulation
            # determines the convex hull of the dataset.
            # If -delta (point of the "negative class") is in the convex hull
            # (determined by finding if a simplex in the Delaunay
            # triangulation contains the point) then the the dataset is
            # not linearly separable w.r.t. the current phi.
            rho, sign = problem.query_critique(x, x_bar, targets)
            assert rho > 0

        dataset.append((x_bar, x))
        triangulation.add_points(np.array([delta]))

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
            loss = {loss}

            rho = {rho}
            sign = {sign}

            is_satisfied = {is_satisfied}
            """).format(**locals()))

        t2 = time()
        if not is_satisfied:
            if rho is None:
                assert not can_critique or (delta != 0).any(), "phi(x) and phi(x_bar) projections are identical"
                w += delta
            else:
                w[rho] = sign * problem.get_feature_radius()
                targets.append(rho)

                # Recompute the triangulation w.r.t. the new phi
                deltas = []
                for x_bar_1, x_1 in dataset:
                    delta_1 = (problem.phi(x_bar_1, targets) -
                               problem.phi(x_1, targets))
                    deltas.append(delta_1)
                triangulation = Delaunay(np.array(deltas), incremental=True)

        t2 = time() - t2

        num_targets = len(targets)

        print(dedent("""\
            x_bar =
            {x_bar}
            phi(x_bar) =
            {phi_bar}

            phi(x_bar) - phi(x) =
            {delta}

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
