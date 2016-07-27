# -*- encoding: utf-8 -*-

import numpy as np
from pymzn import minizinc
from itertools import product, combinations
from sklearn.utils import check_random_state

from . import Problem, array_to_assignment, assignment_to_array

_TEMPLATE = """\
int: N_ATTRIBUTES;
set of int: ATTRIBUTES = 1..N_ATTRIBUTES;

int: N_FEATURES;
set of int: FEATURES = 1..N_FEATURES;

array[FEATURES] of float: W;
array[FEATURES] of var bool: phi;
array[ATTRIBUTES] of bool: INPUT_X;
array[ATTRIBUTES] of var bool: x;
var float: objective;
bool: IS_IMPROVEMENT_QUERY;

{phis}

constraint objective = sum(j in FEATURES)(W[j] * (2 * bool2int(phi[j]) - 1));

constraint IS_IMPROVEMENT_QUERY ->
    sum(i in ATTRIBUTES)(bool2int(x[i] != INPUT_X[i])) <= 1;

{solve}
"""

_SATISFY = "solve satisfy;"
_MINIMIZE = "solve maximize objective;"

class RandProblem(Problem):
    """A randomly-constrained Boolean problem.

    Constraints are conjunctions of attributes.

    Parameters
    ----------
    num_attributes : positive int
        Number of base attributes.
    max_length : positive int
        Maximum length of the clauses (features).
    sparsity : float, defaults to 0.2
        Percentage of non-zero weights.
    noise : float, defaults to 0.1
        Amplitude of normal noise applied to weights during improvement.
    rng : None or int or numpy.random.RandomState, defaults to None
        The RNG.
    """
    def __init__(self, num_attributes, max_length=3, noise=0.1, sparsity=0.2,
                 rng=None):
        rng = check_random_state(rng)
        self.noise, self.rng = noise, rng

        # Enumerate the feature constraints
        self.features, deps, j = [], [], 0
        attributes = list(range(num_attributes))
        for length in range(1, max_length + 1):
            for clique in combinations(attributes, length):
                for attribute in clique:
                    deps.append((j, clique))
                clause = " /\\ ".join(["x[{}]".format(i + 1) for i in clique])
                feature = "constraint phi[{}] = ({});".format(j + 1, clause)
                self.features.append(feature)
                j += 1
        num_features = len(self.features)

        # Sample w_star
        num_nonzeros = max(1, int(np.rint(num_attributes * sparsity)))
        nonzero_attributes = \
            set(list(rng.permutation(num_attributes)[:num_nonzeros]))

        nonzero_features = []
        for j, clique in deps:
            if set(clique) & nonzero_attributes:
                nonzero_features.append(j)

        w_star = np.zeros(num_features, dtype=np.float32)
        w_star[nonzero_features] = rng.normal(0, 1, size=len(nonzero_features))

        super().__init__(num_attributes, num_attributes, num_features, w_star)

    def get_feature_radius(self):
        return 1.0

    def phi(self, x, features):
        features = self.enumerate_features(features)
        assert x.shape == (self.num_attributes,)

        PATH = "rand-constr-bool-phi.mzn"

        phis = "\n".join([self.features[j] for j in features])
        solve = _SATISFY
        with open(PATH, "wb") as fp:
            fp.write(_TEMPLATE.format(**locals()).encode("utf-8"))

        data = {
            "N_ATTRIBUTES": self.num_attributes,
            "N_FEATURES": len(features),
            "W": [0.0] * len(features),
            "x": array_to_assignment(x, bool),
            "INPUT_X": ["false"] * self.num_attributes,
            "IS_IMPROVEMENT_QUERY": "false",
        }
        assignments = minizinc(PATH, data=data, output_vars=["phi"])

        return assignment_to_array(assignments[0]["phi"])

    def infer(self, w, features):
        features = self.enumerate_features(features)
        assert w.shape == (len(features),)

        if (w == 0).all():
            print("inference with w == 0")

        PATH = "rand-constr-bool-infer.mzn"

        phis = "\n".join([self.features[j] for j in features])
        solve = _MINIMIZE
        with open(PATH, "wb") as fp:
            fp.write(_TEMPLATE.format(**locals()).encode("utf-8"))

        data = {
            "N_ATTRIBUTES": self.num_attributes,
            "N_FEATURES": len(features),
            "W": array_to_assignment(w, float),
            "INPUT_X": ["false"] * self.num_attributes, # doesn't matter
            "IS_IMPROVEMENT_QUERY": "false",
        }
        assignments = minizinc(PATH, data=data, output_vars=["x"])

        return assignment_to_array(assignments[0]["x"])

    def query_improvement(self, x, features):
        features = self.enumerate_features(features)
        assert x.shape == (self.num_attributes,)

        w_star = self.w_star[features]
        if (w_star == 0).all():
            print("improvement query with w == 0")
        if self.noise > 0:
            w_star += self.rng.normal(0, self.noise, size=w_star.shape[0]).astype(np.float32)

        PATH = "rand-constr-bool-improve.mzn"

        phis = "\n".join([self.features[j] for j in features])
        solve = _MINIMIZE
        with open(PATH, "wb") as fp:
            fp.write(_TEMPLATE.format(**locals()).encode("utf-8"))

        data = {
            "N_ATTRIBUTES": self.num_attributes,
            "N_FEATURES": len(features),
            "W": array_to_assignment(w_star, float),
            "INPUT_X": array_to_assignment(x, bool),
            "IS_IMPROVEMENT_QUERY": "true",
        }
        assignments = minizinc(PATH, data=data, output_vars=["x"])

        return assignment_to_array(assignments[0]["x"])

    def query_critique(self, x, features):
        features = self.enumerate_features(features)
        assert x.shape == (self.num_attributes,)

        w_star = self.w_star
        if (w_star == 0).all():
            print("critique query with w == 0")

        scores = w_star * self.phi(x, "all")
        scores[features] = np.nan
        rho = np.nanargmin(scores)
        sign = np.sign(scores[rho])

        x_bar = self.query_improvement(x, features)
        u       = self.utility(x, "all")
        u_bar   = self.utility(x_bar, "all")
        u_star  = self.utility(self.x_star, "all")

        if (u_bar - u) >= 0.1 * (u_star - u):
            return None

        return sign * rho
