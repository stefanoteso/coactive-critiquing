# -*- encoding: utf-8 -*-

import numpy as np
from pymzn import minizinc
from itertools import product, combinations
from sklearn.utils import check_random_state

from . import Problem, array_to_assignment, assignment_to_array, sdepnormal

_TEMPLATE = """\
int: N_ATTRIBUTES;
set of int: ATTRIBUTES = 1..N_ATTRIBUTES;

int: N_FEATURES;
set of int: FEATURES = 1..N_FEATURES;

set of int: ACTIVE_FEATURES;

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
_MAXIMIZE = "solve maximize objective;"

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

        attributes = list(range(num_attributes))

        self.features, deps, j = [], [], 0
        for length in range(1, max_length + 1):
            for clique in combinations(attributes, length):
                conjunction = " /\\ ".join(["x[{}]".format(i + 1) for i in clique])
                feature = "constraint phi[{}] = ({});".format(j + 1, conjunction)
                self.features.append(feature)
                deps.append((j, clique))
                j += 1
        num_features = len(self.features)

        global _TEMPLATE
        _TEMPLATE = \
            _TEMPLATE.format(phis="\n".join(self.features), solve="{solve}")

        w_star = sdepnormal(num_attributes, num_features, deps,
                            sparsity=sparsity, rng=rng, dtype=np.float32)

        super().__init__(num_attributes, num_attributes, num_features, w_star)

    def get_feature_radius(self):
        return 1.0

    def phi(self, x, features):
        assert x.shape == (self.num_attributes,)

        targets = self.enumerate_features(features)

        PATH = "rand-phi.mzn"
        with open(PATH, "wb") as fp:
            fp.write(_TEMPLATE.format(solve=_SATISFY).encode("utf-8"))

        data = {
            "N_ATTRIBUTES": self.num_attributes,
            "N_FEATURES": self.num_features,
            "ACTIVE_FEATURES": set([0]),
            "W": [0.0] * self.num_features,
            "x": array_to_assignment(x, bool),
            "INPUT_X": ["false"] * self.num_attributes,
            "IS_IMPROVEMENT_QUERY": "false",
        }
        assignments = minizinc(PATH, data=data, output_vars=["phi", "objective"])

        phi = assignment_to_array(assignments[0]["phi"])
        mask = np.ones_like(phi, dtype=bool)
        mask[targets] = False
        phi[mask] = 0.0

        return phi

    def infer(self, w, features):
        assert w.shape == (self.num_features,)

        targets = self.enumerate_features(features)
        assert (w[targets] != 0).any()

        PATH = "rand-infer.mzn"
        with open(PATH, "wb") as fp:
            fp.write(_TEMPLATE.format(solve=_MAXIMIZE).encode("utf-8"))

        data = {
            "N_ATTRIBUTES": self.num_attributes,
            "N_FEATURES": self.num_features,
            "ACTIVE_FEATURES": set(targets),
            "W": array_to_assignment(w, float),
            "INPUT_X": ["false"] * self.num_attributes, # doesn't matter
            "IS_IMPROVEMENT_QUERY": "false",
        }
        assignments = minizinc(PATH, data=data, output_vars=["x", "objective"])

        return assignment_to_array(assignments[0]["x"])

    def query_improvement(self, x, features):
        assert x.shape == (self.num_attributes,)

        targets = self.enumerate_features(features)
        assert (self.w_star[targets] != 0).any()

        PATH = "rand-improve.mzn"
        with open(PATH, "wb") as fp:
            fp.write(_TEMPLATE.format(solve=_MAXIMIZE).encode("utf-8"))

        w_star = np.array(self.w_star)
        if self.noise:
            w_star += self.rng.normal(0, self.noise, size=w_star.shape).astype(np.float32)

        data = {
            "N_ATTRIBUTES": self.num_attributes,
            "N_FEATURES": self.num_features,
            "ACTIVE_FEATURES": set(targets),
            "W": array_to_assignment(w_star, float),
            "INPUT_X": array_to_assignment(x, bool),
            "IS_IMPROVEMENT_QUERY": "true",
        }
        assignments = minizinc(PATH, data=data, output_vars=["x", "objective"])

        return assignment_to_array(assignments[0]["x"])
