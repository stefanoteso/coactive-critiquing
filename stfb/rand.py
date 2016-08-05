# -*- encoding: utf-8 -*-

import numpy as np
from pymzn import minizinc
from itertools import product, combinations
from sklearn.utils import check_random_state

from . import Problem, sdepnormal

_TEMPLATE = """\
int: N_ATTRIBUTES;
set of int: ATTRIBUTES = 1..N_ATTRIBUTES;

int: N_FEATURES;
set of int: FEATURES = 1..N_FEATURES;

set of int: ACTIVE_FEATURES;

array[FEATURES] of float: W;
array[FEATURES] of var int: phi;
array[ATTRIBUTES] of var bool: x;
array[ATTRIBUTES] of bool: INPUT_X;
float: INPUT_UTILITY;

{phis}

{solve}
"""

_PHI = "solve satisfy;"

_INFER = """\
var float: objective;

constraint objective =
    sum(j in ACTIVE_FEATURES)(W[j] * phi[j]);

solve maximize objective;
"""

_IMPROVE = """\
var int: objective;

constraint objective =
    sum(i in ATTRIBUTES)(x[i] != INPUT_X[i]);

constraint sum(j in ACTIVE_FEATURES)(W[j] * phi[j]) > INPUT_UTILITY;

constraint objective >= 1;

solve minimize objective;
"""

class RandProblem(Problem):
    """A randomly-constrained Boolean problem.

    Constraints are exclusive-ORs of attributes.

    Parameters
    ----------
    num_attributes : positive int
        Number of base attributes.
    max_length : positive int, defaults to 2
        Maximum length of the clauses (features). Note that certain values
        (e.g. 3) render the problem fully independent of the features.
    sparsity : float, defaults to 0.2
        Percentage of non-zero weights.
    noise : float, defaults to 0.1
        Amplitude of normal noise applied to weights during improvement.
    rng : None or int or numpy.random.RandomState, defaults to None
        The RNG.
    """
    def __init__(self, num_attributes, max_length=2, noise=0.1, sparsity=0.2,
                 rng=None):
        rng = check_random_state(rng)
        self.noise, self.rng = noise, rng

        attributes = list(range(num_attributes))

        self.features, cliques, j = [], [], 0
        for length in range(1, max_length + 1):
            for clique in combinations(attributes, length):
                xor = " xor ".join(["x[{}]".format(i + 1) for i in clique])
                feature = "constraint phi[{}] = (2 * ({}) - 1);".format(j + 1, xor)
                self.features.append(feature)
                cliques.append(clique)
                j += 1
        num_features = len(self.features)

        global _TEMPLATE
        _TEMPLATE = \
            _TEMPLATE.format(phis="\n".join(self.features), solve="{solve}")

        w_star = sdepnormal(num_attributes, num_features, cliques,
                            sparsity=sparsity, rng=rng).astype(np.float32)

        super().__init__(num_attributes, num_attributes, num_features, w_star)

    def get_feature_radius(self):
        return 1.0

    def phi(self, x, features):
        assert x.shape == (self.num_attributes,)

        targets = self.enumerate_features(features)

        PATH = "rand-phi.mzn"
        with open(PATH, "wb") as fp:
            fp.write(_TEMPLATE.format(solve=_PHI).encode("utf-8"))

        data = {
            "N_ATTRIBUTES": self.num_attributes,
            "N_FEATURES": self.num_features,
            "ACTIVE_FEATURES": set([1]), # doesn't matter
            "W": [0.0] * self.num_features, # doesn't matter
            "x": self.array_to_assignment(x, bool),
            "INPUT_X": ["false"] * self.num_attributes, # doesn't matter
            "INPUT_UTILITY": 0.0, # doesn't matter
        }
        assignments = minizinc(PATH, data=data, output_vars=["phi"])

        phi = self.assignment_to_array(assignments[0]["phi"])
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
            fp.write(_TEMPLATE.format(solve=_INFER).encode("utf-8"))

        data = {
            "N_ATTRIBUTES": self.num_attributes,
            "N_FEATURES": self.num_features,
            "ACTIVE_FEATURES": {j + 1 for j in targets},
            "W": self.array_to_assignment(w, float),
            "INPUT_X": ["false"] * self.num_attributes, # doesn't matter
            "INPUT_UTILITY": 0.0, # doesn't matter
        }
        assignments = minizinc(PATH, data=data, output_vars=["x", "objective"], parallel=0)

        return self.assignment_to_array(assignments[0]["x"])

    def query_improvement(self, x, features):
        assert x.shape == (self.num_attributes,)

        if self.utility_loss(x, "all") == 0:
            # XXX this is noiseless
            return x

        w_star = np.array(self.w_star)
        if self.noise:
            w_star += self.rng.normal(0, self.noise, size=w_star.shape).astype(np.float32)

        targets = self.enumerate_features(features)
        assert (w_star[targets] != 0).any()

        utility = np.dot(w_star, self.phi(x, "all"))

        PATH = "rand-improve.mzn"
        with open(PATH, "wb") as fp:
            fp.write(_TEMPLATE.format(solve=_IMPROVE).encode("utf-8"))

        data = {
            "N_ATTRIBUTES": self.num_attributes,
            "N_FEATURES": self.num_features,
            "ACTIVE_FEATURES": {j + 1 for j in targets},
            "W": self.array_to_assignment(w_star, float),
            "INPUT_X": self.array_to_assignment(x, bool),
            "INPUT_UTILITY": utility,
        }
        assignments = minizinc(PATH, data=data, output_vars=["x", "objective"], parallel=0)

        x_bar = self.assignment_to_array(assignments[0]["x"])
        utility_bar = np.dot(w_star, self.phi(x, "all"))

        return x_bar
