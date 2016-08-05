# -*- encoding: utf-8 -*-

import numpy as np
from pymzn import minizinc
from itertools import product, combinations
from sklearn.utils import check_random_state

from . import Problem, sdepnormal

_TEMPLATE = """\
int: N_ATTRIBUTES = 2;
set of int: ATTRIBUTES = 1..N_ATTRIBUTES;

int: N_FEATURES;
set of int: FEATURES = 1..N_FEATURES;

set of int: ACTIVE_FEATURES;

array[FEATURES] of float: W;
array[FEATURES] of var int: phi;
array[ATTRIBUTES] of var -1.0..1.0: x;
array[ATTRIBUTES] of -1.0..1.0: INPUT_X;
float: INPUT_UTILITY;

{phis}

{solve}
"""

_PHI = "solve satisfy;"

_INFER = """\
var float: objective =
    sum(j in ACTIVE_FEATURES)(W[j] * phi[j]);

solve ::
    float_search(x, 0.001, first_fail, indomain_middle, complete)
    maximize objective;
"""

_IMPROVE = """\
var int: objective =
    sum(i in ATTRIBUTES)(x[i] != INPUT_X[i]);

constraint sum(j in ACTIVE_FEATURES)(W[j] * phi[j]) > INPUT_UTILITY;

constraint objective >= 1;

solve minimize objective;
"""

class CanvasProblem(Problem):
    def __init__(self, num_features=4, noise=0.1, sparsity=0.2, rng=None):
        rng = check_random_state(rng)
        self.noise, self.rng = noise, rng

        num_attributes = 2

        self.features, cliques = [], []
        for j in range(num_features):
            x1_intercept, x2_intercept = rng.uniform(0.5, 0.75, size=2)
            if (j % 4) in (2, 3):
                x1_intercept = -x1_intercept
            if (j % 4) in (1, 2):
                x2_intercept = -x2_intercept
            coeff = [-1 / x1_intercept, -1 / x2_intercept]
            inequality = "{} * x[1] + {} * x[2] + 1 >= 0".format(*coeff)
            feature = "constraint phi[{}] = 2 * ({}) - 1;".format(j + 1, inequality)
            self.features.append(feature)
            cliques.append([0, 1])
        num_features = len(self.features)

        # XXX arbitrary
        num_base_features = 2

        global _TEMPLATE
        _TEMPLATE = \
            _TEMPLATE.format(phis="\n".join(self.features), solve="{solve}")

        w_star = sdepnormal(num_attributes, num_features, cliques,
                            sparsity=sparsity, rng=rng).astype(np.float32)
        w_star = np.abs(w_star)

        super().__init__(num_attributes, num_attributes, num_features, w_star)

    def get_feature_radius(self):
        return 1.0

    def phi(self, x, features):
        assert x.shape == (self.num_attributes,)

        targets = self.enumerate_features(features)

        PATH = "canvas-phi.mzn"
        with open(PATH, "wb") as fp:
            fp.write(_TEMPLATE.format(solve=_PHI).encode("utf-8"))

        data = {
            "N_FEATURES": self.num_features,
            "ACTIVE_FEATURES": set([1]), # doesn't matter
            "W": [0.0] * self.num_features, # doesn't matter
            "x": self.array_to_assignment(x, bool),
            "INPUT_X": [0.0] * self.num_attributes, # doesn't matter
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
        #assert (w[targets] != 0).any()

        PATH = "canvas-infer.mzn"
        with open(PATH, "wb") as fp:
            fp.write(_TEMPLATE.format(solve=_INFER).encode("utf-8"))

        data = {
            "N_FEATURES": self.num_features,
            "ACTIVE_FEATURES": {j + 1 for j in targets},
            "W": self.array_to_assignment(w, float),
            "INPUT_X": [0.0] * self.num_attributes, # doesn't matter
            "INPUT_UTILITY": 0.0, # doesn't matter
        }
        assignments = minizinc(PATH, data=data, output_vars=["x", "objective"], keep=True)

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

        PATH = "canvas-improve.mzn"
        with open(PATH, "wb") as fp:
            fp.write(_TEMPLATE.format(solve=_IMPROVE).encode("utf-8"))

        data = {
            "N_FEATURES": self.num_features,
            "ACTIVE_FEATURES": {j + 1 for j in targets},
            "W": self.array_to_assignment(w_star, float),
            "INPUT_X": self.array_to_assignment(x, float),
            "INPUT_UTILITY": utility,
        }
        assignments = minizinc(PATH, data=data, output_vars=["x", "objective"], parallel=0)

        return self.assignment_to_array(assignments[0]["x"])
