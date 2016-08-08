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
set of int: TRUTH_VALUES;

array[FEATURES] of float: W;
array[FEATURES] of var TRUTH_VALUES: phi;
array[ATTRIBUTES] of var 1..100: x;
array[ATTRIBUTES] of 1..100: INPUT_X;
float: INPUT_UTILITY;

{phis}

{solve}
"""

_PHI = "solve satisfy;"

_INFER = """\
var float: objective =
    sum(j in ACTIVE_FEATURES)(W[j] * phi[j]);

solve maximize objective;
"""

_IMPROVE = """\
var int: objective =
    sum(i in ATTRIBUTES)(x[i] != INPUT_X[i]);

constraint sum(j in ACTIVE_FEATURES)(W[j] * phi[j]) > INPUT_UTILITY;

constraint objective >= 1;

solve minimize objective;
"""

_rects = None

class CanvasProblem(Problem):
    def __init__(self, num_features=100, noise=0.1, sparsity=0.2, rng=None):
        rng = check_random_state(rng)
        self.noise, self.rng = noise, rng

        # XXX this should be done offline
        global _rects

        if _rects is None:

            reorder = lambda a, b: (min(a, b), max(a, b))

            _rects = []
            for j in range(num_features):
                xmin, xmax = reorder(*rng.randint(1, 100+1, size=2))
                ymin, ymax = reorder(*rng.randint(1, 100+1, size=2))
                _rects.append([xmin, xmax, ymin, ymax])

            print("rects =\n{}".format("\n".join(map(str, _rects))))

        self.features, cliques = [], []
        for j, (xmin, xmax, ymin, ymax) in enumerate(_rects):
            is_inside = "x[1] >= {xmin} /\\ x[1] <= {xmax} /\\ x[2] >= {ymin} /\\ x[2] <= {ymax}".format(**locals())
            feature = "constraint phi[{}] = 2 * ({}) - 1;".format(j + 1, is_inside)
            self.features.append(feature)
            cliques.append([0, 1])

        num_attributes = 2
        num_base_features = 2 # XXX arbitrary
        num_features = len(self.features)

        global _TEMPLATE
        _TEMPLATE = \
            _TEMPLATE.format(phis="\n".join(self.features), solve="{solve}")

        w_star = sdepnormal(num_attributes, num_features, cliques,
                            sparsity=sparsity, rng=rng).astype(np.float32)
        w_star = np.abs(w_star)

        super().__init__(num_attributes, num_base_features, num_features,
                         w_star)

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
            "TRUTH_VALUES": {-1, 1},
            "ACTIVE_FEATURES": set([1]), # doesn't matter
            "W": [0.0] * self.num_features, # doesn't matter
            "x": self.array_to_assignment(x, int),
            "INPUT_X": [1] * self.num_attributes, # doesn't matter
            "INPUT_UTILITY": 0.0, # doesn't matter
        }
        assignments = minizinc(PATH, data=data, output_vars=["phi"], keep=True)

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
            "TRUTH_VALUES": {-1, 1},
            "ACTIVE_FEATURES": {j + 1 for j in targets},
            "W": self.array_to_assignment(w, float),
            "INPUT_X": [1] * self.num_attributes, # doesn't matter
            "INPUT_UTILITY": 0.0, # doesn't matter
        }
        assignments = minizinc(PATH, data=data, output_vars=["x", "objective"],
                               keep=True, parallel=0)

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

        utility = np.dot(w_star, self.phi(x, targets))

        PATH = "canvas-improve.mzn"
        with open(PATH, "wb") as fp:
            fp.write(_TEMPLATE.format(solve=_IMPROVE).encode("utf-8"))

        data = {
            "N_FEATURES": self.num_features,
            "TRUTH_VALUES": {-1, 1},
            "ACTIVE_FEATURES": {j + 1 for j in targets},
            "W": self.array_to_assignment(w_star, float),
            "INPUT_X": self.array_to_assignment(x, int),
            "INPUT_UTILITY": utility,
        }
        assignments = minizinc(PATH, data=data, output_vars=["x", "objective"],
                               keep=True, parallel=0)

        x_bar = self.assignment_to_array(assignments[0]["x"])
        assert (x != x_bar).any(), (x, x_bar)

        utility_bar = np.dot(w_star, self.phi(x_bar, targets))
        assert utility_bar > utility, (utility_bar, ">", utility)

        return x_bar
