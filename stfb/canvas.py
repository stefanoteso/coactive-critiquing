# -*- encoding: utf-8 -*-

import numpy as np
import pickle
from pymzn import minizinc
from itertools import product, combinations
from sklearn.utils import check_random_state

from . import Problem

_TEMPLATE = """\
int: N_ATTRIBUTES = 2;
set of int: ATTRIBUTES = 1..N_ATTRIBUTES;

int: N_FEATURES;
set of int: FEATURES = 1..N_FEATURES;

set of int: ACTIVE_FEATURES;
set of int: TRUTH_VALUES;

array[FEATURES] of int: W;
array[FEATURES] of var TRUTH_VALUES: phi;
array[FEATURES] of var TRUTH_VALUES: INPUT_PHI;
array[ATTRIBUTES] of var 1..{canvas_size}: x;
array[ATTRIBUTES] of 1..{canvas_size}: INPUT_X;

{phis}

{solve}
"""

_PHI = "solve satisfy;"

_INFER = """\
var int: objective =
    sum(j in ACTIVE_FEATURES)(W[j] * phi[j]);

solve maximize objective;
"""

_IMPROVE = """\
var int: objective =
    sum(i in ATTRIBUTES)(x[i] != INPUT_X[i]);

constraint
    sum(j in ACTIVE_FEATURES)(W[j] * phi[j]) >
        sum(j in ACTIVE_FEATURES)(W[j] * INPUT_PHI[j]);

constraint objective >= 1;

solve minimize objective;
"""

class CanvasProblem(Problem):
    def __init__(self, num_features=100, noise=0.1, sparsity=0.2, rng=None):
        rng = check_random_state(rng)
        self.noise, self.rng = noise, rng

        with open("datasets/canvas.pickle", "rb") as fp:
            dataset = pickle.load(fp)
            canvas_size = dataset["canvas_size"]
            rectangles = dataset["rectangles"]

        self.features = []
        for j, (xmin, xmax, ymin, ymax) in enumerate(rectangles):
            is_inside = "x[1] >= {xmin} /\\ x[1] <= {xmax} /\\ x[2] >= {ymin} /\\ x[2] <= {ymax}".format(**locals())
            feature = "constraint phi[{}] = 2 * ({}) - 1;".format(j + 1, is_inside)
            self.features.append(feature)
        num_features = len(self.features)

        global _TEMPLATE
        _TEMPLATE = \
            _TEMPLATE.format(canvas_size=canvas_size,
                             phis="\n".join(self.features), solve="{solve}")

        w_star = 2 * rng.randint(0, 2, size=num_features) - 1
        if sparsity < 1.0:
            nnz_features = max(1, int(np.ceil(sparsity * num_features)))
            zeros = rng.permutation(num_features)[nnz_features:]
            w_star[zeros] = 0

        super().__init__(2, 2, num_features, w_star)

    def get_feature_radius(self):
        return 1.0

    def phi(self, x, features):
        PATH = "canvas-phi.mzn"

        with open(PATH, "wb") as fp:
            fp.write(_TEMPLATE.format(solve=_PHI).encode("utf-8"))

        data = {
            "N_FEATURES": self.num_features,
            "TRUTH_VALUES": {-1, 1},
            "ACTIVE_FEATURES": set([1]), # doesn't matter
            "W": [0] * self.num_features, # doesn't matter
            "x": self.array_to_assignment(x, int),
            "INPUT_X": [1] * self.num_attributes, # doesn't matter
            "INPUT_PHI": [1] * self.num_features, # doesn't matter
        }

        return super().phi(x, features, PATH, data)

    def infer(self, w, features):
        PATH = "canvas-infer.mzn"

        with open(PATH, "wb") as fp:
            fp.write(_TEMPLATE.format(solve=_INFER).encode("utf-8"))

        targets = self.enumerate_features(features)
        data = {
            "N_FEATURES": self.num_features,
            "TRUTH_VALUES": {-1, 1},
            "ACTIVE_FEATURES": {j + 1 for j in targets},
            "W": self.array_to_assignment(w, int),
            "INPUT_X": [1] * self.num_attributes, # doesn't matter
            "INPUT_PHI": [1] * self.num_features, # doesn't matter
        }

        return super().infer(w, features, PATH, data)

    def query_improvement(self, x, features):
        w_star = np.array(self.w_star)
        if self.noise:
            raise NotImplementedError()

        PATH = "canvas-improve.mzn"

        with open(PATH, "wb") as fp:
            fp.write(_TEMPLATE.format(solve=_IMPROVE).encode("utf-8"))

        targets = self.enumerate_features(features)
        phi = self.phi(x, "all") # XXX the sum is over ACTIVE_FEATURES anyway
        data = {
            "N_FEATURES": self.num_features,
            "TRUTH_VALUES": {-1, 1},
            "ACTIVE_FEATURES": {j + 1 for j in targets},
            "W": self.array_to_assignment(w_star, int),
            "INPUT_X": self.array_to_assignment(x, int),
            "INPUT_PHI": self.array_to_assignment(phi, int),
        }

        return super().query_improvement(x, w_star, features, PATH, data)
