# -*- encoding: utf-8 -*-

import numpy as np
from pymzn import minizinc
from itertools import product, combinations
from sklearn.utils import check_random_state

from . import Problem, array_to_assignment, assignment_to_array, spnormal

_PROBLEM = """\
int: N_ATTRIBUTES;
set of int: ATTRIBUTES = 1..N_ATTRIBUTES;

array[ATTRIBUTES] of float: W;
array[ATTRIBUTES] of 0..1: INPUT_X;
array[ATTRIBUTES] of var 0..1: x;
var float: objective;
bool: IS_IMPROVEMENT_QUERY;

constraint objective = sum(i in ATTRIBUTES)(W[i] * x[i]);

constraint IS_IMPROVEMENT_QUERY ->
    (sum(i in ATTRIBUTES)(bool2int(x[i] != INPUT_X[i])) <= 3);

solve maximize objective;
"""
_PROBLEM_PATH = "unconstr-bool-infer.mzn"

class UnconstrBoolProblem(Problem):
    """An unconstrained Boolean problem with trivial feature map.

    The true weight vector is sampled from a standard normal.

    Parameters
    ----------
    num_attributes : positive int
        Number of base attributes.
    rng : None or int or numpy.random.RandomState, defaults to None
        The RNG.
    """
    def __init__(self, num_attributes, rng=None):
        rng = check_random_state(rng)

        with open(_PROBLEM_PATH, "wb") as fp:
            fp.write(_PROBLEM.encode("utf-8"))

        w_star = rng.normal(0, 1, size=num_attributes).astype(np.float32)
        super().__init__(num_attributes, num_attributes, num_attributes,
                         w_star)

    def get_feature_radius(self):
        return 1.0

    def _check_features(self, features):
        return features in ("attributes", "all") or \
               features == list(range(self.num_attributes))

    def phi(self, x, features):
        assert x.shape == (self.num_attributes,)
        assert self._check_features(features)
        return x

    def infer(self, w, features):
        assert w.shape == (self.num_attributes,)
        assert self._check_features(features)

        if (w == 0).all():
            raise RuntimeError("inference with w == 0 is undefined")

        data = {
            "N_ATTRIBUTES": self.num_attributes,
            "W": array_to_assignment(w, float),
            "INPUT_X": [0] * self.num_attributes, # doesn't matter
            "IS_IMPROVEMENT_QUERY": "false",
        }
        assignments = minizinc(_PROBLEM_PATH, data=data)

        return assignment_to_array(assignments[0]["x"])

    def query_improvement(self, x, features):
        assert x.shape == (self.num_attributes,)
        assert self._check_features(features)

        if (self.w_star == 0).all():
            raise RuntimeError("improvement with w_star == 0 is undefined")

        data = {
            "N_ATTRIBUTES": self.num_attributes,
            "W": array_to_assignment(self.w_star, float),
            "INPUT_X": array_to_assignment(x, int),
            "IS_IMPROVEMENT_QUERY": "true",
        }
        assignments = minizinc(_PROBLEM_PATH, data=data)

        return assignment_to_array(assignments[0]["x"])

    def query_critique(self, x, features):
        # No-op
        return None
