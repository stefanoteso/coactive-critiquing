# -*- encoding: utf-8 -*-

import numpy as np
from pymzn import minizinc
from itertools import product, combinations
from sklearn.utils import check_random_state

from . import Problem, array_to_assignment, assignment_to_array, spnormal

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
        super().__init__(num_attributes, num_attributes, num_attributes,
                         rng.normal(0, 1, size=num_attributes))

    def get_feature_radius(self):
        return 1.0

    def phi(self, x, features):
        assert x.shape == (self.num_attributes,)
        assert features in ("attributes", "all")
        return x

    def infer(self, w, features):
        assert w.shape == (self.num_features,)
        assert features in ("attributes", "all")

        if (w == 0).all():
            raise RuntimeError("inference with w == 0 is undefined")

        data = {
            "num_features": self.num_features,
            "w": array_to_assignment(w, float),
        }
        assignments = minizinc("stfb/dumb-infer.mzn", data=data)

        return assignment_to_array(assignments[0]["x"])

    def query_critique(self, x, features):
        # No-op
        return None

    def query_improvement(self, x, features):
        assert x.shape == (self.num_attributes,)
        assert features in ("attributes", "all")

        if (self.w_star == 0).all():
            raise RuntimeError("improvement with w_star == 0 is undefined")

        data = {
            "num_features": self.num_features,
            "w": array_to_assignment(self.w_star, float),
            "x": array_to_assignment(x, int),
            "max_changes": 2,
        }
        assignments = minizinc("stfb/dumb-improve.mzn", data=data)

        return assignment_to_array(assignments[0]["x_bar"])
