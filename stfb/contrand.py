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
array[ATTRIBUTES] of -1.0 .. 1.0: INPUT_X;
array[ATTRIBUTES] of var -1.0 .. 1.0: x;
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

class ContRandProblem(Problem):
    def __init__(self, num_attributes, max_length=3, rng=None):
        rng = check_random_state(rng)

        attributes = list(range(num_attributes))

        self.constraints, deps, j = [], [], 0
        for length in range(1, max_length + 1):
            for clique in combinations(attributes, length):
                coefficients = rng.randint(10, size=len(clique))
                dot = " + ".join(["({} * x[{}])".format(c, i + 1)
                                  for c, i in zip(coefficients, clique)])
                bias = rng.randint(10)
                constraint = "constraint phi[{}] = ({} >= {});".format(j + 1, dot, bias)
                self.constraints.append(constraint)
                deps.append((j, clique))
                j += 1
        num_features = len(self.constraints)

        global _TEMPLATE
        _TEMPLATE = \
            _TEMPLATE.format(phis="\n".join(self.constraints), solve="{solve}")

        w_star = sdepnormal(num_attributes, num_features, deps,
                            sparsity=0.1, rng=rng, dtype=np.float32)

        super().__init__(num_attributes, num_attributes, num_features, w_star)

    def get_feature_radius(self):
        return 1.0

    def phi(self, x, features):
        assert x.shape == (self.num_attributes,)

        targets = self.enumerate_features(features)

        PATH = "contrand-phi.mzn"
        with open(PATH, "wb") as fp:
            fp.write(_TEMPLATE.format(solve=_SATISFY).encode("utf-8"))

        data = {
            "N_ATTRIBUTES": self.num_attributes,
            "N_FEATURES": self.num_features,
            "ACTIVE_FEATURES": set([0]),
            "W": [0.0] * self.num_features,
            "x": array_to_assignment(x, float),
            "INPUT_X": [0.0] * self.num_attributes,
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

        PATH = "contrand-infer.mzn"
        with open(PATH, "wb") as fp:
            fp.write(_TEMPLATE.format(solve=_MAXIMIZE).encode("utf-8"))

        data = {
            "N_ATTRIBUTES": self.num_attributes,
            "N_FEATURES": self.num_features,
            "ACTIVE_FEATURES": set(targets),
            "W": array_to_assignment(w, float),
            "INPUT_X": [0.0] * self.num_attributes,
            "IS_IMPROVEMENT_QUERY": "false",
        }
        assignments = minizinc(PATH, data=data, output_vars=["x", "objective"], parallel=0)

        return assignment_to_array(assignments[0]["x"])

    def query_improvement(self, x, features):
        assert x.shape == (self.num_attributes,)

        targets = self.enumerate_features(features)
        assert (self.w_star[targets] != 0).any()

        PATH = "contrand-improve.mzn"
        with open(PATH, "wb") as fp:
            fp.write(_TEMPLATE.format(solve=_MAXIMIZE).encode("utf-8"))

        # TODO apply noise to w_star

        data = {
            "N_ATTRIBUTES": self.num_attributes,
            "N_FEATURES": self.num_features,
            "ACTIVE_FEATURES": set(targets),
            "W": array_to_assignment(self.w_star, float),
            "INPUT_X": array_to_assignment(x, float),
            "IS_IMPROVEMENT_QUERY": "true",
        }
        assignments = minizinc(PATH, data=data, output_vars=["x", "objective"])

        return assignment_to_array(assignments[0]["x"])

    def query_critique(self, x, features):
        assert x.shape == (self.num_attributes,)

        x_bar = self.query_improvement(x, features)

        u = self.utility(x, "all")
        u_bar = self.utility(x_bar, "all")
        u_star = self.utility(self.x_star, "all")

        if (u_bar - u) >= 0.1 * (u_star - u):
            return None, None

        targets = self.enumerate_features(features)

        scores = self.w_star * self.phi(x, "all")
        scores[targets] = np.nan
        rho = np.nanargmin(scores)
        sign = np.sign(self.w_star[rho])

        return rho, sign
