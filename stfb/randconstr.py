# -*- encoding: utf-8 -*-

import numpy as np
from pymzn import minizinc
from itertools import product, combinations
from sklearn.utils import check_random_state

from . import Problem, array_to_assignment, assignment_to_array, spnormal

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

constraint objective = sum(j in FEATURES)(W[j] * phi[j]);

constraint IS_IMPROVEMENT_QUERY ->
    sum(i in ATTRIBUTES)(bool2int(x[i] != INPUT_X[i])) <= 3;

{solve}
"""

_SATISFY = "solve satisfy;"
_MINIMIZE = "solve maximize objective;"

class RandConstrBoolProblem(Problem):
    """A randomly-constrained Boolean problem.

    Constraints are conjunctions of attributes.

    Parameters
    ----------
    num_attributes : positive int
        Number of base attributes.
    max_length : positive int
        Maximum length of the clauses (features).
    rng : None or int or numpy.random.RandomState, defaults to None
        The RNG.
    """
    def __init__(self, num_attributes, max_length=3, rng=None):
        rng = check_random_state(rng)

        # Enumerate all features (with attribute-level features first)
        self.features, j = [], 1
        attributes = list(range(num_attributes))
        for length in range(1, max_length + 1):
            for clique in combinations(attributes, length):
                clause = " /\\ ".join(["x[{}]".format(i + 1) for i in clique])
                feature = "constraint phi[{j}] = ({clause});".format(**locals())
                self.features.append(feature)
                j += 1
        num_features = len(self.features)

        # Sample the weight vector
        w_star = spnormal(num_features, rng=rng, dtype=np.float32)

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
        assignments = minizinc(PATH, data=data)

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
        assignments = minizinc(PATH, data=data)

        return assignment_to_array(assignments[0]["x"])

    def query_improvement(self, x, features):
        features = self.enumerate_features(features)
        assert x.shape == (self.num_attributes,)

        w_star = self.w_star[features]
        if (w_star == 0).all():
            print("improvement query with w == 0")

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
        assignments = minizinc(PATH, data=data)

        return assignment_to_array(assignments[0]["x"])

    def query_critique(self, x, features):
        features = self.enumerate_features(features)
        assert x.shape == (self.num_attributes,)

        w_star = self.w_star[features]
        if (w_star == 0).all():
            print("critique query with w == 0")

        latent_features = list(set(range(self.num_features)) - set(features))

        raise NotImplementedError()
