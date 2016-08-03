# -*- encoding: utf-8 -*-

import numpy as np
from pymzn import minizinc
from itertools import product, combinations
from sklearn.utils import check_random_state
from textwrap import dedent

from . import Problem

_TEMPLATE = """\
% constants
int: T;

int: N_LOCATIONS;
set of int: LOCATIONS = 1..N_LOCATIONS;
set of int: LOCATIONS1 = 1..N_LOCATIONS+1;
int: NO_LOCATION = N_LOCATIONS+1;

int: N_ACTIVITIES;
set of int: ACTIVITIES = 1..N_ACTIVITIES;

array[LOCATIONS1, ACTIVITIES] of 0..1: LOCATION_ACTIVITIES;
array[LOCATIONS1] of int: LOCATION_COST;
array[LOCATIONS, LOCATIONS] of int: TRAVEL_TIME;

% variables
array[1..T] of var LOCATIONS1: location;
array[1..T] of var int: duration;
array[1..T-1] of var int: travel;

% bounds on duration
constraint forall(i in 1..T)(duration[i] >= 0);
constraint forall(i in 1..T)(duration[i] <= T);

% bounds on travel
constraint forall(i in 1..T-1)(travel[i] >= 0);
constraint forall(i in 1..T-1)(travel[i] <= T);

% trip must fit into available time
constraint (sum(i in 1..T)(duration[i]) + sum(i in 1..T-1)(travel[i])) = T;

% location implies durations and conversely
constraint forall(i in 1..T)(
    location[i] = NO_LOCATION <-> duration[i] = 0);

% null travels are only at the end
constraint forall(i in 1..T-1)(
    location[i+1] = NO_LOCATION <-> travel[i] = 0);

% consecutive locations must be different
constraint forall(i in 1..T-1 where location[i] != NO_LOCATION)(
    location[i] != location[i+1]);

% null locations are only at the end
constraint forall(i in 1..T-1 where location[i] = NO_LOCATION)(
    location[i+1] = NO_LOCATION);

% traveling from one location to another takes time
constraint forall(i in 1..T-1 where location[i+1] != NO_LOCATION)(
    travel[i] >= TRAVEL_TIME[location[i], location[i+1]]);

% configuration
array[1..3*T-1] of var int: x;
constraint forall(i in 1..T)(x[i] = location[i]);
constraint forall(i in 1..T)(x[T+i] = duration[i]);
constraint forall(i in 1..T-1)(x[2*T+i] = travel[i]);

% features
int: N_FEATURES;
set of int: FEATURES = 1..N_FEATURES;

array[FEATURES] of float: W;
array[FEATURES] of var float: phi;

{phis}

{solve}

% support for the improvement query
bool: IS_IMPROVEMENT_QUERY;
array[1..3*T-1] of int: INPUT_X;
constraint IS_IMPROVEMENT_QUERY ->
    (sum(attr in 1..3*T-1)(bool2int(x[attr] != INPUT_X[attr])) <= 3);
"""

_SOLVE_PHI = "solve satisfy;"
_SOLVE_INFER_IMPROVE = "solve maximize sum(feat in FEATURES)(W[feat] * phi[feat]);"

N_LOCATIONS = 10
N_ACTIVITIES = 10

class TravelProblem(Problem):
    def __init__(self, horizon=10, rng=None):
        rng = check_random_state(rng)

        self._horizon = horizon
        num_attributes = 3 * horizon - 1

        # Generate the dataset
        # XXX ideally these would be provided externally
        self._location_activities = np.vstack([
            rng.randint(0, 2, size=(N_LOCATIONS, N_ACTIVITIES)),
            np.zeros((1, N_ACTIVITIES))
        ]).astype(int)
        self._location_cost = np.hstack([
            rng.randint(1, 10, size=N_LOCATIONS),
            [0]
        ])
        temp = rng.randint(1, 5, size=(N_LOCATIONS, N_LOCATIONS))
        self._travel_time = temp + temp.T

        print(dedent("""\
            TRAVEL DATASET:

            location_activities =
            {}

            location_cost =
            {}

            travel_time =
            {}
        """).format(self._location_activities, self._location_cost,
                    self._travel_time))

        # Generate the features

        j = 1
        self.features = []

        # Number of time slots spent in a location
        for location in range(1, N_LOCATIONS + 1):
            feature = "constraint phi[{j}] = sum(i in 1..T)(bool2int(location[i] = {location}));".format(**locals())
            self.features.append(feature)
            j += 1

        # Number of time slots with access to an activity
        for activity in range(1, N_ACTIVITIES + 1):
            feature = "constraint phi[{j}] = sum(i in 1..T)(LOCATION_ACTIVITIES[location[i], {activity}]);".format(**locals())
            self.features.append(feature)
            j += 1

        # Total time spent traveling
        feature = "constraint phi[{j}] = sum(i in 1..T-1)(travel[i]);".format(**locals())
        self.features.append(feature)
        j += 1

        # Total cost
        feature = "constraint phi[{j}] = sum(i in 1..T)(LOCATION_COST[location[i]]);".format(**locals())
        self.features.append(feature)
        j += 1

        # TODO add user features:
        # - local: sequence features
        # - global: stay within a subsets of locations

        num_base_features = j - 1
        num_features = num_base_features

        # Sample the weight vector
        w_star = spnormal(num_features, sparsity=0.2, rng=rng, dtype=np.float32)

        super().__init__(num_attributes, num_base_features, num_features,
                         w_star)

    def get_feature_radius(self):
        # XXX incorrect
        return float(max(N_LOCATIONS, N_ACTIVITIES))

    def phi(self, x, features):
        features = self.enumerate_features(features)
        assert x.shape == (self.num_attributes,)

        phis = "\n".join([self.features[j] for j in features])
        solve = _SOLVE_PHI;
        problem = _TEMPLATE.format(**locals())

        PATH = "travel-phi.mzn"
        with open(PATH, "wb") as fp:
            fp.write(problem.encode("utf-8"))

        data = {
            "N_FEATURES": len(features),
            "T": self._horizon,
            "N_LOCATIONS": N_LOCATIONS,
            "N_ACTIVITIES": N_ACTIVITIES,
            "LOCATION_ACTIVITIES": self._location_activities,
            "LOCATION_COST": self._location_cost,
            "TRAVEL_TIME": self._travel_time,
            "W": [0] * len(features), # doesn't matter
            "x": self.array_to_assignment(x, int),
            "INPUT_X": [0] * self.num_attributes, # doesn't matter
            "IS_IMPROVEMENT_QUERY": "false",
        }
        assignments = minizinc(PATH, data=data)

        return self.assignment_to_array(assignments[0]["phi"])

    def infer(self, w, features):
        features = self.enumerate_features(features)
        assert w.shape == (len(features),)

        if (w == 0).all():
            raise RuntimeError("inference with w == 0 is undefined")

        phis = "\n".join([self.features[j] for j in features])
        solve = _SOLVE_INFER_IMPROVE;
        problem = _TEMPLATE.format(**locals())

        PATH = "travel-infer.mzn"
        with open(PATH, "wb") as fp:
            fp.write(problem.encode("utf-8"))

        data = {
            "N_FEATURES": len(features),
            "T": self._horizon,
            "N_LOCATIONS": N_LOCATIONS,
            "N_ACTIVITIES": N_ACTIVITIES,
            "LOCATION_ACTIVITIES": self._location_activities,
            "LOCATION_COST": self._location_cost,
            "TRAVEL_TIME": self._travel_time,
            "W": self.array_to_assignment(w, float),
            "INPUT_X": [0] * self.num_attributes, # doesn't matter
            "IS_IMPROVEMENT_QUERY": "false",
        }
        assignments = minizinc(PATH, data=data)

        return self.assignment_to_array(assignments[0]["x"])

    def query_improvement(self, x, features):
        features = self.enumerate_features(features)
        assert x.shape == (self.num_attributes,)

        w_star = self.w_star[features]
        if (w_star == 0).all():
            raise RuntimeError("inference with w == 0 is undefined")

        phis = "\n".join([self.features[j] for j in features])
        solve = _SOLVE_INFER_IMPROVE;
        problem = _TEMPLATE.format(**locals())

        PATH = "travel-improve.mzn"
        with open(PATH, "wb") as fp:
            fp.write(problem.encode("utf-8"))

        data = {
            "N_FEATURES": len(features),
            "T": self._horizon,
            "N_LOCATIONS": N_LOCATIONS,
            "N_ACTIVITIES": N_ACTIVITIES,
            "LOCATION_ACTIVITIES": self._location_activities,
            "LOCATION_COST": self._location_cost,
            "TRAVEL_TIME": self._travel_time,
            "W": self.array_to_assignment(w_star, float),
            "INPUT_X": self.array_to_assignment(x, int),
            "IS_IMPROVEMENT_QUERY": "true",
        }
        assignments = minizinc(PATH, data=data)

        x_bar = self.assignment_to_array(assignments[0]["x"])
        return x_bar
