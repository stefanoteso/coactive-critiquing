# -*- encoding: utf-8 -*-

import numpy as np
import pickle
from pymzn import minizinc
from itertools import product, combinations
from sklearn.utils import check_random_state
from textwrap import dedent

from . import Problem

_TEMPLATE = """\
include "globals.mzn";

% constants
int: T;

int: N_LOCATIONS;
set of int: LOCATIONS = 1..N_LOCATIONS;
set of int: LOCATIONS1 = 1..N_LOCATIONS+1;
int: NO_LOCATION = N_LOCATIONS+1;

int: N_REGIONS;
set of int: REGIONS = 1..N_REGIONS;
set of int: REGIONS1 = 1..N_REGIONS+1;
int: NO_REGION = N_REGIONS+1;
array[LOCATIONS] of REGIONS: LOCATION_REGION;

int: N_ACTIVITIES;
set of int: ACTIVITIES = 1..N_ACTIVITIES;

array[LOCATIONS1, ACTIVITIES] of 0..1: LOCATION_ACTIVITIES;
array[LOCATIONS1] of int: LOCATION_COST;
array[LOCATIONS, LOCATIONS] of int: TRAVEL_TIME;

% variables
array[1..T] of var LOCATIONS1: location;
array[1..T] of var int: duration;
array[1..T-1] of var int: travel;
var int: travel_time = sum(travel);

array[1..T] of var REGIONS1: regions = [if location[t] == NO_LOCATION then NO_REGION else LOCATION_REGION[location[t]] endif | t in 1..T];
array[REGIONS1] of var 0..N_LOCATIONS: region_counts;
constraint global_cardinality(regions, [i | i in REGIONS1], region_counts);
var int: n_different_regions =
    among(region_counts, 1..N_REGIONS);

% count number of distinct locations
array[LOCATIONS] of var int: location_counts;
constraint global_cardinality(location, [i | i in LOCATIONS], location_counts);
var int: n_different_locations =
    among(location_counts, 1..N_LOCATIONS);

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
    travel[i] = TRAVEL_TIME[location[i], location[i+1]]);

% configuration
array[1..3*T-1] of var int: x;
constraint forall(t in 1..T)(x[t] = location[t]);
constraint forall(t in 1..T)(x[T+t] = duration[t]);
constraint forall(t in 1..T-1)(x[2*T+t] = travel[t]);
array[1..3*T-1] of int: INPUT_X;

% features
int: N_FEATURES;
set of int: FEATURES = 1..N_FEATURES;

set of int: ACTIVE_FEATURES;

array[FEATURES] of float: W;
array[FEATURES] of var int: phi;
array[FEATURES] of int: INPUT_PHI;

{phis}

{solve}
"""

_PHI = "solve satisfy;"

_INFER = """\
var float: objective =
    sum(feat in ACTIVE_FEATURES)(W[feat] * phi[feat]);

solve maximize objective;
"""

_IMPROVE = """\
var int: objective =
    sum(t in 1..3*T-1)(x[t] != INPUT_X[t]);

constraint
    sum(feat in ACTIVE_FEATURES)(W[feat] * (phi[feat] - INPUT_PHI[feat])) > 0;

constraint objective >= 1;

solve minimize objective;
"""

class TravelProblem(Problem):
    def __init__(self, horizon=10, noise=0.1, sparsity=0.2, rng=None, 
                 w_star=None):
        rng = check_random_state(rng)
        self.noise, self.rng = noise, rng

        self._horizon = horizon
        num_attributes = 3 * horizon - 1

        with open("datasets/travel_tn.pickle", "rb") as fp:
            dataset = pickle.load(fp)

        self._location_activities = dataset["location_activities"]
        self._num_locations = dataset["location_activities"].shape[0] - 1
        self._num_activities = dataset["location_activities"].shape[1]
        self._location_cost = dataset["location_cost"]
        self._travel_time = dataset["travel_time"]
        self._regions = dataset["regions"]
        self._num_regions = dataset["num_regions"]

        # Generate the features

        j = 1
        self.features = []

        # Number of time slots spent in a location
        for location in range(1, self._num_locations + 1):
            feature = "constraint phi[{j}] = sum(i in 1..T)(location[i] = {location});".format(**locals())
            self.features.append(feature)
            j += 1

        # Number of time slots with access to an activity
        for activity in range(1, self._num_activities + 1):
            feature = "constraint phi[{j}] = sum(i in 1..T)(LOCATION_ACTIVITIES[location[i], {activity}]);".format(**locals())
            self.features.append(feature)
            j += 1

        num_base_features = j - 1

        # Number of distinct locations
        feature = "constraint phi[{j}] = n_different_locations;".format(**locals())
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

        # Regions
        for region in range(1, self._num_regions + 1):
            feature = "constraint phi[{j}] = 2 * (region_counts[{region}] + travel_time == T) - 1;".format(**locals())
            self.features.append(feature)
            j += 1

        # Number of different regions
        features = "constraint phi[{j}] = n_different_regions;".format(**locals())
        self.features.append(features)
        j += 1

        # Soft dependencies between locations
        for location1, location2 in combinations(range(1, self._num_locations + 1), 2):
            feature = "constraint phi[{j}] = 2 * (location_counts[{location1}] = 0 \/ location_counts[{location2}] > 0) - 1;".format(**locals())
            self.features.append(feature)
            j += 1
            feature = "constraint phi[{j}] = 2 * (location_counts[{location2}] = 0 \/ location_counts[{location1}] > 0) - 1;".format(**locals())
            self.features.append(feature)
            j += 1

        num_features = j - 1

        global _TEMPLATE
        _TEMPLATE = \
            _TEMPLATE.format(phis="\n".join(self.features), solve="{solve}")

        if w_star is None:
            # Sample the weight vector
            w_star = rng.normal(size=num_features)
            if sparsity < 1.0:
                nnz_features = max(1, int(np.ceil(sparsity * num_features)))
                zeros = rng.permutation(num_features)[nnz_features:]
                w_star[zeros] = 0

        super().__init__(num_attributes, num_base_features, num_features,
                         w_star)

    def get_feature_radius(self):
        # XXX incorrect
        return float(max(self._num_locations, self._num_activities))

    def phi(self, x, features):
        PATH = "travel-phi.mzn"

        with open(PATH, "wb") as fp:
            fp.write(_TEMPLATE.format(solve=_PHI).encode("utf-8"))

        data = {
            "N_FEATURES": self.num_features,
            "ACTIVE_FEATURES": set([1]), # doesn't matter
            "T": self._horizon,
            "N_REGIONS": self._num_regions,
            "LOCATION_REGION": self._regions,
            "N_LOCATIONS": self._num_locations,
            "N_ACTIVITIES": self._num_activities,
            "LOCATION_ACTIVITIES": self._location_activities,
            "LOCATION_COST": self._location_cost,
            "TRAVEL_TIME": self._travel_time,
            "W": [0] * self.num_features, # doesn't matter
            "x": self.array_to_assignment(x, int),
            "INPUT_X": [0] * self.num_attributes, # doesn't matter
            "INPUT_PHI": [0] * self.num_features, # doesn't matter
        }

        return super().phi(x, features, PATH, data).astype(np.int32)

    def infer(self, w, features):
        PATH = "travel-infer.mzn"

        with open(PATH, "wb") as fp:
            fp.write(_TEMPLATE.format(solve=_INFER).encode("utf-8"))

        targets = self.enumerate_features(features)
        data = {
            "N_FEATURES": self.num_features,
            "ACTIVE_FEATURES": {j + 1 for j in targets},
            "T": self._horizon,
            "N_REGIONS": self._num_regions,
            "LOCATION_REGION": self._regions,
            "N_LOCATIONS": self._num_locations,
            "N_ACTIVITIES": self._num_activities,
            "LOCATION_ACTIVITIES": self._location_activities,
            "LOCATION_COST": self._location_cost,
            "TRAVEL_TIME": self._travel_time,
            "W": self.array_to_assignment(w, float),
            "INPUT_X": [0] * self.num_attributes, # doesn't matter
            "INPUT_PHI": [0] * self.num_features, # doesn't matter
        }

        return super().infer(w, features, PATH, data)

    def query_improvement(self, x, features):
        w_star = np.array(self.w_star)
        if self.noise:
            nnz = w_star.nonzero()[0]
            w_star[nnz] += self.rng.normal(0, self.noise, size=len(nnz)).astype(np.float32)

        PATH = "travel-improve.mzn"

        with open(PATH, "wb") as fp:
            fp.write(_TEMPLATE.format(solve=_IMPROVE).encode("utf-8"))

        targets = self.enumerate_features(features)
        phi = self.phi(x, "all") # XXX the sum is on ACTIVE_FEATURES anyway
        data = {
            "N_FEATURES": self.num_features,
            "ACTIVE_FEATURES": {j + 1 for j in targets},
            "T": self._horizon,
            "N_REGIONS": self._num_regions,
            "LOCATION_REGION": self._regions,
            "N_LOCATIONS": self._num_locations,
            "N_ACTIVITIES": self._num_activities,
            "LOCATION_ACTIVITIES": self._location_activities,
            "LOCATION_COST": self._location_cost,
            "TRAVEL_TIME": self._travel_time,
            "W": self.array_to_assignment(w_star, float),
            "INPUT_X": self.array_to_assignment(x, int),
            "INPUT_PHI": self.array_to_assignment(phi, int),
        }

        return super().query_improvement(x, w_star, features, PATH, data)
