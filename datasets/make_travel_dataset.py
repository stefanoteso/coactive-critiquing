#!/usr/bin/env python3

import numpy as np
import pickle

HORIZON = 10
NUM_LOCATIONS = 10
NUM_ACTIVITIES = 5

rng = np.random.RandomState(0)

LOCATION_ACTIVITIES = np.vstack([
    rng.randint(0, 2, size=(NUM_LOCATIONS, NUM_ACTIVITIES)),
    np.zeros((1, NUM_ACTIVITIES))
])

LOCATION_COST = np.hstack([
    rng.randint(1, 10, size=NUM_LOCATIONS),
    [0]
])

temp = rng.randint(1, 5, size=(NUM_LOCATIONS, NUM_LOCATIONS))
TRAVEL_TIME = np.rint((temp + temp.T) / 2)

print("""\
location activities =
{}

location costs =
{}

travel time =
{}
""".format(LOCATION_ACTIVITIES, LOCATION_COST, TRAVEL_TIME))

DATASET = (LOCATION_ACTIVITIES, LOCATION_COST, TRAVEL_TIME)

with open("datasets/travel.pickle", "wb") as fp:
    pickle.dump(DATASET, fp)
