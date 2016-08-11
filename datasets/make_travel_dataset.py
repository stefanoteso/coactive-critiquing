#!/usr/bin/env python3

import numpy as np
import pickle

NUM_LOCATIONS = 10
NUM_ACTIVITIES = 5
MAX_TRAVEL_TIME = 5

np.random.seed(0)

location_activities = np.vstack([
    np.random.randint(0, 2, size=(NUM_LOCATIONS, NUM_ACTIVITIES)),
    np.zeros((1, NUM_ACTIVITIES), dtype=int),
])

location_cost = np.hstack([
    np.random.randint(1, 10, size=NUM_LOCATIONS),
    [0],
])

half = np.random.randint(1, MAX_TRAVEL_TIME + 1,
                         size=(NUM_LOCATIONS, NUM_LOCATIONS))
travel_time = np.rint((half + half.T) / 2).astype(int)

num_regions = np.random.randint(2, NUM_LOCATIONS)
regions = [np.random.randint(1, num_regions) for _ in range(NUM_LOCATIONS)]

print("""\
location activities =
{location_activities}

location costs =
{location_cost}

travel time =
{travel_time}

regions =
{regions}
""".format(**locals()))

dataset = {
    "location_activities": location_activities,
    "location_cost": location_cost,
    "travel_time": travel_time,
    "num_regions": num_regions,
    "regions": regions,
}
with open("datasets/travel.pickle", "wb") as fp:
    pickle.dump(dataset, fp)
