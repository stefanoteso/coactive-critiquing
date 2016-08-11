#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import numpy as np
import pickle

CANVAS_SIZE = 100
NUM_RECTANGLES = 100

np.random.seed(0)

reorder = lambda a, b: (min(a, b), max(a, b))

rectangles = []
for j in range(NUM_RECTANGLES):
    xmin, xmax = reorder(*np.random.randint(1, 100+1, size=2))
    ymin, ymax = reorder(*np.random.randint(1, 100+1, size=2))
    rectangles.append([xmin, xmax, ymin, ymax])

print("rectangles =\n{}".format("\n".join(map(str, rectangles))))

dataset = {
    "canvas_size": CANVAS_SIZE,
    "rectangles": rectangles,
}
with open("datasets/canvas.pickle", "wb") as fp:
    pickle.dump(dataset, fp)
