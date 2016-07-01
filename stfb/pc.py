# -*- encoding: utf-8 -*-

import numpy as np
from pymzn import minizinc, verbose
from itertools import product, combinations
from sklearn.utils import check_random_state

from . import Problem, array_to_assignment, assignment_to_array, spnormal

_TEMPLATE = """\
int: NUM_FEATURES;
set of int: FEATURES = 1..NUM_FEATURES;

int: NUM_TYPES = 3;
set of int: TYPES = 1..NUM_TYPES;

int: NUM_MANUFACTURERS = 8;
set of int: MANUFACTURERS = 1..NUM_MANUFACTURERS;

int: NUM_CPUS = 37;
set of int: CPUS = 1..NUM_CPUS;
set of int: CPU_AMDS = 1..4;
int: CPU_CRUSOE = 5;
set of int: CPU_CELERONS = 6..15;
set of int: CPU_PENTIUMS = 16..27;
set of int: CPU_POWERPCS = 28..37;

int: NUM_MONITORS = 8;
set of int: MONITORS = 1..NUM_MONITORS;

int: NUM_RAM = 10;
set of int: RAM = 1..NUM_RAM;
set of int: RAM_LAPTOPS = {{1, 2, 3, 4, 5, 6, 7, 8, 9}};
set of int: RAM_DESKTOPS = {{2, 5, 8, 9}};
set of int: RAM_TOWERS = {{5, 8, 9, 10}};

int: NUM_HD = 10;
set of int: HD = 1..NUM_HD;

array[TYPES] of float: COST_TYPE = [
    50, 0, 80
];
array[MANUFACTURERS] of float: COST_MANUFACTURER = [
    100, 0, 100, 50, 0, 0, 50, 50
];
array[CPUS] of float: COST_CPU = [
    1.4*100,    1.4*130,    1.1*70,     1.1*90,     1.2*50,
    1.2*60,     1.2*80,     1.2*90,     1.2*100,    1.2*110,
    1.2*120,    1.2*130,    1.2*140,    1.2*170,    1.5*50,
    1.5*60,     1.5*80,     1.5*90,     1.5*100,    1.5*110,
    1.5*130,    1.5*150,    1.5*160,    1.5*170,    1.5*180,
    1.5*220,    1.4*27,     1.4*30,     1.4*40,     1.4*45,
    1.4*50,     1.4*55,     1.4*60,     1.4*70,     1.6*70,
    1.6*73,     1.2*80
];
array[MONITORS] of float: COST_MONITOR = [
    0.6*100,    0.6*104,    0.6*120,    0.6*133,    0.6*140,
    0.6*150,    0.6*170,    0.6*210
];
array[RAM] of float: COST_RAM = [
    0.8*64,     0.8*128,    0.8*160,    0.8*192,    0.8*256,
    0.8*320,    0.8*384,    0.8*512,    0.8*1024,   0.8*2048
];
array[HD] of float: COST_HD = [
    4*8,    4*10,   4*12,   4*15,   4*20,
    4*30,   4*40,   4*60,   4*80,   4*120
];

var TYPES: x_type;
var MANUFACTURERS: x_manufacturer;
var CPUS: x_cpu;
var MONITORS: x_monitor;
var RAM: x_ram;
var HD: x_hd;

var float: x_cost =
    COST_TYPE[x_type] +
    COST_MANUFACTURER[x_manufacturer] +
    COST_CPU[x_cpu] +
    COST_MONITOR[x_monitor] +
    COST_RAM[x_ram] +
    COST_HD[x_hd];

array[1..7] of var float: x;
constraint x[1] = x_type;
constraint x[2] = x_manufacturer;
constraint x[3] = x_cpu;
constraint x[4] = x_monitor;
constraint x[5] = x_ram;
constraint x[6] = x_hd;
constraint x[7] = x_cost;

bool: IS_IMPROVEMENT_QUERY;
array[1..7] of float: INPUT_X;
% NOTE we do not require the costs to coincide: cost is a dependent variable.
% this also works around the difficulty of enforcing equality between floats
% without running into unsats.
constraint IS_IMPROVEMENT_QUERY ->
    (sum(attr in 1..6)(bool2int(INPUT_X[attr] != x[attr])) <= 3);

array[FEATURES] of float: W;
array[FEATURES] of var float: phi;

{phis}

{solve}

% Compaq -> Laptop or Desktop
constraint (x_manufacturer = 2) -> (x_type = 1 \/ x_type = 2);

% Fujitsu -> Laptop
constraint (x_manufacturer = 4) -> (x_type = 1);

% HP -> Desktop
constraint (x_manufacturer = 6) -> (x_type = 2);

% Sony -> Laptop or Tower
constraint (x_manufacturer = 7) -> (x_type = 1 \/ x_type = 3);

% Apple -> PowerPC*
constraint (x_manufacturer = 1) -> (x_cpu in CPU_POWERPCS);

% Compac or Sony -> AMD* or Intel*
constraint (x_manufacturer = 2 \/ x_manufacturer = 7) ->
    (x_cpu in CPU_AMDS \/ x_cpu in CPU_CELERONS \/ x_cpu in CPU_PENTIUMS);

% Fujitsu -> Crusoe or Intel*
constraint (x_manufacturer = 4) ->
    (x_cpu in CPU_CELERONS \/ x_cpu in CPU_PENTIUMS \/ x_cpu = CPU_CRUSOE);

% Dell or Gateway or Toshiba -> Intel*
constraint (x_manufacturer = 5 \/ x_manufacturer = 8) ->
    (x_cpu in CPU_CELERONS \/ x_cpu in CPU_PENTIUMS);

% HP -> Intel Pentium*
constraint (x_manufacturer = 6) -> (x_cpu in CPU_PENTIUMS);

% Type -> RAM size
constraint (x_type = 1) -> (x_ram in RAM_LAPTOPS);
constraint (x_type = 2) -> (x_ram in RAM_DESKTOPS);
constraint (x_type = 3) -> (x_ram in RAM_TOWERS);

% Type -> HD size
constraint (x_type = 1) -> (x_hd <= 6);
constraint (x_type = 2 \/ x_type = 3) -> (x_hd >= 5);

% Type -> Monitor size
constraint (x_type = 1) -> (x_monitor <= 6);
constraint (x_type = 2 \/ x_type = 3) -> (x_monitor >= 7);
"""

_SOLVE_PHI = "solve satisfy;"
_SOLVE_INFER_IMPROVE = "solve maximize sum(feat in FEATURES)(W[feat] * phi[feat]);"

class PCProblem(Problem):
    def __init__(self, rng=None):
        rng = check_random_state(rng)

        BOOL_DOMAINS = {
            "type":         list(range(1, 3+1)),
            "manufacturer": list(range(1, 8+1)),
            "cpu":          list(range(1, 37+1)),
            "monitor":      list(range(1, 8+1)),
            "ram":          list(range(1, 10+1)),
            "hd":           list(range(1, 10+1)),
        }

        # NOTE the maximum-cost feasible configuration has cost 2753.4
        COST_THRESHOLDS = np.arange(0, 2800, 100)

        self.features = []

        # Attribute-level features
        j = 1
        for attr, domain in BOOL_DOMAINS.items():
            for value in domain:
                features = "constraint phi[{j}] = bool2float(x_{attr} = {value});" \
                               .format(**locals())
                self.features.append(features)
                j += 1

        self.features.append("constraint phi[{j}] = x_cost;".format(**locals()))
        j += 1

        num_base_features = j - 1

#        # Horn-like features between boolean attr and bool subsets
#        for head, body in product(BOOL_DOMAINS.items(), repeat=2):
#            head_attr, head_domain = head
#            body_attr, body_domain = body
#            if head_attr == body_attr:
#                continue
#            for body_subset_size in range(1, 2):
#                body_subsets = combinations(body_domain, body_subset_size)
#                for head_value, body_subset in product(head_domain, body_subsets):
#                    body = " \/ ".join(["x_{} = {}".format(body_attr, body_value)
#                                        for body_value in body_subset])
#                    feature = "constraint phi[{j}] = bool2float(x_{head_attr} != {head_value} \/ ({body}));" \
#                                  .format(**locals())
#                    self.features.append(feature)
#                    j += 1

#        # Enumerate all Horn rules bool attr -> cost threshold
#        for head, threshold in product(BOOL_DOMAINS.items(), COST_THRESHOLDS):
#            head_attr, head_domain = head
#            for head_value, op in product(head_domain, ["<=", ">="]):
#                feature = "constraint phi[{j}] = bool2float(x_{head_attr} != {head_value} \/ x_cost {op} {threshold});" \
#                              .format(**locals())
#                self.features.append(feature)
#                j += 1

        num_attributes = len(BOOL_DOMAINS) + 1
        num_features = len(self.features)
        assert num_features == j - 1

        # Sample the weight vector
        w_star = spnormal(num_features, rng=rng, dtype=np.float32)

        super().__init__(num_attributes, num_base_features, num_features,
                         w_star)

    def get_feature_radius(self):
        return 1.0

    def phi(self, x, features):
        features = self.enumerate_features(features)
        assert x.shape == (self.num_attributes,)

        phis = "\n".join([self.features[j] for j in features])
        solve = _SOLVE_PHI;
        problem = _TEMPLATE.format(**locals())

        PATH = "pc-phi.mzn"
        with open(PATH, "wb") as fp:
            fp.write(problem.encode("utf-8"))

        # XXX due to rounding errors, the last component of x (the PC cost)
        # ends up losing one tiny bit of precision; here we remove everything
        # below the resolution of 0.2
        x[-1] = float(int(x[-1] * 10)) / 10

        data = {
            "NUM_FEATURES": len(features),
            "W": [0] * len(features), # doesn't matter
            "x_type": int(x[0]),
            "x_manufacturer": int(x[1]),
            "x_cpu": int(x[2]),
            "x_monitor": int(x[3]),
            "x_ram": int(x[4]),
            "x_hd": int(x[5]),
            "INPUT_X": [0, 0, 0, 0, 0, 0, 0], # doesn't matter
            "IS_IMPROVEMENT_QUERY": "false",
        }
        assignments = minizinc(PATH, data=data)

        return assignment_to_array(assignments[0]["phi"])

    def infer(self, w, features):
        features = self.enumerate_features(features)
        assert w.shape == (len(features),)

        if (w == 0).all():
            raise RuntimeError("inference with w == 0 is undefined")

        phis = "\n".join([self.features[j] for j in features])
        solve = _SOLVE_INFER_IMPROVE;
        problem = _TEMPLATE.format(**locals())

        PATH = "pc-infer.mzn"
        with open(PATH, "wb") as fp:
            fp.write(problem.encode("utf-8"))

        data = {
            "NUM_FEATURES": len(features),
            "W": array_to_assignment(w, float),
            "INPUT_X": [0, 0, 0, 0, 0, 0, 0], # doesn't matter
            "IS_IMPROVEMENT_QUERY": "false",
        }
        assignments = minizinc(PATH, data=data)

        return assignment_to_array(assignments[0]["x"])

    def query_improvement(self, x, features):
        features = self.enumerate_features(features)
        assert x.shape == (self.num_attributes,)

        w_star = self.w_star[features]
        if (w_star == 0).all():
            raise RuntimeError("inference with w == 0 is undefined")

        phis = "\n".join([self.features[j] for j in features])
        solve = _SOLVE_INFER_IMPROVE;
        problem = _TEMPLATE.format(**locals())

        PATH = "pc-improve.mzn"
        with open(PATH, "wb") as fp:
            fp.write(problem.encode("utf-8"))

        data = {
            "NUM_FEATURES": len(features),
            "W": array_to_assignment(w_star, float),
            "INPUT_X": array_to_assignment(x, float),
            "IS_IMPROVEMENT_QUERY": "true",
        }
        assignments = minizinc(PATH, data=data)

        x_bar = assignment_to_array(assignments[0]["x"])
        return x_bar

    def query_critique(self, x, features):
        features = self.enumerate_features(features)
        assert x.shape == (self.num_attributes,)

        # WRITEME

        return None
