# -*- encoding: utf-8 -*-

import numpy as np
from pymzn import minizinc
from itertools import product, combinations
from sklearn.utils import check_random_state

from . import Problem

_RAM_DESKTOPS = {2, 5, 8, 9};
_RAM_TOWERS = {5, 8, 9, 10};

_TEMPLATE = """\
int: N_TYPES = 3;
set of int: TYPES = 1..N_TYPES;

int: N_MANUFACTURERS = 8;
set of int: MANUFACTURERS = 1..N_MANUFACTURERS;

int: N_CPUS = 37;
set of int: CPUS = 1..N_CPUS;
set of int: CPU_AMDS = 1..4;
int: CPU_CRUSOE = 5;
set of int: CPU_CELERONS = 6..15;
set of int: CPU_PENTIUMS = 16..27;
set of int: CPU_POWERPCS = 28..37;

int: N_MONITORS = 8;
set of int: MONITORS = 1..N_MONITORS;

int: N_RAM = 10;
set of int: RAM = 1..N_RAM;
set of int: RAM_LAPTOPS = 1..9;
set of int: RAM_DESKTOPS;
set of int: RAM_TOWERS;

int: N_HD = 10;
set of int: HD = 1..N_HD;

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

var float: x_cost = (1.0 / 2753.4) * (
    COST_TYPE[x_type] +
    COST_MANUFACTURER[x_manufacturer] +
    COST_CPU[x_cpu] +
    COST_MONITOR[x_monitor] +
    COST_RAM[x_ram] +
    COST_HD[x_hd]);

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

int: N_FEATURES;
set of int: FEATURES = 1..N_FEATURES;

set of int: TRUTH_VALUES;
set of int: ACTIVE_FEATURES;

array[FEATURES] of int: W;
array[FEATURES] of var TRUTH_VALUES: phi;
array[FEATURES] of var TRUTH_VALUES: INPUT_PHI;

array[1..6] of var int: x = [
    x_type,
    x_manufacturer,
    x_cpu,
    x_monitor,
    x_ram,
    x_hd
];
array[1..6] of int: INPUT_X;

{phis}

{solve}
"""

_PHI = "solve satisfy;"

_INFER = """\
var int: objective =
    sum(j in ACTIVE_FEATURES)(W[j] * phi[j]);

solve maximize objective;
"""

_IMPROVE = """\
var int: objective =
    sum(i in 1..6)(x[i] != INPUT_X[i]);

constraint
    sum(j in ACTIVE_FEATURES)(W[j] * phi[j]) >
        sum(j in ACTIVE_FEATURES)(W[j] * INPUT_PHI[j]);

constraint objective >= 1;

solve minimize objective;
"""

class PCProblem(Problem):
    def __init__(self, noise=0.1, sparsity=0.2, rng=None):
        rng = check_random_state(rng)
        self.noise, self.rng = noise, rng

        BOOL_DOMAINS = [
            ("type",         list(range(1, 3+1))),
            ("manufacturer", list(range(1, 8+1))),
            ("cpu",          list(range(1, 37+1))),
            ("monitor",      list(range(1, 8+1))),
            ("ram",          list(range(1, 10+1))),
            ("hd",           list(range(1, 10+1))),
        ]

        num_attributes = len(BOOL_DOMAINS)

        self.features, j = [], 0
        for attr, domain in BOOL_DOMAINS:
            for value in domain:
                equality = "(x_{} = {})".format(attr, value)
                features = "constraint phi[{}] = 2 * ({}) - 1;".format(j + 1, equality)
                self.features.append(features)
                j += 1

        num_base_features = j

        for head, body in product(BOOL_DOMAINS, repeat=2):
            head_attr, head_domain = head
            body_attr, body_domain = body
            if head_attr == body_attr:
                continue
            for body_subset_size in range(1, 2):
                body_subsets = combinations(body_domain, body_subset_size)
                for head_value, body_subset in product(head_domain, body_subsets):
                    body = " \/ ".join(["x_{} = {}".format(body_attr, body_value)
                                        for body_value in body_subset])
                    implication = "x_{head_attr} != {head_value} \/ ({body})".format(**locals())
                    feature = "constraint phi[{}] = 2 * ({}) - 1;".format(j + 1, implication)
                    self.features.append(feature)
                    j += 1

        for threshold in np.arange(0, 2999, 250):
            implication = "x_cost >= {} /\\ x_cost < {}".format(
                threshold, threshold + 250)
            feature = "constraint phi[{}] = 2 * ({}) - 1;".format(j + 1, implication)
            self.features.append(feature)
            j += 1

        num_features = len(self.features)

        global _TEMPLATE
        _TEMPLATE = \
            _TEMPLATE.format(phis="\n".join(self.features), solve="{solve}")

        # Sample the weight vector
        w_star = 2 * rng.randint(0, 2, size=num_features) - 1
        if sparsity < 1.0:
            nnz_features = max(1, int(np.ceil(sparsity * num_features)))
            zeros = rng.permutation(num_features)[nnz_features:]
            w_star[zeros] = 0

        super().__init__(num_attributes, num_base_features, num_features,
                         w_star)

    def get_feature_radius(self):
        return 1.0

    def phi(self, x, features):
        PATH = "pc-phi.mzn"

        with open(PATH, "wb") as fp:
            fp.write(_TEMPLATE.format(solve=_PHI).encode())

        data = {
            "TRUTH_VALUES": {-1, 1},
            "RAM_DESKTOPS": _RAM_DESKTOPS,
            "RAM_TOWERS": _RAM_TOWERS,
            "N_FEATURES": self.num_features,
            "ACTIVE_FEATURES": set([1]), # doesn't matter
            "W": [0] * self.num_features, # doesn't matter
            "x_type": int(x[0]),
            "x_manufacturer": int(x[1]),
            "x_cpu": int(x[2]),
            "x_monitor": int(x[3]),
            "x_ram": int(x[4]),
            "x_hd": int(x[5]),
            "INPUT_X": [1] * self.num_attributes, # doesn't matter
            "INPUT_PHI": [1] * self.num_features, # doesn't matter
        }

        return super().phi(x, features, PATH, data)

    def infer(self, w, features):
        assert w.shape == (self.num_features,)

        targets = self.enumerate_features(features)
        if (w[targets] == 0).all():
            print("Warning: all-zero w!")

        PATH = "pc-infer.mzn"
        with open(PATH, "wb") as fp:
            fp.write(_TEMPLATE.format(solve=_INFER).encode())

        data = {
            "TRUTH_VALUES": {-1, 1},
            "RAM_DESKTOPS": _RAM_DESKTOPS,
            "RAM_TOWERS": _RAM_TOWERS,
            "N_FEATURES": self.num_features,
            "ACTIVE_FEATURES": {j + 1 for j in targets},
            "W": self.array_to_assignment(w, int),
            "INPUT_X": [1] * self.num_attributes, # doesn't matter
            "INPUT_PHI": [1] * self.num_features, # doesn't matter
        }
        assignments = minizinc(PATH, data=data, output_vars=["x", "objective"],
                               keep=True, parallel=0)

        return self.assignment_to_array(assignments[0]["x"])

    def query_improvement(self, x, features):
        assert x.shape == (self.num_attributes,)

        if self.utility_loss(x, "all") == 0:
            # XXX this is noiseless
            return x

        w_star = np.array(self.w_star)
        if self.noise:
            raise NotImplementedError()
            w_star += self.rng.normal(0, self.noise, size=w_star.shape).astype(np.float32)

        targets = self.enumerate_features(features)
        assert (w_star[targets] != 0).any()

        PATH = "pc-improve.mzn"
        with open(PATH, "wb") as fp:
            fp.write(_TEMPLATE.format(solve=_IMPROVE).encode("utf-8"))

        phi = self.phi(x, "all") # XXX the sum is on ACTIVE_FEATURES anyway

        data = {
            "TRUTH_VALUES": {-1, 1},
            "RAM_DESKTOPS": _RAM_DESKTOPS,
            "RAM_TOWERS": _RAM_TOWERS,
            "N_FEATURES": self.num_features,
            "ACTIVE_FEATURES": {j + 1 for j in targets},
            "W": self.array_to_assignment(w_star, int),
            "INPUT_X": self.array_to_assignment(x, int),
            "INPUT_PHI": self.array_to_assignment(phi, int),
        }
        assignments = minizinc(PATH, data=data, output_vars=["x", "objective"],
                               keep=True, parallel=0)

        x_bar = self.assignment_to_array(assignments[0]["x"])
        assert (x != x_bar).any(), (x, x_bar)

        phi_bar = self.phi(x_bar, "all")
        assert (phi != phi_bar).any()

        utility = np.dot(w_star, self.phi(x, targets))
        utility_bar = np.dot(w_star, self.phi(x_bar, targets))
        assert utility_bar > utility, \
            "u^k({}) = {} is not larger than u^k({}) = {}".format(
                x_bar, utility_bar, x, utility)

        return x_bar
