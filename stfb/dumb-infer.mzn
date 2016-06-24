% DumbProblem.infer()

int: num_features;
set of int: features = 1..num_features;

array[features] of float: w;

array[features] of var 0..1: x;

solve maximize sum(i in features)(x[i] * w[i]);
