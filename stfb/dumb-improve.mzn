% DumbProblem.query_improvement()

int: num_features;
int: max_changes;
set of int: features = 1..num_features;

array[features] of float: w;
array[features] of 0..1: x;

array[features] of var 0..1: x_bar;

solve maximize sum(i in features)(x_bar[i] * w[i]);

constraint
    sum(i in features)(bool2int(x_bar[i] != x[i])) <= max_changes
;
