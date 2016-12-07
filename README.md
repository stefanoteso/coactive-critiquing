Coactive Critiquing for Constructive Preference Elicitation
===========================================================

A python3 implementation of Coactive Critiquing for preference elicitation.

Coactive critiquing extends coactive learning with support for example
critiquing interaction.

Please see [our paper](http://arxiv.org/abs/1612.01941):

    Stefano Teso, Paolo Dragone, Andrea Passerini. "Coactive Critiquing: Elicitation of Preferences and Features", accepted at AAAI'17, 2017.

## Requirements

The following packages are required:

- [numpy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [pymzn](https://github.com/paolodragone/pymzn), tested with version 0.10.7
- [minizinc](https://minizinc.org), tested with version 2.0.13
- [gecode](http://www.gecode.org/), tested with version 4.4.0
- [cvxpy](www.cvxpy.org), tested with version 0.4.3

## Usage

Type:
```
 $ ./main --help
```
to get the full list of options.

To run the synthetic experiment, type:
```
 $ ./main canvas ${method} -U 20 -T 100 -S ${sparsity} -E 0.1 -s 0 -d -W users/canvas_${sparsity}.pickle
```
where `${method}` can be:
* `pp-attr` for pure Coactive Learning (fixed feature space) over the base features only
* `pp-all` for pure Coactive Learning (fixed feature space) over the full feature space
* `cpp` for Coactive Critiquing (dynamic feature space acquisition)
and `${sparsity}` is the degree of sparsity. The values used in the paper are
`0.2` (sparse case) and `1.0` (non-sparse case).

Similarly, to run the travel planning experiment, type:
```
 $ ./main travel ${method} -U 20 -T 100 -S ${sparsity} -E 0.1 -s 0 -d -W users/travel_${sparsity}_tn_10.pickle
```

## Funding

The project is supported by the CARITRO Foundation through grant 2014.0372.
