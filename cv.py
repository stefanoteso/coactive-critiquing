#!/usr/bin/env python3

import sys
import main
import pickle
from collections import namedtuple

problem, sparsity, noise = sys.argv[1:4]
gammas = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

aucs = {}

for gamma in gammas:
    print('gamma={}'.format(gamma))
    args = {
        'problem': problem,
        'method': 'cpp',
        'num_users': 20,
        'max_iters': 100,
        'sparsity': float(sparsity),
        'noise': float(noise),
        'update': 'perceptron',
        'perturbation': 0.0,
        'gamma': gamma,
        'weights': None,
        'seed': 0,
        'debug': True,
        'verbose': False
    }
    
    Args = namedtuple('Args', args.keys())
    args = Args(**args)
    main.main(args)
    
    result_file_name = main._get_experiment_path(args)
    matrix_loss = pickle.load(open(result_file_name, 'rb'))['matrix_loss']
    auc = matrix_loss.mean(axis=0).sum()
    aucs[str(gamma)] = auc
    print('gamma={}, auc={}'.format(gamma, auc))

best_gamma = max(gammas, gammas.get)
print('Best gamma: problem={}, sparsity={}, noise={}, gamma={}, auc={}'.format(best_gamma, gammas[best_gamma]))
