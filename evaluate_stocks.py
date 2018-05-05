from __future__ import division
from __future__ import absolute_import

from generate_data import *
from misc_utils import make_sure_path_exists, make_buckets
from sklearn.model_selection import train_test_split
from theano_time_corex import *

import pickle
import argparse
import baselines
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nt', type=int, help='number of buckets')
    parser.add_argument('--nv', type=int, help='number of variables')
    parser.add_argument('--m', type=int, help='number of latent factors')
    parser.add_argument('--train_cnt', default=16, type=int, help='number of train samples')
    parser.add_argument('--val_cnt', default=16, type=int, help='number of validation samples')
    parser.add_argument('--test_cnt', default=100, type=int, help='number of test samples')
    parser.add_argument('--eval_iter', type=int, default=1, help='number of evaluation iterations')
    parser.add_argument('--prefix', type=str, default='', help='optional prefix of experiment name')
    parser.add_argument('--data_type', dest='data_type', action='store', default='syn_nglf_buckets',
                        choices=['stock_day', 'stock_week'],
                        help='which dataset to load/create')
    args = parser.parse_args()

    args.train_data, args.val_data, args.test_data = load_stock_data(
        nv=args.nv, train_cnt=args.train_cnt, val_cnt=args.val_cnt, test_cnt=args.test_cnt,
        data_type=args.data_type, stride='full')
    args.ground_truth_covs = None
    args.nt = len(args.train_data)

    ''' Define baselines and the grid of parameters '''
    methods = [
        (baselines.TimeVaryingGraphLasso(name='T-GLASSO'), {
            'lamb': [0.01, 0.03, 0.1, 0.3],
            'beta': [0.03, 0.1, 0.3, 1.0],
            'indexOfPenalty': [1],  # TODO: extend grid of this one
            'max_iter': 100}),

        (baselines.TimeVaryingGraphLasso(name='T-GLASSO (no reg)'), {
            'lamb': [0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
            'beta': [0.0],
            'indexOfPenalty': [1],
            'max_iter': 100}),

        (baselines.TCorex(tcorex=TCorexWeights, name='T-Corex (W, weighted samples)'), {
            'nv': args.nv,
            'n_hidden': [8, 16, 32],
            'max_iter': 500,
            'anneal': True,
            # 'l1': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            'l1': [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            'l2': [],
            'gamma': [1.0, 1.25, 1.5, 2.0, 2.5],
            'reg_type': 'W',
            'init': True
        })
    ]

    exp_name = '{}.m{}.train_cnt{}.val_cnt{}.test_cnt{}'.format(args.data_type, args.m, args.train_cnt,
                                                                args.val_cnt, args.test_cnt)
    exp_name = args.prefix + '.' + exp_name
    results_path = "results/{}.results.json".format(exp_name)
    make_sure_path_exists(results_path)

    results = {}
    for (method, params) in methods[:]:
        name = method.name
        best_params, best_score = method.select(args.train_data, args.val_data, params)
        results[name] = method.evaluate(args.train_data, args.test_data, best_params, args.eval_iter)
        results[name]['best_params'] = best_params
        results[name]['best_val_score'] = best_score

        with open(results_path, 'w') as f:
            json.dump(results, f)

    print("Results are saved in {}".format(results_path))


if __name__ == '__main__':
    main()
