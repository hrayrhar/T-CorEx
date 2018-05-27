from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

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
    parser.add_argument('--m', type=int, help='number of latent factors')
    parser.add_argument('--bs', type=int, help='block size')
    parser.add_argument('--train_cnt', default=16, type=int, help='number of train samples')
    parser.add_argument('--val_cnt', default=16, type=int, help='number of validation samples')
    parser.add_argument('--test_cnt', default=1000, type=int, help='number of test samples')
    parser.add_argument('--snr', type=float, default=5.0, help='signal to noise ratio')
    parser.add_argument('--min_var', type=float, default=0.25, help='minimum x-variance')
    parser.add_argument('--max_var', type=float, default=4.0, help='maximum x-variance')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='whether to shuffle parent-child relation')
    parser.add_argument('--prefix', type=str, default='', help='optional prefix of experiment name')
    parser.add_argument('--data_type', dest='data_type', action='store', default='nglf_sudden_change',
                        choices=['nglf_sudden_change', 'general_sudden_change'],
                        help='which dataset to load/create')
    parser.set_defaults(shuffle=False)
    args = parser.parse_args()
    args.nv = args.m * args.bs
    print(args)

    ''' Load data '''
    args.train_data, args.val_data, args.test_data, args.ground_truth_covs = load_sudden_change(
        nv=args.nv, m=args.m, nt=args.nt, train_cnt=args.train_cnt, val_cnt=args.val_cnt, test_cnt=args.test_cnt,
        snr=args.snr, min_var=args.min_var, max_var=args.max_var, nglf=(args.data_type == 'nglf_sudden_change'),
        shuffle=args.shuffle)

    ''' Define baselines and the grid of parameters '''
    if args.train_cnt < 32:
        tcorex_gamma_range = [1.25, 1.5, 2.0, 2.5, 1e5]
    elif args.train_cnt < 64:
        tcorex_gamma_range = [1.5, 2.0, 2.5, 1e5]
    elif args.train_cnt < 128:
        tcorex_gamma_range = [2.0, 2.5, 1e5]
    else:
        tcorex_gamma_range = [2.5, 1e5]

    methods = [
        (baselines.GroundTruth(name='Ground Truth',
                               covs=args.ground_truth_covs,
                               test_data=args.test_data), {}),

        (baselines.Diagonal(name='Diagonal'), {}),

        (baselines.LedoitWolf(name='Ledoit-Wolf'), {}),

        (baselines.OAS(name='Oracle approximating shrinkage'), {}),

        (baselines.PCA(name='PCA'), {'n_components': [args.m]}),

        (baselines.FactorAnalysis(name='Factor Analysis'), {'n_components': [args.m]}),

        (baselines.GraphLasso(name='Graphical LASSO (sklearn)'), {
            'alpha': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3],
            'mode': 'lars',
            'max_iter': 500}),

        (baselines.LinearCorex(name='Linear CorEx (applied bucket-wise)'), {
            'n_hidden': [args.m],
            'max_iter': 500,
            'anneal': True}),

        (baselines.LinearCorexWholeData(name='Linear CorEx (applied on whole data)'), {
            'n_hidden': [args.m],
            'max_iter': 500,
            'anneal': True}),

        (baselines.TimeVaryingGraphLasso(name='T-GLASSO'), {
            'lamb': [0.03, 0.1, 0.3, 1.0, 3.0],
            'beta': [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            'indexOfPenalty': [1],
            'max_iter': 500}),  # NOTE: checked 1500 no improvement

        (baselines.TimeVaryingGraphLasso(name='T-GLASSO (no reg)'), {
            'lamb': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
            'beta': [0.0],
            'indexOfPenalty': [1],
            'max_iter': 500}),

        (baselines.TCorex(tcorex=TCorex, name='T-Corex (W)'), {
            'nv': args.nv,
            'n_hidden': args.m,
            'max_iter': 500,
            'anneal': True,
            'reg_params': {
                'l1': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                # 'l1': [0, 0.001, 0.003],
                'l2': []
            },
            'reg_type': 'W'
        }),

        (baselines.TCorex(tcorex=TCorexWeights, name='T-Corex (W, weighted samples)'), {
            'nv': args.nv,
            'n_hidden': [args.m],
            'max_iter': 500,
            'anneal': True,
            'reg_params': {
                # 'l1': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                'l1': [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
                'l2': [],
            },
            'gamma': tcorex_gamma_range,
            'reg_type': 'W',
            'init': True
        }),

        (baselines.TCorex(tcorex=TCorexWeights, name='T-Corex (W, weighted samples, no reg)'), {
            'nv': args.nv,
            'n_hidden': [args.m],
            'max_iter': 500,
            'anneal': True,
            'reg_params': {
                'l1': [0.0],
                'l2': [],
            },
            'gamma': tcorex_gamma_range,
            'reg_type': 'W',
            'init': True
        }),

        (baselines.TCorex(tcorex=TCorexWeights, name='T-Corex (W, weighted samples, no init)'), {
            'nv': args.nv,
            'n_hidden': [args.m],
            'max_iter': 500,
            'anneal': True,
            'reg_params': {
                # 'l1': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                'l1': [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
                'l2': [],
            },
            'gamma': tcorex_gamma_range,
            'reg_type': 'W',
            'init': False
        }),

        (baselines.QUIC(name='QUIC'), {
            'lamb': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3],
            'tol': 1e-6,
            'msg': 1,  # NOTE: 0 - no verbosity; 1 - just two lines; 2 - max verbosity
            'max_iter': 100  # NOTE: tried 500, no improvement
        })
    ]

    exp_name = '{}.nt{}.m{}.bs{}.train_cnt{}.val_cnt{}.test_cnt{}.snr{:.2f}.min_var{:.2f}.max_var{:.2f}'.format(
        args.data_type, args.nt, args.m, args.bs, args.train_cnt, args.val_cnt, args.test_cnt,
        args.snr, args.min_var, args.max_var)
    exp_name = args.prefix + exp_name
    results_path = "results/{}.results.json".format(exp_name)
    make_sure_path_exists(results_path)

    results = {}
    for (method, params) in methods[:]:
        name = method.name
        best_score, best_params, _, _ = method.select(args.train_data, args.val_data, params)
        results[name] = {}
        results[name]['test_score'] = method.evaluate(args.test_data, best_params)
        results[name]['best_params'] = best_params
        results[name]['best_val_score'] = best_score

        with open(results_path, 'w') as f:
            json.dump(results, f)

    print("Results are saved in {}".format(results_path))


if __name__ == '__main__':
    main()
