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
    parser.add_argument('--test_cnt', default=100, type=int, help='number of test samples')
    parser.add_argument('--snr', type=float, default=None, help='signal to noise ratio')
    parser.add_argument('--min_cor', type=float, default=0.8, help='minimum correlation between a child and parent')
    parser.add_argument('--max_cor', type=float, default=1.0, help='minimum correlation between a child and parent')
    parser.add_argument('--min_var', type=float, default=1.0, help='minimum x-variance')
    parser.add_argument('--max_var', type=float, default=1.0, help='maximum x-variance')
    parser.add_argument('--prefix', type=str, default='', help='optional prefix of experiment name')
    parser.add_argument('--data_type', dest='data_type', action='store', default='syn_nglf_buckets_smooth',
                        choices=['syn_nglf_buckets_smooth'], help='which dataset to load/create')
    args = parser.parse_args()
    args.nv = args.m * args.bs

    ''' Load data '''
    (data, args.ground_truth_covs) = generate_nglf_smooth(nv=args.nv, m=args.m, nt=args.nt,
                                                          ns=args.train_cnt + args.val_cnt + args.test_cnt,
                                                          snr=args.snr, min_cor=args.min_cor, max_cor=args.max_cor,
                                                          min_var=args.min_var, max_var=args.max_var)
    args.train_data = [x[:args.train_cnt] for x in data]
    args.val_data = [x[args.train_cnt:args.train_cnt + args.val_cnt] for x in data]
    args.test_data = [x[-args.test_cnt:] for x in data]

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
            'max_iter': 100}),

        (baselines.LinearCorex(name='Linear CorEx (applied bucket-wise)'), {
            'n_hidden': [args.m],
            'max_iter': 500,
            'anneal': True}),

        (baselines.LinearCorexWholeData(name='Linear CorEx (applied on whole data)'), {
            'n_hidden': [args.m],
            'max_iter': 500,
            'anneal': True}),

        (baselines.TimeVaryingGraphLasso(name='T-GLASSO (L1)'), {
            'lamb': [0.01, 0.03, 0.1, 0.3],
            'beta': [0.03, 0.1, 0.3, 1.0, 3.0],
            'indexOfPenalty': [1],
            'max_iter': 100}),

        # (baselines.TimeVaryingGraphLasso(name='T-GLASSO (no reg)'), {
        #     'lamb': [0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
        #     'beta': [0.0],
        #     'indexOfPenalty': [2],
        #     'max_iter': 10}),  # NOTE: was 100

        (baselines.TCorex(tcorex=TCorex, name='T-Corex (W, L1)'), {
            'nv': args.nv,
            'n_hidden': args.m,
            'max_iter': 500,
            'anneal': True,
            'reg_params': {
                'l1': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                # 'l2': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                # 'l2': [0, 0.001, 0.003],
            },
            'reg_type': 'W'
        }),

        # (baselines.TCorex(tcorex=TCorexPrior1, name='T-Corex + priors (W, method 1)'), {
        #     'nv': args.nv,
        #     'n_hidden': [args.m],
        #     'max_iter': 500,
        #     'anneal': True,
        #     'reg_params': {
        #         'l1': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        #         # 'l2': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        #         # 'l2': [0, 0.001, 0.003],
        #     },
        #     'lamb': [0.0, 0.5, 0.9, 0.99],
        #     'reg_type': 'W',
        #     'init': True
        # }),

        # (baselines.TCorex(tcorex=TCorexPrior2, name='T-Corex + priors (W, method 2)'), {
        #     'nv': args.nv,
        #     'n_hidden': [args.m],
        #     'max_iter': 500,
        #     'anneal': True,
        #     'reg_params': {
        #         'l1': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        #         # 'l2': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        #         # 'l2': [0, 0.001, 0.003],
        #     },
        #     'lamb': [0.0, 0.5, 0.9, 0.99],
        #     'reg_type': 'W',
        #     'init': True
        # }),

        # (baselines.TCorex(tcorex=TCorexPrior2Weights, name='T-Corex + priors (W, method 2, weighted samples)'), {
        #     'nv': args.nv,
        #     'n_hidden': [args.m],
        #     'max_iter': 500,
        #     'anneal': True,
        #     'reg_params': {
        #         'l1': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        #         # 'l2': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        #         # 'l2': [0, 0.001, 0.003],
        #     },
        #     'lamb': [0.0, 0.5, 0.9, 0.99],
        #     'gamma': tcorex_gamma_range,
        #     'reg_type': 'W',
        #     'init': True
        # }),

        (baselines.TCorex(tcorex=TCorexWeights, name='T-Corex (W, L1, weighted samples)'), {
            'nv': args.nv,
            'n_hidden': [args.m],
            'max_iter': 500,
            'anneal': True,
            'reg_params': {
                'l1': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                # 'l2': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                # 'l2': [0, 0.001, 0.003],
            },
            'gamma': tcorex_gamma_range,
            'reg_type': 'W',
            'init': True
        }),

        # (baselines.TCorex(tcorex=TCorexWeights, name='T-Corex (W, weighted samples, no init)'), {
        #     'nv': args.nv,
        #     'n_hidden': [args.m],
        #     'max_iter': 500,
        #     'anneal': True,
        #     'reg_params': {
        #         'l1': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        #         # 'l2': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        #         # 'l2': [0, 0.001, 0.003],
        #     },
        #     'gamma': tcorex_gamma_range,
        #     'reg_type': 'W',
        #     'init': False
        # }),

        # (baselines.TCorex(tcorex=TCorexWeightedObjective, name='T-Corex (W, weighted objective)'), {
        #     'nv': args.nv,
        #     'n_hidden': [args.m],
        #     'max_iter': 500,
        #     'anneal': True,
        #     'reg_params': {
        #         'l1': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        #         # 'l2': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        #         # 'l2': [0, 0.001, 0.003],
        #     },
        #     'gamma': tcorex_gamma_range,
        #     'reg_type': 'W',
        #     'init': True
        # }),

        # (baselines.TCorex(tcorex=TCorexWeightsMod, name='T-Corex (W, weighted samples, modified)'), {
        #     'nv': args.nv,
        #     'n_hidden': [args.m],
        #     'max_iter': 500,
        #     'anneal': True,
        #     'reg_params': {
        #         'l1': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        #         # 'l2': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        #         # 'l2': [0, 0.001, 0.003],
        #     },
        #     'gamma': tcorex_gamma_range,
        #     'reg_type': 'W',
        #     'init': True,
        #     'sample_cnt': 256
        # })
    ]

    exp_name = '{}.nt{}.m{}.bs{}.train_cnt{}.val_cnt{}.test_cnt{}'.format(
        args.data_type, args.nt, args.m, args.bs, args.train_cnt, args.val_cnt, args.test_cnt)
    if args.snr:
        suffix = '.snr{:.2f}'.format(args.snr)
    else:
        suffix = '.min_cor{:.2f}.max_cor{:.2f}'.format(args.min_cor, args.max_cor)
    exp_name = args.prefix + exp_name + suffix
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
