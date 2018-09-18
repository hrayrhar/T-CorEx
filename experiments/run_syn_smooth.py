from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from experiments.generate_data import *
from experiments.utils import make_sure_path_exists
from sklearn.model_selection import train_test_split
from pytorch_tcorex import *
from experiments import baselines

import pickle
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nt', type=int, help='number of buckets')
    parser.add_argument('--m', type=int, help='number of latent factors')
    parser.add_argument('--bs', type=int, help='block size')
    parser.add_argument('--val_cnt', default=16, type=int, help='number of validation samples')
    parser.add_argument('--test_cnt', default=128, type=int, help='number of test samples')
    parser.add_argument('--snr', type=float, default=5.0, help='signal to noise ratio')
    parser.add_argument('--min_var', type=float, default=0.25, help='minimum x-variance')
    parser.add_argument('--max_var', type=float, default=4.0, help='maximum x-variance')
    parser.add_argument('--prefix', type=str, default='', help='optional prefix of experiment name')
    parser.add_argument('--data_type', dest='data_type', action='store', default='nglf',
                        choices=['nglf', 'general', 'sparse'], help='which dataset to load/create')
    parser.add_argument('--output_dir', type=str, default='experiments/results/')
    parser.add_argument('--window_size', type=int, default=8)
    args = parser.parse_args()
    args.nv = args.m * args.bs
    print(args)

    ''' Load data '''
    (data, args.ground_truth_covs) = load_nglf_smooth_change(nv=args.nv, m=args.m, nt=args.nt,
                                                             ns=args.val_cnt + args.test_cnt + 1,
                                                             snr=args.snr, min_var=args.min_var, max_var=args.max_var)
    args.train_data = [x[-1] for x in data]
    args.val_data = [x[:args.val_cnt] for x in data]
    args.test_data = [x[args.val_cnt:args.val_cnt+args.test_cnt] for x in data]

    ''' Define baselines and the grid of parameters '''
    # gamma --- eps means samples only from the current bucket, while 1-eps means all samples
    tcorex_gamma_range = None
    if 0 < args.window_size <= 16:
        tcorex_gamma_range = [0.5, 0.6, 0.7, 0.85, 0.9, 0.95]
    if 16 < args.window_size <= 64:
        tcorex_gamma_range = [0.3, 0.5, 0.6, 0.7, 0.85, 0.9]
    elif 64 < args.window_size:
        tcorex_gamma_range = [1e-9, 0.3, 0.5, 0.6, 0.7, 0.85]

    # TODO: use 'half' ? instead of 'full' ?
    methods = [
        (baselines.GroundTruth(name='Ground Truth',
                               covs=args.ground_truth_covs,
                               test_data=args.test_data), {}),

        (baselines.Diagonal(name='Diagonal'), {
            'window': args.window_size,
            'stride': 'full'
        }),

        (baselines.LedoitWolf(name='Ledoit-Wolf'), {
            'window': args.window_size,
            'stride': 'full'
        }),

        (baselines.OAS(name='Oracle approximating shrinkage'), {
            'window': args.window_size,
            'stride': 'full'
        }),

        (baselines.PCA(name='PCA'), {
            'n_components': [args.m],
            'window': args.window_size,
            'stride': 'full'
        }),

        (baselines.SparsePCA(name='SparsePCA'), {
            'n_components': [args.m],
            'alpha': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
            'ridge_alpha': [0.01],
            'tol': 1e-6,
            'max_iter': 500,
            'window': args.window_size,
            'stride': 'full'
        }),

        (baselines.FactorAnalysis(name='Factor Analysis'), {
            'n_components': [args.m],
            'window': args.window_size,
            'stride': 'full'
        }),

        (baselines.GraphLasso(name='Graphical LASSO (sklearn)'), {
            'alpha': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3],
            'mode': 'lars',
            'max_iter': 500,
            'window': args.window_size,
            'stride': 'full'
        }),

        (baselines.LinearCorex(name='Linear CorEx'), {
            'n_hidden': [args.m],
            'max_iter': 500,
            'anneal': True,
            'window': args.window_size,
            'stride': 'full',
        }),

        (baselines.TimeVaryingGraphLasso(name='T-GLASSO'), {
            'lamb': [0.03, 0.1, 0.3, 1.0, 3.0],
            'beta': [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            'indexOfPenalty': [1],  # NOTE: L2 is very slow and gives bad results
            'max_iter': 500,  # NOTE: checked 1500 no improvement
            'lengthOfSlice': args.window_size,
        }),

        (baselines.TimeVaryingGraphLasso(name='T-GLASSO (no reg)'), {
            'lamb': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
            'beta': [0.0],
            'indexOfPenalty': [1],
            'max_iter': 500,
            'lengthOfSlice': args.window_size,
        }),

        (baselines.TCorex(tcorex=TCorex, name='T-Corex (simple)'), {
            'nv': args.nv,
            'n_hidden': args.m,
            'max_iter': 500,
            'anneal': True,
            'reg_params': {
                'l1': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                # 'l2': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            },
            'reg_type': 'W',
            'gamma': 1e9,
            'init': False,
            'window': args.window_size,
            'stride': 'full',
        }),

        (baselines.TCorex(tcorex=TCorex, name='T-Corex'), {
            'nv': args.nv,
            'n_hidden': [args.m],
            'max_iter': 500,
            'anneal': True,
            'reg_params': {
                'l1': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],  # NOTE: L1 works slightly better
                # 'l2': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            },
            'gamma': tcorex_gamma_range,
            'reg_type': 'W',
            'init': True,
            'window': args.window_size,
            'stride': 'full'
        }),

        (baselines.TCorex(tcorex=TCorex, name='T-Corex (no reg)'), {
            'nv': args.nv,
            'n_hidden': [args.m],
            'max_iter': 500,
            'anneal': True,
            'l1': 0.0,
            'l2': 0.0,
            'gamma': tcorex_gamma_range,
            'reg_type': 'W',
            'init': True,
            'window': args.window_size,
            'stride': 'full'
        }),

        (baselines.TCorex(tcorex=TCorex, name='T-Corex (no init)'), {
            'nv': args.nv,
            'n_hidden': [args.m],
            'max_iter': 500,
            'anneal': True,
            'reg_params': {
                'l1': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                # 'l2': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            },
            'gamma': tcorex_gamma_range,
            'reg_type': 'W',
            'init': False,
            'window': args.window_size,
            'stride': 'full'
        }),

        (baselines.QUIC(name='QUIC'), {
            'lamb': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3],
            'tol': 1e-6,
            'msg': 1,  # NOTE: 0 - no verbosity; 1 - just two lines; 2 - max verbosity
            'max_iter': 100,  # NOTE: tried 500, no improvement,
            'window': args.window_size,
            'stride': 'full'
        }),

        (baselines.BigQUIC(name='BigQUIC'), {
            'lamb': [0.01, 0.03, 0.1, 0.3, 1, 3, 10.0, 30.0],
            'tol': 1e-3,
            'verbose': 1,  # NOTE: 0 - no verbosity; 1 - just two lines; 2 - max verbosity
            'max_iter': 100,  # NOTE: tried 500, no improvement
            'window': args.window_size,
            'stride': 'full'
        })
    ]

    exp_name = 'smooth.{}.nt{}.m{}.bs{}.window{}.val_cnt{}.test_cnt{}.snr{:.2f}.min_var{:.2f}.max_var{:.2f}'.format(
        args.data_type, args.nt, args.m, args.bs, args.window_size, args.val_cnt, args.test_cnt,
        args.snr, args.min_var, args.max_var)
    exp_name = args.prefix + exp_name
    results_path = "{}.results.json".format(exp_name)
    results_path = os.path.join(args.output_dir, results_path)
    make_sure_path_exists(results_path)

    results = {}
    for (method, params) in methods[0:-1]:
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
