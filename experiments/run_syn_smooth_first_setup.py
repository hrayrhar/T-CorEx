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
    parser.add_argument('--train_cnt', default=16, type=int, help='number of train samples')
    parser.add_argument('--val_cnt', default=16, type=int, help='number of validation samples')
    parser.add_argument('--test_cnt', default=128, type=int, help='number of test samples')
    parser.add_argument('--snr', type=float, default=5.0, help='signal to noise ratio')
    parser.add_argument('--min_std', type=float, default=0.25, help='minimum x-std')
    parser.add_argument('--max_std', type=float, default=4.0, help='maximum x-std')
    parser.add_argument('--prefix', type=str, default='', help='optional prefix of experiment name')
    parser.add_argument('--data_type', dest='data_type', action='store', default='nglf',
                        choices=['nglf', 'general', 'sparse'], help='which dataset to load/create')
    parser.add_argument('--output_dir', type=str, default='experiments/results/')
    parser.add_argument('--n_segments', type=int, default=1)
    args = parser.parse_args()
    args.nv = args.m * args.bs
    print(args)

    ''' Load data '''
    if args.data_type == 'nglf':
        (data, ground_truth_covs) = load_nglf_smooth_change(nv=args.nv, m=args.m, nt=args.nt,
                                                            ns=args.train_cnt + args.val_cnt + args.test_cnt,
                                                            snr=args.snr, min_std=args.min_std, max_std=args.max_std,
                                                            n_segments=args.n_segments)
    else:
        raise ValueError("data_type={} is not implemented yet.".format(args.data_type))
    train_data = [x[:args.train_cnt] for x in data]
    val_data = [x[args.train_cnt:args.train_cnt + args.val_cnt] for x in data]
    test_data = [x[-args.test_cnt:] for x in data]

    ''' Define baselines and the grid of parameters '''
    # gamma --- eps means samples only from the current bucket, while 1-eps means all samples
    tcorex_gamma_range = None
    if 0 < args.train_cnt <= 16:
        tcorex_gamma_range = [0.4, 0.5, 0.6, 0.7, 0.85, 0.9, 0.95]
    if 16 < args.train_cnt <= 64:
        tcorex_gamma_range = [0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 0.9]
    elif 64 < args.train_cnt:
        tcorex_gamma_range = [1e-9, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85]

    methods = [
        (baselines.GroundTruth(name='Ground Truth',
                               covs=ground_truth_covs,
                               test_data=test_data), {}),

        (baselines.Diagonal(name='Diagonal'), {}),

        (baselines.LedoitWolf(name='Ledoit-Wolf'), {}),

        (baselines.OAS(name='Oracle approximating shrinkage'), {}),

        (baselines.PCA(name='PCA'), {
            'n_components': [args.m],
        }),

        (baselines.SparsePCA(name='SparsePCA'), {
            'n_components': [args.m],
            'alpha': [0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
            'ridge_alpha': [0.01],
            'tol': 1e-6,
            'max_iter': 500,
        }),

        (baselines.FactorAnalysis(name='Factor Analysis'), {
            'n_components': [args.m],
        }),

        (baselines.GraphLasso(name='Graphical LASSO (sklearn)'), {
            'alpha': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3],
            'mode': 'lars',
            'max_iter': 500,
        }),

        (baselines.LinearCorex(name='Linear CorEx'), {
            'n_hidden': [args.m],
            'max_iter': 500,
            'anneal': True,
        }),

        (baselines.TimeVaryingGraphLasso(name='T-GLASSO'), {
            'lamb': [0.03, 0.1, 0.3, 1.0, 3.0],
            'beta': [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            'indexOfPenalty': [1],  # NOTE: L2 is very slow and gives bad results
            'max_iter': 500,  # NOTE: checked 1500 no improvement
            'lengthOfSlice': args.train_cnt
        }),

        (baselines.TimeVaryingGraphLasso(name='T-GLASSO (no reg)'), {
            'lamb': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
            'beta': [0.0],
            'indexOfPenalty': [1],
            'max_iter': 500,
            'lengthOfSlice': args.train_cnt
        }),

        (baselines.TCorex(tcorex=TCorex, name='T-Corex (simple)'), {
            'nv': args.nv,
            'n_hidden': args.m,
            'max_iter': 500,
            'anneal': True,
            'reg_params': {
                'l1': [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],  # NOTE: L1 works slightly better
                # 'l2': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            },
            'reg_type': 'W',
            'gamma': 1e9,
            'init': False,
        }),

        (baselines.TCorex(tcorex=TCorex, name='T-Corex'), {
            'nv': args.nv,
            'n_hidden': [args.m],
            'max_iter': 500,
            'anneal': True,
            'reg_params': {
                'l1': [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                # 'l2': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            },
            'gamma': tcorex_gamma_range,
            'reg_type': 'W',
            'init': True,
        }),

        (baselines.TCorex(tcorex=TCorex, name='T-Corex (weighted objective)'), {
            'nv': args.nv,
            'n_hidden': [args.m],
            'max_iter': 500,
            'anneal': True,
            'reg_params': {
                'l1': [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                # 'l2': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            },
            'gamma': tcorex_gamma_range,
            'reg_type': 'W',
            'init': True,
            'weighted_obj': True
        }),

        (baselines.TCorex(tcorex=TCorexLearnable, name='T-Corex (learnable)'), {
            'nv': args.nv,
            'n_hidden': [args.m],
            'max_iter': 500,
            'anneal': True,
            'reg_params': {
                'l1': [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                # 'l2': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            },
            'reg_type': 'W',
            'init': True,
            'entropy_lamb': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            'weighted_obj': True
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
        }),

        (baselines.TCorex(tcorex=TCorex, name='T-Corex (no init)'), {
            'nv': args.nv,
            'n_hidden': [args.m],
            'max_iter': 500,
            'anneal': True,
            'reg_params': {
                'l1': [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                # 'l2': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            },
            'gamma': tcorex_gamma_range,
            'reg_type': 'W',
            'init': False,
        }),

        (baselines.QUIC(name='QUIC'), {
            'lamb': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3],
            'tol': 1e-6,
            'msg': 1,  # NOTE: 0 - no verbosity; 1 - just two lines; 2 - max verbosity
            'max_iter': 100,  # NOTE: tried 500, no improvement,
        }),

        (baselines.BigQUIC(name='BigQUIC'), {
            'lamb': [0.01, 0.03, 0.1, 0.3, 1, 3, 10.0, 30.0],
            'tol': 1e-3,
            'verbose': 1,  # NOTE: 0 - no verbosity; 1 - just two lines; 2 - max verbosity
            'max_iter': 100,  # NOTE: tried 500, no improvement
        })
    ]

    exp_name = 'smooth_first_setup.{}.nt{}.m{}.bs{}.train_cnt{}.val_cnt{}.test_cnt{}.snr{:.2f}.min_std{:.2f}.max_std{:.2f}.n_segments{}'.format(
        args.data_type, args.nt, args.m, args.bs, args.train_cnt, args.val_cnt, args.test_cnt,
        args.snr, args.min_std, args.max_std, args.n_segments)
    exp_name = args.prefix + exp_name

    best_results_path = "{}.results.json".format(exp_name)
    best_results_path = os.path.join(args.output_dir, 'best', best_results_path)
    make_sure_path_exists(best_results_path)

    all_results_path = "{}.results.json".format(exp_name)
    all_results_path = os.path.join(args.output_dir, 'all', all_results_path)
    make_sure_path_exists(all_results_path)

    best_results = {}
    all_results = {}
    for (method, params) in methods[:-2]:
        name = method.name
        best_score, best_params, _, _, all_cur_results = method.select(train_data, val_data, params)

        best_results[name] = {}
        best_results[name]['test_score'] = method.evaluate(test_data, best_params)
        best_results[name]['best_params'] = best_params
        best_results[name]['best_val_score'] = best_score

        all_results[name] = all_cur_results

        with open(best_results_path, 'w') as f:
            json.dump(best_results, f)

        with open(all_results_path, 'w') as f:
            json.dump(all_results, f)

    print("Best results are saved in {}".format(best_results_path))
    print("All results are saved in {}".format(all_results_path))

if __name__ == '__main__':
    main()
