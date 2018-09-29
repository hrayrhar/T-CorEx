from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from experiments.generate_data import load_sp500
from experiments.utils import make_sure_path_exists
from experiments import baselines
from pytorch_tcorex import *

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nt', type=int, help='number of buckets')
    parser.add_argument('--train_cnt', default=16, type=int, help='number of train samples')
    parser.add_argument('--val_cnt', default=4, type=int, help='number of validation samples')
    parser.add_argument('--test_cnt', default=4, type=int, help='number of test samples')
    parser.add_argument('--commodities', dest='commodities', action='store_true',
                        help='whether to include commodity prices too')
    parser.add_argument('--log_return', dest='log_return', action='store_true',
                        help='whether to take log returns or normal returns')
    parser.add_argument('--start_date', type=str, default='2000-01-01')
    parser.add_argument('--end_date', type=str, default='2018-01-01')
    parser.add_argument('--noise_var', type=float, default=1e-4,
                        help='variance of Gaussian noise that will be added to time series')
    parser.add_argument('--prefix', type=str, default='', help='optional prefix of experiment name')
    parser.add_argument('--output_dir', type=str, default='experiments/results/')
    parser.add_argument('--left', type=int, default=0)
    parser.add_argument('--right', type=int, default=-2)
    parser.set_defaults(commodities=False)
    parser.set_defaults(log_return=True)
    args = parser.parse_args()

    ''' Load data '''
    train_data, val_data, test_data, _, _ = load_sp500(
        train_cnt=args.train_cnt, val_cnt=args.val_cnt, test_cnt=args.test_cnt, commodities=args.commodities,
        log_return=args.log_return, start_date=args.start_date, end_date=args.end_date, noise_var=args.noise_var)

    # take last nt time steps
    nv = train_data[0].shape[-1]
    train_data = train_data[-args.nt:]
    val_data = val_data[-args.nt:]
    test_data = test_data[-args.nt:]

    ''' Define baselines and the grid of parameters '''
    # gamma --- eps means samples only from the current bucket, while 1-eps means all samples
    tcorex_gamma_range = None
    if 0 < args.train_cnt <= 16:
        tcorex_gamma_range = [0.3, 0.4, 0.5, 0.6, 0.7]
    if 16 < args.train_cnt <= 32:
        tcorex_gamma_range = [0.1, 0.3, 0.4, 0.5, 0.6]
    if 32 < args.train_cnt <= 64:
        tcorex_gamma_range = [1e-9, 0.1, 0.3, 0.4, 0.5]
    elif 64 < args.train_cnt:
        tcorex_gamma_range = [1e-9, 0.1, 0.3]

    n_hidden_grid = [11]  # the number of sectors

    methods = [
        (baselines.Diagonal(name='Diagonal'), {}),

        (baselines.LedoitWolf(name='Ledoit-Wolf'), {}),

        (baselines.OAS(name='Oracle approximating shrinkage'), {}),

        (baselines.PCA(name='PCA'), {
            'n_components': n_hidden_grid
        }),

        (baselines.SparsePCA(name='SparsePCA'), {
            'n_components': n_hidden_grid,
            'alpha': [0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
            'ridge_alpha': [0.01],
            'tol': 1e-6,
            'max_iter': 500,
        }),

        (baselines.FactorAnalysis(name='Factor Analysis'), {
            'n_components': n_hidden_grid
        }),

        (baselines.GraphLasso(name='Graphical LASSO (sklearn)'), {
            'alpha': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3],
            'mode': 'lars',
            'max_iter': 500,
        }),

        (baselines.LinearCorex(name='Linear CorEx (applied bucket-wise)'), {
            'n_hidden': n_hidden_grid,
            'max_iter': 500,
            'anneal': True
        }),

        (baselines.TimeVaryingGraphLasso(name='T-GLASSO'), {
            'lamb': [0.03, 0.1, 0.3, 1.0, 3.0],
            'beta': [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            'indexOfPenalty': [1],  # NOTE: L2 is very slow and gives bad results
            'max_iter': 500,        # NOTE: checked 1500 no improvement
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
            'nv': nv,
            'n_hidden': n_hidden_grid,
            'max_iter': 500,
            'anneal': True,
            'reg_params': {
                'l1': [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],  # NOTE: L1 works slightly better
                # 'l2': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            },
            'reg_type': 'W',
            'gamma': 1e-9,
            'init': False,
        }),

        (baselines.TCorex(tcorex=TCorex, name='T-Corex'), {
            'nv': nv,
            'n_hidden': n_hidden_grid,
            'max_iter': 500,
            'anneal': True,
            'reg_params': {
                'l1': [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                # 'l2': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            },
            'gamma': tcorex_gamma_range,
            'reg_type': 'W',
            'init': True,
        }),

        (baselines.TCorex(tcorex=TCorex, name='T-Corex (weighted objective)'), {
            'nv': nv,
            'n_hidden': n_hidden_grid,
            'max_iter': 500,
            'anneal': True,
            'reg_params': {
                'l1': [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                # 'l2': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            },
            'gamma': tcorex_gamma_range,
            'reg_type': 'W',
            'init': True,
            'weighted_obj': True
        }),

        (baselines.TCorex(tcorex=TCorex, name='T-Corex (no reg)'), {
            'nv': nv,
            'n_hidden': n_hidden_grid,
            'max_iter': 500,
            'anneal': True,
            'l1': 0.0,
            'l2': 0.0,
            'gamma': tcorex_gamma_range,
            'reg_type': 'W',
            'init': True,
        }),

        (baselines.TCorex(tcorex=TCorex, name='T-Corex (no init)'), {
            'nv': nv,
            'n_hidden': n_hidden_grid,
            'max_iter': 500,
            'anneal': True,
            'reg_params': {
                'l1': [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                # 'l2': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            },
            'gamma': tcorex_gamma_range,
            'reg_type': 'W',
            'init': False,
        }),

        (baselines.TCorex(tcorex=TCorexLearnable, name='T-Corex (learnable)'), {
            'nv': nv,
            'n_hidden': n_hidden_grid,
            'max_iter': 500,
            'anneal': True,
            'reg_params': {
                'l1': [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                # 'l2': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            },
            'reg_type': 'W',
            'init': True,
            'entropy_lamb': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            'weighted_obj': True
        }),

        (baselines.QUIC(name='QUIC'), {
            'lamb': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3],
            'tol': 1e-6,
            'msg': 1,         # NOTE: 0 - no verbosity; 1 - just two lines; 2 - max verbosity
            'max_iter': 100,  # NOTE: tried 500, no improvement,
        }),

        (baselines.BigQUIC(name='BigQUIC'), {
            'lamb': [0.3, 1, 3, 10.0, 30.0],
            'tol': 1e-3,
            'verbose': 0,     # NOTE: 0 - no verbosity; 1 - just two lines; 2 - max verbosity
            'max_iter': 100,  # NOTE: tried 500, no improvement
        })
    ]

    exp_name = 'stocks_first_setup.nt{}.train_cnt{}.val_cnt{}.test_cnt{}.start_date{}.end_date{}.noise_var{}'.format(
        args.nt, args.train_cnt, args.val_cnt, args.test_cnt, args.start_date, args.end_date, args.noise_var)
    exp_name = args.prefix + exp_name
    if args.commodities:
        exp_name += '.commodities'
    if args.log_return:
        exp_name += '.log_return'

    best_results_path = "{}.results.json".format(exp_name)
    best_results_path = os.path.join(args.output_dir, 'best', best_results_path)
    make_sure_path_exists(best_results_path)

    all_results_path = "{}.results.json".format(exp_name)
    all_results_path = os.path.join(args.output_dir, 'all', all_results_path)
    make_sure_path_exists(all_results_path)

    best_results = {}
    all_results = {}

    # read previously stored values
    if os.path.exists(best_results_path):
        with open(best_results_path, 'r') as f:
            best_results = json.load(f)
    if os.path.exists(all_results_path):
        with open(all_results_path, 'r') as f:
            all_results = json.load(f)

    for (method, params) in methods[args.left:args.right]:
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
