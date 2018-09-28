from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from experiments.generate_data import *
from experiments.utils import make_sure_path_exists
from pytorch_tcorex import *
from experiments import baselines

import argparse
import json
import os

# TODO: fix tcorex lambda=0 and entropy_lambda=0
# TODO: fix gamma range
# TODO: fix l1 for learnable
# TODO: in general copy grid from first_setup

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nt', type=int, help='number of buckets')
    parser.add_argument('--m', type=int, help='number of latent factors')
    parser.add_argument('--bs', type=int, help='block size')
    parser.add_argument('--val_cnt', default=16, type=int, help='number of validation samples')
    parser.add_argument('--test_cnt', default=128, type=int, help='number of test samples')
    parser.add_argument('--snr', type=float, default=5.0, help='signal to noise ratio')
    parser.add_argument('--min_std', type=float, default=0.25, help='minimum x-std')
    parser.add_argument('--max_std', type=float, default=4.0, help='maximum x-std')
    parser.add_argument('--prefix', type=str, default='', help='optional prefix of experiment name')
    parser.add_argument('--data_type', dest='data_type', action='store', default='nglf',
                        choices=['nglf', 'general', 'sparse'], help='which dataset to load/create')
    parser.add_argument('--output_dir', type=str, default='experiments/results/')
    parser.add_argument('--window_size', type=int, default=8)
    parser.add_argument('--stride', type=str, default='full')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='whether to shuffle parent-child relation')
    parser.add_argument('--n_segments', type=int, default=2)
    parser.add_argument('--left', type=int, default=0)
    parser.add_argument('--right', type=int, default=-2)
    parser.set_defaults(shuffle=False)
    args = parser.parse_args()
    args.nv = args.m * args.bs
    print(args)

    ''' Load data '''
    if args.data_type == 'nglf':
        (data, ground_truth_covs) = load_nglf_sudden_change(nv=args.nv, m=args.m, nt=args.nt,
                                                            ns=args.val_cnt + args.test_cnt + 1,
                                                            snr=args.snr, min_std=args.min_std,
                                                            max_std=args.max_std, shuffle=args.shuffle,
                                                            n_segments=args.n_segments)
    else:
        raise ValueError("data_type={} is not implemented yet.".format(args.data_type))
    train_data = [x[-1] for x in data]
    val_data = [x[:args.val_cnt] for x in data]
    test_data = [x[args.val_cnt:args.val_cnt + args.test_cnt] for x in data]

    ''' Define baselines and the grid of parameters '''
    # gamma --- eps means samples only from the current bucket, while 1-eps means all samples
    tcorex_gamma_range = None
    if 0 < args.window_size <= 16:
        tcorex_gamma_range = [0.4, 0.5, 0.6, 0.7, 0.85, 0.9, 0.95]
    if 16 < args.window_size <= 64:
        tcorex_gamma_range = [0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 0.9]
    elif 64 < args.window_size:
        tcorex_gamma_range = [1e-9, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85]

    methods = [
        (baselines.GroundTruth(name='Ground Truth',
                               covs=ground_truth_covs,
                               test_data=test_data), {}),

        (baselines.Diagonal(name='Diagonal'), {
            'window': args.window_size,
            'stride': args.stride
        }),

        (baselines.LedoitWolf(name='Ledoit-Wolf'), {
            'window': args.window_size,
            'stride': args.stride
        }),

        (baselines.OAS(name='Oracle approximating shrinkage'), {
            'window': args.window_size,
            'stride': args.stride
        }),

        (baselines.PCA(name='PCA'), {
            'n_components': [args.m],
            'window': args.window_size,
            'stride': args.stride
        }),

        (baselines.SparsePCA(name='SparsePCA'), {
            'n_components': [args.m],
            'alpha': [0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
            'ridge_alpha': [0.01],
            'tol': 1e-6,
            'max_iter': 500,
            'window': args.window_size,
            'stride': args.stride
        }),

        (baselines.FactorAnalysis(name='Factor Analysis'), {
            'n_components': [args.m],
            'window': args.window_size,
            'stride': args.stride
        }),

        (baselines.GraphLasso(name='Graphical LASSO (sklearn)'), {
            'alpha': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3],
            'mode': 'lars',
            'max_iter': 500,
            'window': args.window_size,
            'stride': args.stride
        }),

        (baselines.LinearCorex(name='Linear CorEx'), {
            'n_hidden': [args.m],
            'max_iter': 500,
            'anneal': True,
            'window': args.window_size,
            'stride': args.stride,
        }),

        (baselines.TimeVaryingGraphLasso(name='T-GLASSO'), {
            'lamb': [0.03, 0.1, 0.3, 1.0, 3.0],
            'beta': [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            'indexOfPenalty': [1],  # NOTE: L2 is very slow and gives bad results
            'max_iter': 500,        # NOTE: checked 1500 no improvement
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
                'l1': [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],  # NOTE: L1 works slightly better
                # 'l2': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            },
            'reg_type': 'W',
            'gamma': 1e-9,
            'init': False,
            'window': args.window_size,
            'stride': args.stride,
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
            'window': args.window_size,
            'stride': args.stride
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
            'window': args.window_size,
            'stride': args.stride,
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
            'window': args.window_size,
            'stride': args.stride,
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
            'window': args.window_size,
            'stride': args.stride
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
            'window': args.window_size,
            'stride': args.stride
        }),

        (baselines.QUIC(name='QUIC'), {
            'lamb': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3],
            'tol': 1e-6,
            'msg': 1,         # NOTE: 0 - no verbosity; 1 - just two lines; 2 - max verbosity
            'max_iter': 100,  # NOTE: tried 500, no improvement,
            'window': args.window_size,
            'stride': args.stride
        }),

        (baselines.BigQUIC(name='BigQUIC'), {
            'lamb': [0.01, 0.03, 0.1, 0.3, 1, 3, 10.0, 30.0],
            'tol': 1e-3,
            'verbose': 1,     # NOTE: 0 - no verbosity; 1 - just two lines; 2 - max verbosity
            'max_iter': 100,  # NOTE: tried 500, no improvement
            'window': args.window_size,
            'stride': args.stride
        })
    ]

    exp_name = 'sudden.{}.nt{}.m{}.bs{}.window{}.stride{}.val_cnt{}.test_cnt{}.snr{:.2f}.min_std{:.2f}.max_std{:.2f}.n_segments{}'.format(
        args.data_type, args.nt, args.m, args.bs, args.window_size, args.stride, args.val_cnt, args.test_cnt,
        args.snr, args.min_std, args.max_std, args.n_segments)
    exp_name = args.prefix + exp_name
    if args.shuffle:
        exp_name += '.shuffle'

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
