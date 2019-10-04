from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from tcorex.experiments.data import load_sp500
from tcorex.experiments.misc import make_sure_path_exists
from tcorex.experiments import baselines
from tcorex import TCorex

from cvxopt import matrix, solvers

import numpy as np
import argparse
import pickle
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nt', type=int, default=10, help='number of train time periods')
    parser.add_argument('--train_cnt', default=12, type=int, help='number of train samples')
    parser.add_argument('--val_cnt', default=3, type=int, help='number of validation samples')
    parser.add_argument('--start_period', type=int, default=1)
    parser.add_argument('--noise_var', type=float, default=1e-4,
                        help='variance of Gaussian noise that will be added to time series')
    parser.add_argument('--prefix', type=str, default='', help='optional prefix of experiment name')
    parser.add_argument('--output_dir', type=str, default='outputs/portfolio/')
    parser.add_argument('--left', type=int, default=0)
    parser.add_argument('--right', type=int, default=-2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    print(args)

    ''' Load data '''
    train_data, val_data, _, _, _, df_index = load_sp500(
        train_cnt=args.train_cnt, val_cnt=args.val_cnt, test_cnt=0, commodities=False,
        log_return=False, start_date='2000-01-01', end_date='2016-01-01', noise_var=args.noise_var,
        standardize=False, return_index=True, seed=args.seed)

    # Take last nt+1 time steps. Use first nt of them for training / validation.
    # The last time period is used for testing.
    start_period = args.start_period
    test_period = args.start_period + args.nt
    nv = train_data[0].shape[-1]

    test_data = np.concatenate([train_data[test_period], val_data[test_period]], axis=0)
    train_data = train_data[start_period:test_period]
    val_data = val_data[start_period:test_period]

    start_date = df_index[start_period * (args.train_cnt + args.val_cnt)].date()
    end_date = df_index[(test_period + 1) * (args.train_cnt + args.val_cnt) - 1].date()

    print("Number of train/val time periods: {}".format(len(train_data)))
    print("Start date: {}".format(start_date))
    print("End date: {}".format(end_date))
    print("Test data shape: {}".format(test_data.shape))

    ''' Define baselines and the grid of parameters '''
    # gamma --- eps means samples only from the current bucket, while 1 means all samples
    tcorex_gamma_range = [0.8, 0.9]
    n_hidden_grid = [16, 32]  # [16, 32, 64]

    methods = [
        (baselines.LedoitWolf(name='Ledoit-Wolf'), {}),

        (baselines.LinearCorex(name='Linear CorEx'), {
            'n_hidden': n_hidden_grid,
            'max_iter': 500,
            'anneal': True
        }),

        (baselines.TCorex(tcorex=TCorex, name='T-Corex'), {
            'nv': nv,
            'n_hidden': n_hidden_grid,
            'max_iter': 500,
            'anneal': True,
            'reg_params': {
                'l1': [0.3, 1.0, 3.0, 10.0, 30.0],  # [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            },
            'gamma': tcorex_gamma_range,
            'reg_type': 'W',
            'init': True,
            'device': args.device,
            'verbose': 1
        }),

        (baselines.TimeVaryingGraphLasso(name='T-GLASSO'), {
            'lamb': [0.03, 0.1, 0.3, 1.0, 3.0],
            'beta': [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            'indexOfPenalty': [1],  # NOTE: L2 is very slow and gives bad results
            'max_iter': 500,        # NOTE: checked 1500 no improvement
            'lengthOfSlice': args.train_cnt
        }),

        (baselines.LTGL(name='LTGL'), {
            'alpha': [0.3, 1.0, 3.0, 10.0],
            'tau': [30.0, 100.0, 300.0, 1e3],
            'beta': [10.0, 30.0, 100.0],
            'psi': 'l1',
            'eta': [0.3, 1.0, 3.0],
            'phi': 'l1',      # NOTE: tried Laplacian and l2 too no improvement
            'rho': 1.0 / np.sqrt(args.train_cnt),
            'max_iter': 500,  # NOTE: tried 1000 no improvement
            'verbose': False
        })
    ]

    exp_name = 'nt{}.train_cnt{}.val_cnt{}.start_date{}.end_date{}.noise_var{}'.format(
        args.nt, args.train_cnt, args.val_cnt, start_date, end_date, args.noise_var)
    exp_name = args.prefix + exp_name

    best_results_path = "{}.results.json".format(exp_name)
    best_results_path = os.path.join(args.output_dir, 'best', best_results_path)
    make_sure_path_exists(best_results_path)
    best_results = {}
    if os.path.exists(best_results_path):
        with open(best_results_path, 'r') as f:
            best_results = json.load(f)

    all_results_path = "{}.results.json".format(exp_name)
    all_results_path = os.path.join(args.output_dir, 'all', all_results_path)
    make_sure_path_exists(all_results_path)
    all_results = {}
    if os.path.exists(all_results_path):
        with open(all_results_path, 'r') as f:
            all_results = json.load(f)

    mu_path = "{}.pkl".format(exp_name)
    mu_path = os.path.join(args.output_dir, 'mu', mu_path)
    make_sure_path_exists(mu_path)
    mus = {}
    if os.path.exists(mu_path):
        with open(mu_path, 'rb') as f:
            mus = pickle.load(f)

    sigma_path = "{}.pkl".format(exp_name)
    sigma_path = os.path.join(args.output_dir, 'sigma', sigma_path)
    make_sure_path_exists(sigma_path)
    sigmas = {}
    if os.path.exists(sigma_path):
        with open(sigma_path, 'rb') as f:
            sigmas = pickle.load(f)

    qp_solutions_path = "{}.pkl".format(exp_name)
    qp_solutions_path = os.path.join(args.output_dir, 'qp_solution', qp_solutions_path)
    make_sure_path_exists(qp_solutions_path)
    qp_solutions = {}
    if os.path.exists(qp_solutions_path):
        with open(qp_solutions_path, 'rb') as f:
            qp_solutions = pickle.load(f)

    test_data_path = "{}.txt".format(exp_name)
    test_data_path = os.path.join(args.output_dir, 'test_data', test_data_path)
    make_sure_path_exists(test_data_path)
    np.savetxt(test_data_path, test_data)

    for (method, params) in methods[args.left:args.right]:
        name = method.name
        best_score, best_params, best_covs, best_method, all_cur_results =\
            method.select(train_data, val_data, params)

        mu = np.mean(train_data[-1], axis=0)
        if name == 'T-Corex':
            mu = best_method.theta[-1][0]
        mu = mu.astype(np.float64)
        sigma = best_covs[-1].astype(np.float64)

        # portfolio optimization using mu and sigma
        # solvers.qp needs:
        #   minimize 1/2 x^T P x + q^T x
        #   subject to Gx <= h
        #              Ax = b
        #
        # our program:
        #   minimize x^T Sigma x
        #   subject to mu^T x >= r
        #              1^T x = 1
        #              x >= 0
        qp_solutions[name] = {}
        for r in np.linspace(0.0, np.percentile(mu, 99), 100):
            P = 2.0 * matrix(sigma)
            q = matrix(0.0, (nv, 1))
            G = matrix(np.concatenate([-np.eye(nv), -mu.reshape((1, -1))], axis=0))
            h = matrix(np.concatenate([np.zeros((nv, 1)), -r * np.ones((1, 1))], axis=0))
            A = matrix(np.ones((1, nv)))
            b = matrix(1.0)
            qp_solutions[name][r] = solvers.qp(P, q, G, h, A, b)

        # save qp_solutions
        with open(qp_solutions_path, 'wb') as f:
            pickle.dump(qp_solutions, f)

        # save mu and sigma
        mus[name] = mu
        sigmas[name] = sigma
        with open(mu_path, 'wb') as f:
            pickle.dump(mus, f)
        with open(sigma_path, 'wb') as f:
            pickle.dump(sigmas, f)

        # save model selection data
        best_results[name] = {}
        best_results[name]['best_params'] = best_params
        best_results[name]['best_val_score'] = best_score
        all_results[name] = all_cur_results

        with open(best_results_path, 'w') as f:
            json.dump(best_results, f)

        with open(all_results_path, 'w') as f:
            json.dump(all_results, f)

    print("Best results are saved in {}".format(best_results_path))
    print("All results are saved in {}".format(all_results_path))
    print("Means are saved in {}".format(mu_path))
    print("Sigmas are saved in {}".format(sigma_path))
    print("Solutions are saved in {}".format(qp_solutions_path))
    print("Test data is saved in {}".format(test_data_path))


if __name__ == '__main__':
    main()
