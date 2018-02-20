from generate_data import generate_nglf_from_matrix, generate_nglf_from_model, \
    generate_general_make_spd, generate_nglf_timeseries
from misc_utils import make_sure_path_exists, make_buckets
from sklearn.model_selection import train_test_split
import numpy as np

import cPickle
import argparse
import baselines
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nt', required=True, type=int, help='number of buckets')
    parser.add_argument('--m', required=True, type=int, help='number of latent factors')
    parser.add_argument('--bs', required=True, type=int, help='block size')
    parser.add_argument('--train_cnt', default=16, type=int, help='number of train samples')
    parser.add_argument('--val_cnt', default=16, type=int, help='number of validation samples')
    parser.add_argument('--test_cnt', default=100, type=int, help='number of test samples')
    parser.add_argument('--snr', type=float, default=None, help='signal to noise ratio')
    parser.add_argument('--min_cor', type=float, default=0.8, help='minimum correlation between a child and parent')
    parser.add_argument('--max_cor', type=float, default=1.0, help='minimum correlation between a child and parent')
    parser.add_argument('--min_var', type=float, default=1.0, help='minimum x-variance')
    parser.add_argument('--max_var', type=float, default=1.0, help='maximum x-variance')
    parser.add_argument('--eval_iter', type=int, default=1, help='number of evaluation iterations')
    parser.add_argument('--prefix', type=str, default='', help='optional prefix of experiment name')
    args = parser.parse_args()
    exp_data = vars(args)
    exp_data['nv'] = exp_data['m'] * exp_data['bs']

    (data, sigma) = generate_nglf_from_model(args.nv, args.m, args.nt // 2,
                                             ns=args.train_cnt + args.val_cnt + args.test_cnt,
                                             snr=args.snr, min_var=args.min_var, max_var=args.max_var,
                                             min_cor=args.min_cor, max_cor=args.max_cor)
    args.ground_truth_covs = [sigma for i in range(args.nt)]
    args.train_data = [x[:args.train_cnt] for x in data]
    args.val_data = [x[args.train_cnt:args.train_cnt + args.val_cnt] for x in data]
    args.test_data = [x[-args.test_cnt:] for x in data]

    ''' Define baselines and the grid of parameters '''
    methods = [
        (baselines.GroundTruth(covs=args.ground_truth_covs, test_data=args.test_data), {}, "Ground Truth"),

        (baselines.LinearCorex(), {'n_hidden': [args.m],
                                   'max_iter': 500,
                                   'anneal': True}, "Linear CorEx"),

        (baselines.TimeVaryingCorex(), {'nv': args.nv,
                                        'n_hidden': [args.m],
                                        'max_iter': 500,
                                        'anneal': True,
                                        'l1': [0.0],
                                        'l2': []}, "Time-Varying Linear CorEx (no reg)"),

        (baselines.TimeVaryingCorex(), {'nv': args.nv,
                                        'n_hidden': [args.m],
                                        'max_iter': 500,
                                        'anneal': True,
                                        'l1': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                                        'l2': []}, "Time-Varying Linear CorEx (Sigma)"),

        (baselines.TimeVaryingCorexW(), {'nv': args.nv,
                                         'n_hidden': [args.m],
                                         'max_iter': 500,
                                         'anneal': True,
                                         'l1': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                                         'l2': []}, "Time-Varying Linear CorEx (W)"),

        (baselines.TimeVaryingCorexWWT(), {'nv': args.nv,
                                           'n_hidden': [args.m],
                                           'max_iter': 500,
                                           'anneal': True,
                                           'l1': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                                           'l2': []}, "Time-Varying Linear CorEx (WWT)"),

        (baselines.TimeVaryingGraphLasso(), {'lamb': [0.01, 0.03, 0.1, 0.3],
                                             'beta': [0.03, 0.1, 0.3, 1.0],
                                             'indexOfPenalty': [1],
                                             'max_iter': 30}, "Time-Varying GLASSO"),

        (baselines.TimeVaryingGraphLasso(), {'lamb': [0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
                                             'beta': [0.0],
                                             'indexOfPenalty': [1],
                                             'max_iter': 30}, "Time-Varying GLASSO (no reg)")
    ]

    results = {}
    for (method, params, name) in methods:
        best_params, best_score = method.select(args.train_data, args.val_data, params)
        results[name] = method.evaluate(args.train_data, args.test_data, best_params, args.eval_iter)
        results[name]['best_params'] = best_params
    print results

    all_train_data = []
    for x in args.train_data:
        all_train_data.append(x)
    all_train_data = np.concatenate(all_train_data, axis=0)

    all_val_data = []
    for x in args.val_data:
        all_val_data.append(x)
    all_val_data = np.concatenate(all_val_data, axis=0)

    all_test_data = []
    for x in args.test_data:
        all_test_data.append(x)
    all_test_data = np.concatenate(all_test_data, axis=0)

    print "all_train_data.shape = {}".format(all_train_data.shape)
    print "all_val_data.shape = {}".format(all_val_data.shape)
    print "all_test_data.shape = {}".format(all_test_data.shape)

    ''' Linear CorEx on whole data '''
    (method, params, name) = methods[1]
    best_params, best_score = method.select([all_train_data], [all_val_data], params)
    print "Linear Corex on whole data: ", method.evaluate([all_train_data], [all_test_data], best_params, args.eval_iter)


    ''' GLASSO on whole data '''
    (method, params, name) = methods[-1]
    best_params, best_score = method.select([all_train_data, all_train_data], [all_val_data, all_val_data], params)
    print "GLASSO on whole data: ", method.evaluate([all_train_data, all_train_data], [all_test_data, all_test_data],
                                                    best_params, args.eval_iter)


if __name__ == '__main__':
    main()
