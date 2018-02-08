from generate_data import generate_nglf_from_matrix, generate_nglf_from_model,\
    generate_general_make_spd, generate_nglf_timeseries
from misc_utils import make_sure_path_exists, make_buckets
from sklearn.model_selection import train_test_split

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
    parser.add_argument('--min_var', type=float, default=1.0, help='minimum x-variance')
    parser.add_argument('--max_var', type=float, default=1.0, help='maximum x-variance')
    parser.add_argument('--eval_iter', type=int, default=5, help='number of evaluation iterations')
    parser.add_argument('--prefix', type=str, default='', help='optional prefix of experiment name')
    parser.add_argument('--data_type', dest='data_type', action='store', default='syn_nglf_buckets',
                        choices=['syn_nglf_buckets', 'syn_general_buckets', 'syn_nglf_ts',
                                 'syn_general_ts', 'stock_day', 'stock_week'],
                        help='which dataset to load/create')
    parser.add_argument('--load_experiment', type=str, default=None, help='path to previously stored experiment')
    args = parser.parse_args()
    exp_data = vars(args)
    exp_data['nv'] = exp_data['m'] * exp_data['bs']

    if args.load_experiment:
        print "Loading previously saved data from {} ...".format(args.load_experiment)
        with open(args.load_experiment, 'r') as f:
            loaded_exp_data = cPickle.load(f)
            for k, v in loaded_exp_data.iteritems():
                if k != 'prefix':
                    exp_data[k] = v
    else:
        if args.data_type in ['syn_nglf_buckets', 'syn_general_buckets']:
            if args.data_type == 'syn_nglf_buckets':
                (data1, sigma1) = generate_nglf_from_model(args.nv, args.m, args.nt // 2,
                                                           ns=args.train_cnt + args.val_cnt + args.test_cnt,
                                                           snr=args.snr, min_var=args.min_var, max_var=args.max_var)
                (data2, sigma2) = generate_nglf_from_model(args.nv, args.m, args.nt // 2,
                                                           ns=args.train_cnt + args.val_cnt + args.test_cnt,
                                                           snr=args.snr, min_var=args.min_var, max_var=args.max_var)
            else:
                (data1, sigma1) = generate_general_make_spd(args.nv, args.m, args.nt // 2,
                                                            ns=args.train_cnt + args.val_cnt + args.test_cnt)
                (data2, sigma2) = generate_general_make_spd(args.nv, args.m, args.nt // 2,
                                                            ns=args.train_cnt + args.val_cnt + args.test_cnt)

            data = data1 + data2
            args.ground_truth_covs = [sigma1 for i in range(args.nt // 2)] + [sigma2 for i in range(args.nt // 2)]
            args.train_data = [x[:args.train_cnt] for x in data]
            args.val_data = [x[args.train_cnt:args.train_cnt + args.val_cnt] for x in data]
            args.test_data = [x[-args.test_cnt:] for x in data]

        if args.data_type in ['syn_nglf_ts']:
            (args.ts_data, args.test_data, args.ground_truth_covs) = generate_nglf_timeseries(
                nv=args.nv, m=args.m, nt=args.nt, ns=args.test_cnt, snr=args.snr,
                min_cor=0.8, min_var=args.min_var, max_var=args.max_var)

    is_time_series = True
    if args.data_type.find('bucket') != -1:
        is_time_series = False
    # windows = [3, 4, 6, 8, 12, 16]
    windows = [4, 8, 12]

    methods = [
        (baselines.GroundTruth(covs=args.ground_truth_covs), {}, "Ground Truth"),

        (baselines.Diagonal(), {}, "Diagonal"),

        (baselines.LedoitWolf(), {}, "Ledoit-Wolf"),

        (baselines.OAS(), {}, "Oracle approximating shrinkage"),

        (baselines.PCA(), {'n_components': [args.m]}, "PCA"),

        (baselines.FactorAnalysis(), {'n_components': [args.m]}, "Factor Analysis"),

        (baselines.GraphLasso(), {'alpha': [0.03, 0.1, 0.3, 1, 3],
                                  'mode': 'lars',
                                  'max_iter': 100}, "Graphical LASSO"),

        (baselines.LinearCorex(), {'n_hidden': [args.m],
                                   'max_iter': 500,
                                   'anneal': True}, "Linear CorEx"),

        (baselines.TimeVaryingCorex(), {'nt': args.nt,
                                        'nv': args.nv,
                                        'n_hidden': [args.m],
                                        'max_iter': 500,
                                        'anneal': True,
                                        'l1': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                                        'l2': []}, "Time-Varying Linear CorEx (Sigma)"),

        # (baselines.TimeVaryingCorexW(), {'nt': args.nt,
        #                                  'nv': args.nv,
        #                                  'n_hidden': [args.m],
        #                                  'max_iter': 500,
        #                                  'anneal': True,
        #                                  'l1': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        #                                  'l2': []}, "Time-Varying Linear CorEx (W)"),

        # (baselines.TimeVaryingCorexMI(), {'nt': args.nt,
        #                                   'nv': args.nv,
        #                                   'n_hidden': [args.m],
        #                                   'max_iter': 500,
        #                                   'anneal': True,
        #                                   'l1': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        #                                   'l2': []}, "Time-Varying Linear CorEx (MI)"),

        # (baselines.TimeVaryingCorexWWT(), {'nt': args.nt,
        #                                    'nv': args.nv,
        #                                    'n_hidden': [args.m],
        #                                    'max_iter': 500,
        #                                    'anneal': True,
        #                                    'l1': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        #                                    'l2': []}, "Time-Varying Linear CorEx (WWT)"),

        (baselines.TimeVaryingGraphLasso(), {'lamb': [0.01, 0.03, 0.1, 0.3],
                                             'beta': [0.01, 0.03, 0.1, 0.3],
                                             'indexOfPenalty': [1],  # TODO: extend grid of this one
                                             'max_iter': 30}, "Time-Varying GLASSO"),

        (baselines.TimeVaryingGraphLasso(), {'lamb': [0.01, 0.03, 0.1, 0.3],
                                             'beta': [0.0],
                                             'indexOfPenalty': [1],
                                             'max_iter': 30}, "Time-Varying GLASSO (no reg)")
    ]

    results = {}
    for (method, params, name) in methods:
        if not is_time_series:  # buckets
            best_params, best_score = method.select(args.train_data, args.val_data, params)
            results[name] = method.evaluate(args.train_data, args.test_data, best_params, args.eval_iter)
            results[name]['best_params'] = best_params
        else:  # time-series
            best_score = 1e18
            best_params = None
            best_window = None
            for window in windows:
                if 2*window + 1 > len(args.ts_data):
                    continue
                data = make_buckets(args.ts_data, window)
                train_data = []
                val_data = []
                for t in range(len(data)):
                    A, B = train_test_split(data[t], test_size=1)
                    train_data.append(A)
                    val_data.append(B)
                cur_best_params, cur_best_score = method.select(train_data, val_data, params)
                if best_params is None or cur_best_score < best_score:
                    best_params = cur_best_params
                    best_score = cur_best_score
                    best_window = window
            train_data = make_buckets(args.ts_data, best_window)
            results[name] = method.evaluate(train_data, args.test_data, best_params, args.eval_iter)
            results[name]['best_params'] = best_params
            results[name]['best_window'] = best_window

    print "Saving the data and parameters of the experiment ..."
    exp_name = '{}.nt{}.m{}.bs{}.train_cnt{}.val_cnt{}.test_cnt{}.snr{:.2f}'.format(
        args.data_type, args.nt, args.m, args.bs, args.train_cnt, args.val_cnt, args.test_cnt, args.snr)
    if args.prefix != '':
        exp_name = args.prefix + '.' + exp_name

    results_path = "results/{}.results.json".format(exp_name)
    print "Saving the results in {}".format(results_path)
    make_sure_path_exists(results_path)
    with open(results_path, 'w') as f:
        json.dump(results, f)

    data_path = "saved_data/{}.pkl".format(exp_name)
    print "Saving data and params in {}".format(data_path)
    make_sure_path_exists(data_path)
    with open(data_path, 'w') as f:
        cPickle.dump(exp_data, f)


if __name__ == '__main__':
    main()
