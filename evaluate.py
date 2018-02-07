from generate_data import generate_nglf_from_matrix, generate_nglf_from_model, generate_general_make_spd
from misc_utils import make_sure_path_exists

import cPickle
import argparse
import baselines
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nt', required=True, type=int, help='number of buckets')
    parser.add_argument('--m', required=True, type=int, help='number of latent factors')
    parser.add_argument('--bs', required=True, type=int, help='block size')
    parser.add_argument('--train_cnt', required=True, type=int, help='number of train samples')
    parser.add_argument('--val_cnt', required=True, type=int, help='number of validation samples')
    parser.add_argument('--test_cnt', required=True, type=int, help='number of test samples')
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
        if args.data_type == 'syn_nglf_buckets':
            (data1, sigma1) = generate_nglf_from_model(args.nv, args.m, args.nt // 2,
                                                       ns=args.train_cnt + args.val_cnt + args.test_cnt,
                                                       snr=args.snr, min_var=args.min_var, max_var=args.max_var)
            (data2, sigma2) = generate_nglf_from_model(args.nv, args.m, args.nt // 2,
                                                       ns=args.train_cnt + args.val_cnt + args.test_cnt,
                                                       snr=args.snr, min_var=args.min_var, max_var=args.max_var)
            data = data1 + data2
            args.ground_truth_covs = [sigma1 for i in range(args.nt // 2)] + [sigma2 for i in range(args.nt // 2)]
            args.train_data = [x[:args.train_cnt] for x in data]
            args.val_data = [x[args.train_cnt:args.train_cnt + args.val_cnt] for x in data]
            args.test_data = [x[-args.test_cnt:] for x in data]

        if args.data_type == 'syn_general_buckets':
            (data1, sigma1) = generate_general_make_spd(args.nv, args.m, args.nt // 2,
                                                        ns=args.train_cnt + args.val_cnt + args.test_cnt)
            (data2, sigma2) = generate_general_make_spd(args.nv, args.m, args.nt // 2,
                                                        ns=args.train_cnt + args.val_cnt + args.test_cnt)
            data = data1 + data2
            args.ground_truth_covs = [sigma1 for i in range(args.nt // 2)] + [sigma2 for i in range(args.nt // 2)]
            args.train_data = [x[:args.train_cnt] for x in data]
            args.val_data = [x[args.train_cnt:args.train_cnt + args.val_cnt] for x in data]
            args.test_data = [x[-args.test_cnt:] for x in data]

    methods = [
        (baselines.GroundTruth(covs=args.ground_truth_covs), {}, "Ground Truth"),

        (baselines.Diagonal(), {}), "Diagonal",

        (baselines.LedoitWolf(), {}, "Ledoit-Wolf"),

        (baselines.OAS(), {}, "Oracle approximating shrinkage"),

        (baselines.PCA(), {'n_components': [args.m]}, "PCA"),

        (baselines.FactorAnalysis(), {'n_components': [args.m]}, "Factor Analysis"),

        (baselines.GraphLasso(), {'alpha': [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3],
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

        (baselines.TimeVaryingCorexW(), {'nt': args.nt,
                                         'nv': args.nv,
                                         'n_hidden': [args.m],
                                         'max_iter': 500,
                                         'anneal': True,
                                         'l1': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                                         'l2': []}, "Time-Varying Linear CorEx (W)"),

        (baselines.TimeVaryingCorexMI(), {'nt': args.nt,
                                          'nv': args.nv,
                                          'n_hidden': [args.m],
                                          'max_iter': 500,
                                          'anneal': True,
                                          'l1': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                                          'l2': []}, "Time-Varying Linear CorEx (MI)"),

        (baselines.TimeVaryingCorexWWT(), {'nt': args.nt,
                                           'nv': args.nv,
                                           'n_hidden': [args.m],
                                           'max_iter': 500,
                                           'anneal': True,
                                           'l1': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                                           'l2': []}, "Time-Varying Linear CorEx (WWT)"),

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
    for (method, params, name) in methods[-1:]:
        best_params = method.select(args.train_data, args.val_data, params)
        results[name] = method.evaluate(args.train_data, args.test_data, best_params, args.eval_iter)
        results[name]['best_params'] = best_params

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
