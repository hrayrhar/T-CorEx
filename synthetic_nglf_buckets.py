from generate_data import generate_nglf_from_matrix, generate_nglf_from_model
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
    parser.add_argument('--load_data', type=str, default=None, help='path to previously stored data set')
    parser.add_argument('--eval_iter', type=int, default=5, help='number of evaluation iterations')
    parser.add_argument('--prefix', type=str, default='', help='optional prefix of experiment name')
    args = parser.parse_args()
    args.nv = args.m * args.bs

    if args.load_data:
        print "Loading previously saved data ..."
        with open(args.load_data, 'r') as f:
            data = cPickle.load(f)
        args.nt = data['nt']
        args.m = data['m']
        args.bs = data['bs']
        args.nv = data['nv']
        args.snr = data['snr']
        train_data = data['train_data']
        val_data = data['val_data']
        test_data = data['test_data']
        ground_truth_covs = data['ground_truth_covs']

    else:
        (data1, sigma1) = generate_nglf_from_model(args.nv, args.m, args.nt // 2,
                                                   ns=args.train_cnt + args.val_cnt + args.test_cnt,
                                                   snr=args.snr, min_var=args.min_var, max_var=args.max_var)
        (data2, sigma2) = generate_nglf_from_model(args.nv, args.m, args.nt // 2,
                                                   ns=args.train_cnt + args.val_cnt + args.test_cnt,
                                                   snr=args.snr, min_var=args.min_var, max_var=args.max_var)
        data = data1 + data2
        ground_truth_covs = [sigma1 for i in range(args.nt // 2)] + [sigma2 for i in range(args.nt // 2)]

        train_data = [x[:args.train_cnt] for x in data]
        val_data = [x[args.train_cnt:args.train_cnt + args.val_cnt] for x in data]
        test_data = [x[-args.test_cnt:] for x in data]

    methods = [
        (baselines.GroundTruth(covs=ground_truth_covs), {}),

        (baselines.Diagonal(), {}),

        (baselines.LedoitWolf(), {}),

        (baselines.OAS(), {}),

        (baselines.PCA(), {'n_components': [args.m]}),

        (baselines.FactorAnalysis(), {'n_components': [args.m]}),

        (baselines.GraphLasso(), {'alpha': [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3],
                                  'mode': 'lars',
                                  'max_iter': 100}),

        (baselines.LinearCorex(), {'n_hidden': [3, 4, 5, 6, args.m//2, args.m, 2*args.m, 20],
                                   'max_iter': 500,
                                   'anneal': True}),

        (baselines.TimeVaryingCorex(), {'nt': args.nt,
                                        'nv': args.nv,
                                        'n_hidden': args.m,  # TODO: add to the grid
                                        'max_iter': 500,
                                        'anneal': True,
                                        'l1': 0.3,  # TODO: add to the grid
                                        'l2': 0.0}),  # TODO: add to the grid

        (baselines.TimeVaryingGraphLasso(), {'lamb': 0.001,  # TODO: add to grid
                                             'beta': 0.1,  # TODO: add to grid
                                             'indexOfPenalty': 1})  # TODO add to grid or find the best value
    ]

    results = {}
    for (method, params) in methods[:8]:
        best_params = method.select(train_data, val_data, params)
        results[method.get_name()] = method.evaluate(train_data, test_data, best_params, args.eval_iter)
        results[method.get_name()]['best_params'] = best_params

    print "Saving the data and parameters of the experiment ..."
    exp_name = '{}.nt{}.m{}.bs{}.train_cnt{}.val_cnt{}.test_cnt{}.snr{:.2f}'.format(
        prefix, args.nt, args.m, args.bs, args.train_cnt, args.val_cnt, args.test_cnt, args.snr)

    results_path = "results/{}.results.json".format(exp_name)
    print "Saving the results in {}".format(results_path)
    make_sure_path_exists(results_path)
    with open(results_path, 'w') as f:
        json.dump(results, f)

    data_path = "saved_data/{}.pkl".format(data_path)
    print "Saving data and params in {}".format(data_path)
    data = dict()
    data['prefix'] = args.prefix
    data['nt'] = args.nt
    data['m'] = args.m
    data['bs'] = args.bs
    data['train_cnt'] = args.train_ctn
    data['val_cnt'] = args.val_cnt
    data['test_cnt'] = args.test_cnt
    data['snr'] = args.snr
    data['min_var'] = args.min_var
    data['max_var'] = args.max_var
    data['eval_iter'] = args.eval_iter
    data['train_data'] = train_data
    data['val_data'] = val_data
    data['test_data'] = test_data
    data['ground_truth_covs'] = ground_truth_covs

    make_sure_path_exists(data_path)
    with open(data_path, 'w') as f:
        cPickle.dump(data, f)


if __name__ == '__main__':
    main()
