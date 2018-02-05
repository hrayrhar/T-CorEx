from generate_data import generate_nglf_from_matrix, generate_nglf_from_model
from sklearn.preprocessing import StandardScaler

import numpy as np
import cPickle
import argparse
import baselines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nt', required=True, type=int, help='number of buckets')
    parser.add_argument('--m', required=True, type=int, help='number of latent factors')
    parser.add_argument('--bs', required=True, type=int, help='block size')
    parser.add_argument('--train_cnt', required=True, type=int, help='number of train samples')
    parser.add_argument('--test_cnt', required=True, type=int, help='number of test samples')
    parser.add_argument('--snr', type=float, required=True, help='signal to noise ratio')
    parser.add_argument('--load_data', type=str, default=None, help='path to previously stored data set')
    parser.add_argument('--eval_iter', type=int, default=5, help='number of evaluation iterations')
    args = parser.parse_args()
    args.nv = args.m * args.bs

    if args.load_data:
        print "Loading previously saved data ..."
        with open(args.load_data, 'r') as f:
            data = cPickle.load(f)
        assert data['nt'] == args.nt
        assert data['m'] == args.m
        assert data['bs'] == args.bs
        assert data['nv'] == args.nv
        assert np.equals(data['snr'], args.snr)
        #TODO: take the data

    # Generate data
    # TODO:

    # Model Selection
    print "Starting model selection ..."
    methods = [
        (model_selection.linear_corex, {'n_hidden': args.m,  # TODO: add to the grid
                                        'max_iter': 500,
                                        'anneal': True}),
        (model_selection.time_varying_corexm, {'nt': args.nt,
                                               'nv': args.nv,
                                               'n_hidden': args.m,  # TODO: add to the grid
                                               'max_iter': 500,
                                               'anneal': True,
                                               'l1': 0.3,  # TODO: add to the grid
                                               'l2': 0.0}),
        (model_selection.pca, {'n_components': args.m}),
        (model_selection.factor_analysis, {'n_components': args.m}),
        (model_selection.glasso, {'alpha': 0.1,  # TODO: add to the grid
                                  'max_iter': 100}),  # TODO: find out the best value for this
        (model_selection.time_varying_glasso, {'lamb': 0.001,  # TODO: add to grid
                                               'beta': 0.1,    # TODO: add to grid
                                               'indexOfPenalty': 1})  # TODO add to grid or find the best value
    ]

    for (method_ms, params) in methods:
        best_params = method_ms(train_data, test_data, params)

    # Evaluation
    print "Starting evaluation of the best models ..."
    methods = [

    ]

    # Storing the results
    print "Starting to save the results ..."


if __name__ == '__main__':
    main()
