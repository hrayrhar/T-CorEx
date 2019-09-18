from tcorex.experiments import data as data_tools
from tcorex.experiments.misc import make_sure_path_exists
from tcorex.corex import Corex as PyCorex
from sklearn.metrics import adjusted_rand_score
import numpy as np
import linearcorex
import argparse
import pickle
import torch
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_hidden', '-m', default=64, type=int,
                        help='number of hidden variables')
    parser.add_argument('--n_observed', '-p', default=128, type=int,
                        help='number of observed variables')
    parser.add_argument('--snr', '-s', default=0.1, type=float,
                        help='signal-to-noise ratio')
    parser.add_argument('--n_samples', '-n', default=300, type=int,
                        help='number of samples')
    parser.add_argument('--num_extra_parents', default=0.1, type=float,
                        help='average number of extra parents for each x_i')
    parser.add_argument('--num_correlated_zs', default=0, type=int,
                        help='number of zs each z_i is correlated with (besides z_i itself)')
    parser.add_argument('--random_scale', dest='random_scale', action='store_true',
                        help='if true x_i will have random scales')
    parser.add_argument('--method', '-M', type=str, choices=['linearcorex', 'pycorex'],
                        default='pycorex', help='which implementation of corex to use')
    parser.add_argument('--device', '-d', type=str, default='cuda:0',
                        help='which device to use for pytorch corex')
    parser.add_argument('--output_dir', '-o', type=str, default='outputs/blessing/')
    parser.set_defaults(random_scale=False)
    args = parser.parse_args()
    print(args)

    p = args.n_observed
    m = args.n_hidden
    snr = args.snr
    n = args.n_samples
    assert p % m == 0

    # generate some data
    data, _ = data_tools.generate_approximately_modular(nv=p, m=m, ns=n, snr=snr,
                                                        num_extra_parents=args.num_extra_parents,
                                                        num_correlated_zs=args.num_correlated_zs,
                                                        random_scale=args.random_scale)

    # select the method
    if args.method == 'linearcorex':
        method = linearcorex.Corex(n_hidden=m, verbose=1)
    else:
        method = PyCorex(nv=p, n_hidden=m, verbose=2, max_iter=10000,
                         tol=1e-6, device=args.device, optimizer_class=torch.optim.Adam,
                         optimizer_params={'lr': 0.01})

    # train and compute the clustering score
    method.fit(data)
    true_clusters = np.arange(m).repeat(p//m, axis=0)

    if args.method == 'linearcorex':
        pred_clusters = method.mis.argmax(axis=0)
    else:
        pred_clusters = method.clusters()

    score = adjusted_rand_score(labels_true=true_clusters, labels_pred=pred_clusters)
    print(pred_clusters, score)

    # save the results
    run_id = str(np.random.choice(10**18))

    save_dict = {
        'p': p,
        'm': m,
        'snr': snr,
        'n': n,
        'num_extra_parents': args.num_extra_parents,
        'num_correlated_zs': args.num_correlated_zs,
        'random_scale': args.random_scale,
        'method': args.method,
        'score': score,
        'run_id': run_id
    }

    output_file = os.path.join(args.output_dir, run_id + '.pkl')
    make_sure_path_exists(output_file)
    with open(output_file, 'wb') as fout:
        pickle.dump(save_dict, fout)


if __name__ == '__main__':
    main()
