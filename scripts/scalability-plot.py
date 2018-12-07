from __future__ import print_function
from __future__ import absolute_import

from experiments.generate_data import load_nglf_sudden_change
from experiments import baselines
from experiments.utils import make_sure_path_exists
import pytorch_tcorex
import argparse
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nt', type=int, default=10, help='number of buckets')
    parser.add_argument('--train_cnt', default=16, type=int, help='number of train samples')
    parser.add_argument('--nvs', type=int, nargs='+', default=[2 ** i for i in range(3, 21)])
    parser.add_argument('--prefix', type=str, default='', help='Additional prefix of out file name')
    args = parser.parse_args()
    print(args)

    methods = [
        # memory error (for 32GB RAM) after 2048 (inclusive)
        (baselines.TimeVaryingGraphLasso(name='T-GLASSO'), {
            'lamb': 0.1,
            'beta': 1.0,
            'indexOfPenalty': 1,
            'max_iter': 100,
            'lengthOfSlice': args.train_cnt,
        }),

        (baselines.TCorex(tcorex=pytorch_tcorex.TCorex, name='T-Corex (pytorch, cpu)'), {
            'max_iter': 100,
            'anneal': True,
            'l1': 0.1,
            'gamma': 0.8,
            'reg_type': 'W',
            'init': True,
            'torch_device': 'cpu'
        }),

        (baselines.TCorex(tcorex=pytorch_tcorex.TCorex, name='T-Corex (pytorch, cuda)'), {
            'max_iter': 100,
            'anneal': True,
            'l1': 0.1,
            'gamma': 0.8,
            'reg_type': 'W',
            'init': True,
            'torch_device': 'cuda'
        }),

        (baselines.QUIC(name='QUIC'), {
            'lamb': 0.1,
            'tol': 1e-6,
            'msg': 1,
            'max_iter': 100
        }),

        (baselines.BigQUIC(name='BigQUIC'), {
            'lamb': 3,
            'tol': 1e-3,
            'verbose': 1,
            'max_iter': 100
        }),

        (baselines.LTGL(name='LTGL'), {
            'alpha': 3.0,
            'tau': 30.0,
            'beta': 30.0,
            'psi': 'l1',
            'eta': 3.0,
            'phi': 'l1',
            'rho': 1.0 / np.sqrt(args.train_cnt),
            'max_iter': 500,
            'verbose': False
        })
    ]

    times = {}
    for method, params in methods:
        times[method.name] = []

    out_file = 'experiments/scalability/{}nt{}.train_cnt{}.json'.format(args.prefix, args.nt, args.train_cnt)
    make_sure_path_exists(out_file)
    print("Output file path = {}".format(out_file))

    stop_methods = set()

    for nv in args.nvs:
        bs = min(nv, 16)
        n_hidden = nv // bs

        # generate data
        data, _, = load_nglf_sudden_change(nv=nv, m=n_hidden, nt=args.nt, ns=args.train_cnt,
                                           shuffle=False, from_matrix=False)

        for method, params in methods:
            # start timing
            print("{}\nTiming method: {}, nv: {}".format('-' * 80, method.name, nv))

            if method.name.find('T-Corex') != -1:
                params['nv'] = nv
                params['n_hidden'] = min(n_hidden, 64)

            if method in stop_methods:
                print("\tskipped")
                continue
            try:
                ct = method.timeit(data, params)
                times[method.name].append((nv, ct))
                print("\ttook {:.2f} seconds".format(ct))
                # do not time this method again if ct is more than 6 hours
                if ct > 3600 * 6:
                    stop_methods.add(method.name)
            except Exception as e:
                print("\tfailed with message: '{}'".format(str(e)))

            # save results
            with open(out_file, 'w') as f:
                json.dump(times, f)

    print("Results are saved in {}".format(out_file))


if __name__ == '__main__':
    main()
