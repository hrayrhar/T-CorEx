from __future__ import print_function
from __future__ import absolute_import

from tcorex.experiments.data import load_nglf_sudden_change
from tcorex.experiments import baselines
from tcorex import base
from tcorex import TCorex
from tcorex import covariance as cov_utils

import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt


def main():
    nv = 32         # number of observed variables
    m = 4           # number of hidden variables
    nt = 10         # number of time periods
    train_cnt = 16  # number of training samples for each time period
    val_cnt = 4     # number of validation samples for each time period

    # Generate some data with a sudden change in the middle.
    data, ground_truth_sigma = load_nglf_sudden_change(nv=nv, m=m, nt=nt, ns=(train_cnt + val_cnt))

    # Split it into train and validation.
    train_data = [X[:train_cnt] for X in data]
    val_data = [X[train_cnt:] for X in data]

    # NOTE: the load_nglf_sudden_change function above creates data where the time axis
    # is already divided into time periods. If your data is not divided into time periods
    # you can use the following procedure to do that:
    # bucketed_data, index_to_bucket = make_buckets(data, window=train_cnt + val_cnt, stride='full')
    # where the make_buckets function can be found at tcorex.experiments.data

    # The core method we have is the tcorex.TCorex class.
    tc = TCorex(nt=nt,
                nv=nv,
                n_hidden=m,
                max_iter=500,
                device='cpu',  # for GPU set 'cuda',
                l1=0.3,        # coefficient of temporal regularization term
                gamma=0.3,     # parameter that controls sample weights
                verbose=1,     # 0, 1, 2
                )

    # Fit the parameters of T-CorEx.
    tc.fit(train_data)

    # We can compute the clusters of observed variables for each time period.
    t = 8
    clusters = tc.clusters()
    print("Clusters at time period {}: {}".format(t, clusters[t]))

    # We can get an estimate of the covariance matrix for each time period.
    # When normed=True, estimates of the correlation matrices will be returned.
    covs = tc.get_covariance()

    # We can visualize the covariance matrices.
    fig, ax = plt.subplots(1, figsize=(5, 5))
    im = ax.imshow(covs[t])
    fig.colorbar(im)
    ax.set_title("Estimated covariance matrix\nat time period {}".format(t))
    fig.savefig('covariance-matrix.png')

    # It is usually useful to compute the inverse correlation matrices,
    # since this matrices can be interpreted as adjacency matrices of
    # Markov random fields.
    cors = tc.get_covariance(normed=True)
    inv_cors = [np.linalg.inv(x) for x in cors]

    # We can visualize the thresholded inverse correlation matrices.
    fig, ax = plt.subplots(1, figsize=(5, 5))
    thresholded_inv_cor = np.abs(inv_cors[t]) > 0.05
    ax.imshow(thresholded_inv_cor)
    ax.set_title("Thresholded inverse correlation\nmatrix at time period {}".format(t))
    fig.savefig('thresholded-inverse-correlation-matrix.png')

    # We can also plot the Frobenius norm of the differences of inverse
    # correlation matrices of  neighboring time periods. This is helpful
    # for detecting the sudden change points of the system.
    diffs = cov_utils.diffs(inv_cors)
    fig, ax = plt.subplots(1, figsize=(5, 5))
    ax.plot(diffs)
    ax.set_xlabel('t')
    ax.set_ylabel('$||\Sigma^{-1}_{t+1} - \Sigma^{-1}_{t}||_2$')
    ax.set_title("Frobenius norms of differences between\ninverse correlation matrices")
    fig.savefig('inv-correlation-difference-norms.png')

    # We can also do grid search on a hyperparameter grid the following way.
    # NOTE: this can take time!
    baseline, grid = (baselines.TCorex(tcorex=TCorex, name='T-Corex'), {
        'nv': nv,
        'n_hidden': m,
        'max_iter': 500,
        'device': 'cpu',
        'l1': [0.0, 0.03, 0.3, 3.0],
        'gamma': [1e-6, 0.3, 0.5, 0.8]
    })

    best_score, best_params, best_covs, best_method, all_results = baseline.select(train_data, val_data, grid)
    tc = best_method  # this is the model that performed the best on the validation data, you can use it as above
    base.save(tc, 'best_method.pkl')


if __name__ == '__main__':
    main()
