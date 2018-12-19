from __future__ import absolute_import
from __future__ import print_function

from tcorex.covariance import calculate_nll_score
from tcorex import Corex
from tqdm import tqdm
import numpy as np
import linearcorex
import os


def test_corex():
    r""" Test pytorch linear CorEx implementation.
    Check if the performance of pytorch CorEx matches that of standard CorEx.
    """
    print("=" * 100)
    print("Testing PyTorch Linear CorEx ...")

    # load data
    resources = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources')
    data_file = os.path.join(resources, 'test_corex_data.npy')
    data = np.load(data_file)
    print("Data is loaded, shape = {}".format(data.shape))

    # train linear corex
    lc_scores = []
    for i in tqdm(range(5)):
        X = data[32 * i: 32 * (i + 1)]
        lc = linearcorex.Corex(n_hidden=8, max_iter=500, verbose=0)
        lc.fit(X)
        covs = lc.get_covariance()
        cur_score = calculate_nll_score(data=[X], covs=[covs])
        lc_scores.append(cur_score)

    # train pytorch corex
    pylc_scores = []
    for i in tqdm(range(5)):
        X = data[32 * i: 32 * (i + 1)]
        lc = Corex(nv=128, n_hidden=8, max_iter=1000, verbose=0)
        lc.fit(X)
        covs = lc.get_covariance()
        cur_score = calculate_nll_score(data=[X], covs=[covs])
        pylc_scores.append(cur_score)

    lc_mean = np.mean(lc_scores)
    pylc_mean = np.mean(pylc_scores)
    assert (pylc_mean - lc_mean) / (np.abs(lc_mean) + 1e-6) < 0.01
