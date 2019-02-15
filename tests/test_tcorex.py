from __future__ import absolute_import
from __future__ import print_function

from tcorex.covariance import calculate_nll_score
from tcorex import TCorex
from tqdm import tqdm
import numpy as np
import os


def test_tcorex_on_synthetic_data():
    r""" Test pytorch implementation of T-CorEx on a synthetic dataset.
    """
    print("=" * 100)
    print("Testing PyTorch T-CorEx on a synthetic dataset ...")
    # load data
    resources = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources')
    data_file = os.path.join(resources, 'test_tcorex_synthetic.npy')
    data = np.load(data_file)
    print("Data is loaded, shape = {}".format(data.shape))

    scores = []
    for i in tqdm(range(5)):
        train_data = data[i, :, :16]
        test_data = data[i, :, 16:]
        tc = TCorex(n_hidden=8, nv=train_data.shape[-1], nt=train_data.shape[0],
                    max_iter=500, anneal=True, l1=0.3, gamma=0.4, reg_type='W',
                    init=True, device='cpu', verbose=0)
        tc.fit(train_data)
        covs = tc.get_covariance()
        cur_score = calculate_nll_score(data=test_data, covs=covs)
        scores.append(cur_score)

    score_mean = np.mean(scores)
    need_score = 249.315542545698
    assert (score_mean - need_score) / need_score < 0.01


def test_tcorex_real_data():
    r""" Test pytorch implementation of T-CorEx on a real-world dataset.
    """
    print("=" * 100)
    print("Testing PyTorch T-CorEx on a real-world dataset ...")
    resources = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources')
    data_file = os.path.join(resources, 'test_tcorex_real_data.npy')
    data = np.load(data_file)
    print("Data is loaded, shape = {}".format(data.shape))
    train_data = data[:, :40, :]
    test_data = data[:, 40:, :]

    scores = []
    for i in tqdm(range(5)):
        tc = TCorex(n_hidden=8, nv=train_data.shape[-1], nt=train_data.shape[0],
                    max_iter=500, anneal=True, l1=0.3, gamma=0.4, reg_type='W',
                    init=True, device='cpu', verbose=1)
        tc.fit(train_data)
        covs = tc.get_covariance()
        cur_score = calculate_nll_score(data=test_data, covs=covs)
        scores.append(cur_score)

    score_mean = np.mean(scores)
    need_score = 396.1597
    print("score: {:.4f}, need score: {:.4f}".format(score_mean, need_score))
    assert (score_mean - need_score) / need_score < 0.01
