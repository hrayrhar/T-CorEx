from __future__ import print_function

import torch
import pytorch_tcorex
import metric_utils
import time
from tests.utils import load_sp500_with_commodities
from generate_data import load_nglf_smooth_change
import warnings


def test_tcorex_pytorch_on_sp500():
    r""" Test pytorch implementation of T-CorEx (S&P 500).
    Check of the performance is below some margin, which was established by Theano T-CorEx.
    """

    train_data, val_data, _ = load_sp500_with_commodities()

    # train pytorch T-CorEx
    start_time = time.time()
    tcorex = pytorch_tcorex.TCorexWeights(n_hidden=11, nv=train_data.shape[-1], nt=train_data.shape[0],
                                          max_iter=100, anneal=True, l1=0.1, gamma=1.5, reg_type='W',
                                          init=True, torch_device='cuda', verbose=1)
    tcorex.fit(train_data)
    tcorex_covs = tcorex.get_covariance()
    tcorex_score = metric_utils.calculate_nll_score(data=val_data, covs=tcorex_covs)
    print("Pytorch T-CorEx training finished:\n\tscore: {:.2f}\n\ttime: {:.2f}".format(
        tcorex_score, time.time() - start_time))
    
    assert tcorex_score < 510


def test_tcorex_pytorch_on_synthetic_data():
    r""" Test pytorch implementation of T-CorEx (synthetic data).
    Check of the performance is below some margin, which was established by Theano T-CorEx.
    """

    data, _ = load_nglf_smooth_change(nv=128, m=8, nt=10, ns=16+16+100)
    train_data = data[:, :16]
    val_data = data[:, 16:32]
    test_data = data[:, 32:]

    # train pytorch T-CorEx
    start_time = time.time()
    tcorex = pytorch_tcorex.TCorexWeights(n_hidden=8, nv=train_data.shape[-1], nt=train_data.shape[0],
                                          max_iter=500, anneal=True, l1=0.3, gamma=2.5, reg_type='W',
                                          init=True, torch_device='cpu', verbose=1)
    tcorex.fit(train_data)
    tcorex_covs = tcorex.get_covariance()
    tcorex_score = metric_utils.calculate_nll_score(data=test_data, covs=tcorex_covs)
    print("Pytorch T-CorEx training finished:\n\tscore: {:.2f}\n\ttime: {:.2f}".format(
        tcorex_score, time.time() - start_time))
    
    # 213 was achieved once, 199 is the ground truth
    assert tcorex_score < 220
    if tcorex_score > 216:
        warnings.warn("Pytorch tcorex on synthetic data gets slightly worse score than expected\n"
                      "\tExpected: [212 - 216], Got: {}".format(tcorex_score))
