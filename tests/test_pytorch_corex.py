from __future__ import print_function

import numpy as np
import linearcorex
import torch
import pytorch_tcorex
import generate_data
import metric_utils
import time


def test_linear_corex_pytorch():
    r""" Test pytorch linear CorEx implementation.
    Check if the performance of pytorch CorEx matches that of standard CorEx.
    """

    # take some data
    data, _ = generate_data.generate_nglf(nv=128, m=8, nt=1, ns=32)
    data = data[0]
    print("Data is loaded, shape = {}".format(data.shape))

    # train standard linear CorEx
    start_time = time.time()
    standard_corex = linearcorex.Corex(n_hidden=8, max_iter=500)
    standard_corex.fit(data)
    standard_corex_cov = standard_corex.get_covariance()
    standard_corex_score = metric_utils.calculate_nll_score(data=[data], covs=[standard_corex_cov])
    print("Standard CorEx training finished:\n\tscore: {:.2f}\n\ttime: {:.2f}\n\tTC: {:.2f}".format(
        standard_corex_score, time.time() - start_time, standard_corex.tc))

    # train pytorch linear CorEx
    start_time = time.time()
    pytorch_corex = pytorch_tcorex.Corex(nv=128, n_hidden=8, max_iter=1000, torch_device='cuda')
    pytorch_corex.fit(data)
    pytorch_corex_cov = pytorch_corex.get_covariance()
    pytorch_corex_score = metric_utils.calculate_nll_score(data=[data], covs=[pytorch_corex_cov])
    print("Pytorch CorEx training finished:\n\tscore: {:.2f}\n\ttime: {:.2f}\n\tTC: {:.2f}".format(
        pytorch_corex_score, time.time() - start_time, pytorch_corex.tc))

    assert np.abs(standard_corex_score - pytorch_corex_score) < 0.01 * np.abs(standard_corex_score)

