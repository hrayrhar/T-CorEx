from __future__ import print_function

import numpy as np
import linearcorex
import torch
import theano_time_corex
import generate_data
import metric_utils
import time


def test_linear_corex_theano():
    r""" Test theano linear CorEx implementation.
    Check if the performance of theano CorEx matches that of standard CorEx.
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

    # train theano linear CorEx
    start_time = time.time()
    theano_corex = theano_time_corex.Corex(nv=128, n_hidden=8, max_iter=1000)
    theano_corex.fit(data)
    theano_corex_cov = theano_corex.get_covariance()
    theano_corex_score = metric_utils.calculate_nll_score(data=[data], covs=[theano_corex_cov])
    print("Theano CorEx training finished:\n\tscore: {:.2f}\n\ttime: {:.2f}\n\tTC: {:.2f}".format(
        theano_corex_score, time.time() - start_time, theano_corex.tc))

    assert np.abs(standard_corex_score - theano_corex_score) < 0.01 * np.abs(standard_corex_score)

