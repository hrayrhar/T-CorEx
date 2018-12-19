from __future__ import absolute_import
from __future__ import print_function

from tcorex.covariance import calculate_nll_score
from tcorex import TCorex
from tqdm import tqdm
import numpy as np
import os


# def test_tcorex_on_sp500():
#     r""" Test pytorch implementation of T-CorEx (S&P 500).
#     Check of the performance is below some margin, which was established by Theano T-CorEx.
#     """
#
#     train_data, val_data, _ = load_sp500_with_commodities()
#
#     # train pytorch T-CorEx
#     start_time = time.time()
#     tcorex = pytorch_tcorex.TCorexWeights(n_hidden=11, nv=train_data.shape[-1], nt=train_data.shape[0],
#                                           max_iter=100, anneal=True, l1=0.1, gamma=1.5, reg_type='W',
#                                           init=True, torch_device='cuda', verbose=1)
#     tcorex.fit(train_data)
#     tcorex_covs = tcorex.get_covariance()
#     tcorex_score = metric_utils.calculate_nll_score(data=val_data, covs=tcorex_covs)
#     print("Pytorch T-CorEx training finished:\n\tscore: {:.2f}\n\ttime: {:.2f}".format(
#         tcorex_score, time.time() - start_time))
#
#     assert tcorex_score < 510


def test_tcorex_on_synthetic_data():
    r""" Test pytorch implementation of T-CorEx (synthetic data).
    Check of the performance is below some margin, which was established by Theano T-CorEx.
    """
    print("=" * 100)
    print("Testing PyTorch T-CorEx on synthetic data ...")
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
