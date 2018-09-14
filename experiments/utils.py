from __future__ import print_function
from __future__ import absolute_import

from scipy.stats import multivariate_normal
import numpy as np
import os


def calculate_nll_score(data, covs):
    """ Calculate negative log-likelihood of covariance estimates under data.
    """
    nt = len(data)
    try:
        nll = [-np.mean([multivariate_normal.logpdf(sx, cov=covs[t]) for sx in x])
               for x, t in zip(data, range(nt))]
    except Exception:
        nll = [np.inf]
    return np.mean(nll)


def make_buckets(ts_data, test_data, window, stride):
    """ Divide timeseries data into buckets.
    """
    nt = len(ts_data)
    K = 2 * window + 1

    if stride == 'one':
        shift = 1
    elif stride == 'half':
        shift = K // 2
    elif stride == 'full':
        shift = K
    else:
        raise ValueError("Unknown value for stride")

    ret_train = []
    midpoints = []
    for i in range(0, nt - K + 1, shift):
        j = i + K
        # if this is the last full bucket, merge it with the tail
        if i + shift >= nt - K + 1:
            j = nt
        midpoints.append((i + j) // 2)
        ret_train.append(ts_data[i:j])

    ret_test = [[] for i in range(len(midpoints))]
    for i in range(nt):
        best_pos = 0
        for pos in range(len(midpoints)):
            if abs(i - midpoints[best_pos]) > abs(i - midpoints[pos]):
                best_pos = pos
        ret_test[best_pos].append(test_data[i])

    for i in range(len(midpoints)):
        ret_test[i] = np.concatenate(ret_test[i], axis=0)

    # make sure all test_data is used
    assert np.sum([len(x) for x in ret_test]) == test_data.shape[0] * test_data.shape[1]

    assert len(ret_train) == len(ret_test)
    return ret_train, ret_test


def make_sure_path_exists(path):
    dir_name = os.path.dirname(path)
    try:
        os.makedirs(dir_name)
    except:
        pass


def plot_cov_matrix(plt, cov, title=None, vmin=-1, vmax=+1):
    plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.imshow(cov, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.show()


def plot_for_next_timestep(plt, data, covs,title="Negative log-likelihood of estimate of time step $t$ under "
                                                 "the test data of timestep $t + 1$"):
    nt = len(data)
    nll = [-np.mean([multivariate_normal.logpdf(sx, cov=covs[t]) for sx in x])
           for x, t in zip(data[1:], range(nt-1))]
    plt.bar(range(1, nt), nll, width=0.6)
    plt.title(title)
    plt.xlabel("Timestep")
    plt.ylim(0)
    plt.ylabel("Negative log-likelihood")
    plt.xticks(range(1, nt))
    plt.show()
    print("NLL for next time step = {}".format(np.mean(nll)))
    return np.mean(nll)
