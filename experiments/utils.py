from __future__ import print_function
from __future__ import absolute_import

from scipy.stats import multivariate_normal
import numpy as np
import os


def calculate_nll_score(data, covs):
    """ Calculate negative log-likelihood of covariance estimates under data.
    """
    nt = len(data)
    assert len(covs) == nt
    try:
        nll = [-multivariate_normal.logpdf(data[t], cov=covs[t]).mean() for t in range(nt)]
    except Exception:
        nll = [np.inf]
    return np.mean(nll)


def make_buckets(ts_data, window, stride):
    """ Divide time series data into buckets.
    Returns the bucketed data plus a map that maps original indices into bucket indices.
    """
    ts_data = np.array(ts_data)
    nt = len(ts_data)

    if stride == 'one':
        shift = 1
    elif stride == 'half':
        shift = window // 2
    elif stride == 'full':
        shift = window
    else:
        raise ValueError("Unknown value for stride")

    start_indices = range(0, nt - window + 1, shift)
    bucketed_data = []
    midpoints = []
    for i, start in enumerate(start_indices):
        end = start + window
        if i == len(start_indices) - 1 and end != nt:  # if the last bucket doesn't include the rightmost sample
            end = nt
        bucketed_data.append(np.array(ts_data[start:end]))
        midpoints.append((start + end - 1.0) / 2.0)

    index_to_bucket = []
    for i in range(nt):
        best = 0
        for j in range(len(midpoints)):
            if np.abs(i - midpoints[j]) < np.abs(i - midpoints[best]):
                best = j
        index_to_bucket.append(best)

    return bucketed_data, index_to_bucket


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


def plot_for_next_timestep(plt, data, covs, title="Negative log-likelihood of estimate of time step $t$ under "
                                                  "the test data of timestep $t + 1$"):
    nt = len(data)
    nll = [-np.mean([multivariate_normal.logpdf(sx, cov=covs[t]) for sx in x])
           for x, t in zip(data[1:], range(nt - 1))]
    plt.bar(range(1, nt), nll, width=0.6)
    plt.title(title)
    plt.xlabel("Timestep")
    plt.ylim(0)
    plt.ylabel("Negative log-likelihood")
    plt.xticks(range(1, nt))
    plt.show()
    print("NLL for next time step = {}".format(np.mean(nll)))
    return np.mean(nll)
