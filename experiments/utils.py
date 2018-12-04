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


""" Helper functions for working with large low-rank plus diagonal matrices
"""


def diag_from_left(A, d):
    """ diag(d) A """
    m, n = A.shape
    X = np.zeros_like(A)
    for i in range(m):
        X[i, :] = A[i, :] * d[i]
    return X


def diag_from_right(A, d):
    """ A diag(d) """
    m, n = A.shape
    X = np.zeros_like(A)
    for i in range(n):
        X[:, i] = A[:, i] * d[i]
    return X


def inverse(A, d):
    """ Compute inverse of A^T A + diag(d) faster than n^2.
        A - (m, n)
        d - (n,)
        Return: V, d_inv such that inverse = d_inv - V^T V
    """
    m, n = A.shape
    d_inv = 1 / d
    M = np.eye(m) + np.dot(diag_from_right(A, d_inv), A.T)  # m^2 * n
    M_inv = np.linalg.inv(M)  # m^3
    R = np.linalg.cholesky(M_inv)  # m^3
    V = diag_from_left(np.dot(A.T, R), d_inv).T  # m^2 * n
    return V, d_inv


def compute_inverses(A):
    """ Given the low-rank parts of correlation matrices, compute low-rank + diagonal representation
    of inverse correlation matrices.
    Returns: B, d_inv such that Sigma^{-1}_t = d_inv_t - B_t^T B_t
    """
    nt = len(A)
    d = [None] * nt
    d_inv = [None] * nt
    B = [None] * nt
    for t in range(nt):
        d[t] = 1 - (A[t] ** 2).sum(axis=0)
        d_inv[t] = 1.0 / d[t]

    for t in range(nt):
        B[t], d_inv[t] = inverse(A[t], d[t])

    return B, d_inv


def estimate_diff_norm(A_1, d_1, A_2, d_2, n_iters=300):
    """ Estimates ||(d_1 - A_1^T A_1) - (d_2 - A_2^T A_2)|| using power method.
    """
    ret = 0
    for iteration in range(n_iters):
        if iteration == 0:
            x = np.random.normal(size=(A_1.shape[1], 1))
        else:
            x = -np.dot(A_1.T, np.dot(A_1, x)) + np.dot(A_2.T, np.dot(A_2, x)) + (d_1 - d_2).reshape((-1, 1)) * x
        s = np.linalg.norm(x, ord=2)
        x = x / s
        ret = s
        # print("\tIteration: {}, ret: {}".format(iteration, ret), end='\r')
    return ret


def compute_diff_norms(A):
    """ Given the low-rank parts of correlation matrices, compute spectral norm of differences of inverse
    correlation matrices of neighboring time steps.
    """
    B, d_inv = compute_inverses(A)
    nt = len(A)
    diffs = [None] * (nt-1)
    for t in range(nt - 1):
        print("Estimating norm of difference at time step: {}".format(t))
        diffs[t] = estimate_diff_norm(B[t], d_inv[t], B[t + 1], d_inv[t + 1])
    return diffs


def compute_diff_norm_fro(A, d_1, B, d_2):
    """ Estimates ||(d_1 - A^T A) - (d_2 - B^T B)||_F analytically.
    """
    # First compute || B^T B - A^T A ||^2_F
    _, sigma_a, _ = np.linalg.svd(A, full_matrices=False)
    _, sigma_b, _ = np.linalg.svd(B, full_matrices=False)
    low_rank_norm = np.sum(sigma_a ** 4) - 2 * np.trace(np.dot(np.dot(B, A.T), np.dot(A, B.T))) + np.sum(sigma_b ** 4)

    # Let X = (B^T B - A^T A), D = diag(d_1 - d_2). Compute ||X + D||^2_F
    d = d_1 - d_2
    ret = low_rank_norm + (np.linalg.norm(d) ** 2) + 2 * np.inner(np.sum(B ** 2 - A ** 2, axis=0), d)
    return np.sqrt(ret)


def compute_diff_norms_fro(A):
    """ Given the low-rank parts of correlation matrices, compute Frobenius norm of differences of inverse
    correlation matrices of neighboring time steps.
    """
    B, d_inv = compute_inverses(A)
    nt = len(A)
    diffs = [None] * (nt-1)
    for t in range(nt - 1):
        print("Calculating Frobenius norm of difference at time step: {}".format(t))
        diffs[t] = compute_diff_norm_fro(B[t], d_inv[t], B[t + 1], d_inv[t + 1])
    return diffs

