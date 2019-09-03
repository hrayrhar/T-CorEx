from __future__ import print_function
from __future__ import absolute_import

from scipy.stats import multivariate_normal
import numpy as np


def calculate_nll_score(data, covs):
    """ Calculate time-averaged negative log-likelihood.
    :param data: 3d array or list of 2d arrays.
    :param covs: list of covariance matrices.
    """
    nt = len(data)
    assert len(covs) == nt
    try:
        nll = [-multivariate_normal.logpdf(data[t], cov=covs[t]).mean() for t in range(nt)]
    except Exception:
        nll = [np.inf]
    return np.mean(nll)


def diffs(matrices, norm='fro'):
    """ Computes the norms of differences of neighboring matrices.
    :param matrices: list of matrices
    :param ord: variable that will be passed to np.linalg.norm
    """
    nt = len(matrices)
    ret = []
    for t in range(nt - 1):
        ret.append(np.linalg.norm(matrices[t] - matrices[t+1], ord=norm))
    return ret


def reorder(mat, clusters):
    """ Given which variable belongs to which group, reorders the matrix. """
    n = len(clusters)
    order = list(range(0, n))
    order = sorted(order, key=lambda i: clusters[i])
    ret = mat.copy()[order]
    ret = ret[:, order]
    return ret


# Below are tools for working with high dimensional covariance matrices
# These tools get the low-rank factorization of cov. matrix T-CorEx provides


def _diag_from_left(A, d):
    """ dot(diag(d), A) """
    m, n = A.shape
    X = np.zeros_like(A)
    for i in range(m):
        X[i, :] = A[i, :] * d[i]
    return X


def _diag_from_right(A, d):
    """ dot(A, diag(d)) """
    m, n = A.shape
    X = np.zeros_like(A)
    for i in range(n):
        X[:, i] = A[:, i] * d[i]
    return X


def _inverse(A, d):
    """ Compute inverse of A^T A + diag(d) faster than n^2.
    :param A: (m, n)
    :param d: (n,)
    :return V, d_inv such that inverse = d_inv - V^T V
    """
    m, n = A.shape
    d_inv = 1 / d
    M = np.eye(m) + np.dot(_diag_from_right(A, d_inv), A.T)  # m^2 * n
    M_inv = np.linalg.inv(M)  # m^3
    R = np.linalg.cholesky(M_inv)  # m^3
    V = _diag_from_left(np.dot(A.T, R), d_inv).T  # m^2 * n
    return V, d_inv


def _compute_inverses(A):
    """ Given the low-rank parts of correlation matrices, compute low-rank + diagonal representation
    of inverse correlation matrices.
    :return B, d_inv such that Sigma^{-1}_t = d_inv_t - B_t^T B_t
    """
    nt = len(A)
    d = [None] * nt
    d_inv = [None] * nt
    B = [None] * nt
    for t in range(nt):
        d[t] = 1 - (A[t] ** 2).sum(axis=0)
        d_inv[t] = 1.0 / d[t]

    for t in range(nt):
        B[t], d_inv[t] = _inverse(A[t], d[t])

    return B, d_inv


def _estimate_diff_norm(A_1, d_1, A_2, d_2, n_iters=300):
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


def spectral_diffs_given_factors(factors):
    """ Given the low-rank parts of correlation matrices, compute spectral norm of differences of inverse
    correlation matrices of neighboring time steps.
    """
    B, d_inv = _compute_inverses(factors)
    nt = len(factors)
    diff = [None] * (nt-1)
    for t in range(nt - 1):
        print("Estimating norm of difference at time step: {}".format(t))
        diff[t] = _estimate_diff_norm(B[t], d_inv[t], B[t + 1], d_inv[t + 1])
    return diff


def _compute_diff_norm_fro(A, d_1, B, d_2):
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


def frob_diffs_given_factors(factors):
    """ Given the low-rank parts of correlation matrices, compute Frobenius norm of
    differences of inverse correlation matrices of neighboring time steps.
    """
    B, d_inv = _compute_inverses(factors)
    nt = len(factors)
    diff = [None] * (nt - 1)
    for t in range(nt - 1):
        print("Calculating Frobenius norm of difference at time step: {}".format(t))
        diff[t] = _compute_diff_norm_fro(B[t], d_inv[t], B[t + 1], d_inv[t + 1])
    return diff


def _compute_diff_row_norms(A, d1, B, d2):
    """ Given two low-rank plus diagonal matrices, compute the 2-norm of all
    rows of the difference of those matrices. Total complexity: O(m^2 n).
    """
    m, n = A.shape
    norms = np.zeros(n, dtype=np.float)

    # dome some once, complexity: O(m^2 n)
    AAT = np.dot(A, A.T)
    BBT = np.dot(B, B.T)
    ABT = np.dot(A, B.T)

    for i in range(n):
        d_term = (d1[i] - d2[i]) ** 2
        cross_term = (d1[i] - d2[i]) * (np.sum(A[:, i]**2) - np.sum(B[:, i]**2))
        ab_term = np.dot(np.dot(A[:, i:i+1].T, AAT), A[:, i:i+1])[0, 0]\
            - 2 * np.dot(np.dot(A[:, i:i+1].T, ABT), B[:, i:i+1])[0, 0]\
            + np.dot(np.dot(B[:, i:i+1].T, BBT), B[:, i:i+1])[0, 0]
        norms[i] = d_term + 2 * cross_term + ab_term

    return np.sqrt(norms)


def compute_diff_row_norms(factors):
    """ Given the low-rank parts of correlation matrices, compute the 2-norm of all
    rows of the difference of inverse correlation matrices of neighboring time steps.
    Total complexity: O(T m^2 n).
    """
    B, d_inv = _compute_inverses(factors)
    nt = len(factors)
    row_norms = [None] * (nt - 1)
    for t in range(nt - 1):
        print("Calculating row norms of difference matrix at time step: {}".format(t))
        row_norms[t] = _compute_diff_row_norms(B[t], d_inv[t], B[t+1], d_inv[t+1])
    return row_norms
