from scipy.stats import multivariate_normal
from sklearn.datasets import make_spd_matrix

import numpy as np
np.random.seed(42)
import sklearn.covariance as skcov


def generate_nglf_from_model(nv, m, nt, ns, snr=None, min_cor=0.8, max_cor=1.0, min_var=1.0, max_var=4.0):
    """ Generates data according to an NGLF model.

    :param nv:      Number of observed variables
    :param m:       Number of latent factors
    :param nt:      Number of time steps
    :param ns:      Number of samples for each time step
    :param snr:     Signal to noise ratio.
                    If `snr` is none `min_cor` will be used
    :param min_cor: Minimum absolute value of correlations between x_i and z_j.
                    This will not be used if `snr` is not none.
    :param max_cor: Maximum absolute value of correlations between x_i and z_j.
                    This will not be used if `snr` is not none.
    :param min_var: Minimum variance of x_i.
    :param max_var: Maximum variance of x_i.
    :return: (data, ground_truth_cov)
    """
    assert nv % m == 0
    block_size = nv // m

    # Generate parameters for p(x,z) joint model
    # NOTE: as z_std doesn't matter, we will set it 1.
    x_std = np.random.uniform(min_var, max_var, size=(nv,))
    cor_signs = np.sign(np.random.normal(size=(nv,)))

    if snr is None:
        cor = cor_signs * np.random.uniform(min_cor, max_cor, size=(nv,))
        snr = np.mean([x ** 2 / (1 - x ** 2) for x in cor])  # TODO: check this (upd: seems correct)
        print "Average SNR: {}".format(snr)
    else:
        cor = cor_signs * np.array([np.sqrt(float(snr) / (snr + 1)) for i in range(nv)])
        print "Fixed SNR: {}".format(snr)

    # Construct the ground truth covariance matrix of x
    ground_truth = np.zeros((nv, nv))
    for i in range(nv):
        for j in range(nv):
            if i // block_size != j // block_size:
                continue
            if i == j:
                ground_truth[i][j] = x_std[i] ** 2
            else:
                ground_truth[i][j] = x_std[i] * cor[i] * x_std[j] * cor[j]

    # Generate Data

    """
    # generates following the probabilistic graphical model
    def generate_single():
        z = [np.random.normal(0.0, v) for v in z_std]
        x = np.zeros((nv,))
        for i in range(nv):
            par = i // block_size
            cond_mean = cor[i] * x_std[i] * z[par]
            cond_var = x_std[i] ** 2 * (1 - cor[i] ** 2)
            x[i] = np.random.normal(cond_mean, np.sqrt(cond_var))
        return x
    """

    def generate_single():
        myu = np.zeros((nv,))
        return np.random.multivariate_normal(myu, ground_truth)

    return ([np.array([generate_single() for i in range(ns)])
             for t in range(nt)], ground_truth)


def generate_nglf_from_matrix(nv, m, nt, ns, param=8, normalize=False):
    """ Generate NGLF data from covariance matrix I + dot(v, v.t), where v is a random vector.

    :param nv:        Number of observed variables
    :param m:         Number of latent factors
    :param nt:        Number of time steps
    :param ns:        Number of samples for each time step
    :param param:     Parameter of uniform distribution used to generate random vector v
    :param normalize: Whether to set Var[x] = 1
    :return: (data, ground_truth_cov)
    """

    assert nv % m == 0
    b = nv // m  # block size

    sigma = np.zeros((nv, nv))
    for i in range(m):
        random_vector = np.random.uniform(-param, +param, size=(b,))
        block_cov = np.eye(b) + np.outer(random_vector, random_vector)
        if normalize:
            std = np.sqrt(block_cov.diagonal()).reshape((b, 1))
            block_cov /= std
            block_cov /= std.T
        sigma[i * b:(i + 1) * b, i * b:(i + 1) * b] = block_cov

    def generate_single():
        myu = np.zeros((nv,))
        return np.random.multivariate_normal(myu, sigma)

    return ([np.array([generate_single() for i in range(ns)])
             for t in range(nt)], sigma)


def generate_general_make_spd(nv, m, nt, ns, normalize=False):
    """ Generate general data using make_spd_matrix() function.

    :param nv:        Number of observed variables
    :param m:         Number of latent factors
    :param nt:        Number of time steps
    :param ns:        Number of samples for each time step
    :param normalize: Whether to set Var[x] = 1
    :return: (data, ground_truth_cov)
    """
    assert nv % m == 0
    b = nv // m  # block size

    sigma = np.zeros((nv, nv))
    for i in range(m):
        block_cov = make_spd_matrix(b)
        if normalize:
            std = np.sqrt(block_cov.diagonal()).reshape((b, 1))
            block_cov /= std
            block_cov /= std.T
        sigma[i * b:(i + 1) * b, i * b:(i + 1) * b] = block_cov

    def generate_single():
        myu = np.zeros((nv,))
        return np.random.multivariate_normal(myu, sigma)

    return ([np.array([generate_single() for i in range(ns)])
             for t in range(nt)], sigma)


def generate_nglf_timeseries(nv, m, nt, ns, snr=None, min_cor=0.8, max_cor=1.0, min_var=1.0, max_var=4.0):
    """ Generates data according to an NGLF model.

    :param nv:      Number of observed variables
    :param m:       Number of latent factors
    :param nt:      Number of time steps
    :param ns:      Number of test samples for each time step
    :param snr:     Signal to noise ratio.
                    If `snr` is none `min_cor` will be used
    :param min_cor: Minimum absolute value of correlations between x_i and z_j.
                    This will not be used if `snr` is not none.
    :param max_cor: Maximum absolute value of correlations between x_i and z_j.
                    This will not be used if `snr` is not none.
    :param min_var: Minimum variance of x_i.
    :param max_var: Maximum variance of x_i.
    :return: (data, ground_truth_cov)
    """
    assert nv % m == 0
    block_size = nv // m

    def generate_sufficient_params(nv, m, snr, min_cor, min_var, max_var):
        # Generate parameters for p(x,z) joint model
        # NOTE: as z_std doesn't matter, we will set it 1.
        x_std = np.random.uniform(min_var, max_var, size=(nv,))
        cor_signs = np.sign(np.random.normal(size=(nv,)))

        if snr is None:
            cor = cor_signs * np.random.uniform(min_cor, max_cor, size=(nv,))
            snr = np.mean([x ** 2 / (1 - x ** 2) for x in cor])  # TODO: check this (upd: seems correct)
            print "Average SNR: {}".format(snr)
        else:
            cor = cor_signs * np.array([np.sqrt(float(snr) / (snr + 1)) for i in range(nv)])
            print "Fixed SNR: {}".format(snr)
        return (x_std, cor)

    def construct_ground_truth(nv, x_std, cor):
        # Construct the ground truth covariance matrix of x
        sigma = np.zeros((nv, nv))
        for i in range(nv):
            for j in range(nv):
                if i // block_size != j // block_size:
                    continue
                if i == j:
                    sigma[i][j] = x_std[i] ** 2
                else:
                    sigma[i][j] = x_std[i] * cor[i] * x_std[j] * cor[j]
        return sigma

    # Generate Data
    (x_std_1, cor_1) = generate_sufficient_params(nv, m, snr, min_cor, min_var, max_var)
    (x_std_2, cor_2) = generate_sufficient_params(nv, m, snr, min_cor, min_var, max_var)

    ground_truth = []
    ts_data = np.zeros((nt, nv))
    test_data = np.zeros((nt, ns, nv))

    alphas = np.linspace(0, 1.0, nt)
    for i, alpha in enumerate(alphas):
        x_std = (1-alpha) * x_std_1 + alpha * x_std_2
        cor = (1 - alpha) * cor_1 + alpha * cor_2
        sigma = construct_ground_truth(nv, x_std, cor)
        ground_truth.append(sigma)
        myu = np.zeros((nv,))
        ts_data[i, :] = np.random.multivariate_normal(myu, sigma)
        test_data[i, :] = np.random.multivariate_normal(myu, sigma, size=(ns,))

    return (ts_data, test_data, ground_truth)
