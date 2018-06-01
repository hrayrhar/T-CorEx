from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from scipy.stats import multivariate_normal
from sklearn.datasets import make_spd_matrix

import numpy as np
import random
import pandas as pd
import sklearn.covariance as skcov
import pickle as pkl


def generate_nglf(nv, m, nt, ns, snr=5.0, min_var=0.25, max_var=4.0, shuffle=False):
    """ Generates data according to an NGLF model.

    :param nv:      Number of observed variables
    :param m:       Number of latent factors
    :param nt:      Number of time steps
    :param ns:      Number of samples for each time step
    :param snr:     Signal to noise ratio
    :param min_var: Minimum variance of x_i
    :param max_var: Maximum variance of x_i
    :param shuffle: Whether to shuffle to x_i's
    :return: (data, ground_truth_cov)
    """

    assert nv % m == 0
    block_size = nv // m

    par = [i // block_size for i in range(nv)]
    if shuffle:
        random.shuffle(par)

    # Generate parameters for p(x,z) joint model
    # NOTE: as z_std doesn't matter, we will set it 1.
    x_std = np.random.uniform(min_var, max_var, size=(nv,))
    cor_signs = np.sign(np.random.normal(size=(nv,)))
    mean_rho = np.sqrt(snr / (snr + 1.0))
    cor = cor_signs * mean_rho
    print("Fixed SNR: {}".format(snr))

    # Construct the ground truth covariance matrix of x
    ground_truth = np.zeros((nv, nv))
    for i in range(nv):
        for j in range(nv):
            if par[i] != par[j]:
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
            cond_mean = cor[i] * x_std[i] * z[par[i]]
            cond_var = x_std[i] ** 2 * (1 - cor[i] ** 2)
            x[i] = np.random.normal(cond_mean, np.sqrt(cond_var))
        return x
    """

    def generate_single():
        myu = np.zeros((nv,))
        return np.random.multivariate_normal(myu, ground_truth)

    return ([np.array([generate_single() for i in range(ns)])
             for t in range(nt)], ground_truth)


def generate_general(nv, m, nt, ns, normalize=False, shuffle=False):
    """ Generate general data using make_spd_matrix() function.

    :param nv:        Number of observed variables
    :param m:         Number of latent factors
    :param nt:        Number of time steps
    :param ns:        Number of samples for each time step
    :param normalize: Whether to set Var[x] = 1
    :param shuffle: Whether to shuffle to x_i's
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

    if shuffle:
        perm = range(nv)
        random.shuffle(perm)
        sigma_perm = np.zeros((nv, nv))
        for i in range(nv):
            for j in range(nv):
                sigma_perm[i, j] = sigma[perm[i], perm[j]]
        sigma = sigma_perm

    def generate_single():
        myu = np.zeros((nv,))
        return np.random.multivariate_normal(myu, sigma)

    return ([np.array([generate_single() for i in range(ns)])
             for t in range(nt)], sigma)


def load_sudden_change(nv, m, nt, train_cnt, val_cnt, test_cnt, snr=5.0,
                       min_var=0.25, max_var=4.0, nglf=True, shuffle=False):
    """ Generate data for the synthetic experiment with sudden change.

    :param nv:         Number of observed variables
    :param m:          Number of latent factors
    :param nt:         Number of time steps
    :param train_cnt:  Number of train samples
    :param val_cnt:    Number of validation samples
    :param test_cnt:   Number of test samples
    :param snr:        Signal to noise ratio
    :param min_var:    Minimum variance of x_i
    :param max_var:    Maximum variance of x_i
    :param nglf:       Whether to use NGLF model
    :param shuffle:    Whether to shuffle to x_i's
    :return: (train_data, val_data, test_data, ground_truth_covs)
    """

    random.seed(42)
    np.random.seed(42)

    if nglf:
        (data1, sigma1) = generate_nglf(nv=nv, m=m, nt=nt // 2, ns=train_cnt + val_cnt + test_cnt,
                                        snr=snr, min_var=min_var, max_var=max_var, shuffle=shuffle)
        # make sure the second generated matrix will be the same no matter of train_cnt
        random.seed(77)
        np.random.seed(77)
        (data2, sigma2) = generate_nglf(nv=nv, m=m, nt=nt // 2, ns=train_cnt + val_cnt + test_cnt,
                                        snr=snr, min_var=min_var, max_var=max_var, shuffle=shuffle)
    else:
        (data1, sigma1) = generate_general_make_spd(nv=nv, m=m, nt=nt // 2, ns=train_cnt + val_cnt + test_cnt,
                                                    shuffle=shuffle)
        # make sure the second generated matrix will be the same no matter of train_cnt
        random.seed(77)
        np.random.seed(77)
        (data2, sigma2) = generate_general_make_spd(nv=nv, m=m, nt=nt // 2, ns=train_cnt + val_cnt + test_cnt,
                                                    shuffle=shuffle)

    data = data1 + data2
    ground_truth_covs = [sigma1 for t in range(nt // 2)] + [sigma2 for t in range(nt // 2)]
    train_data = [x[:train_cnt] for x in data]
    val_data = [x[train_cnt:train_cnt + val_cnt] for x in data]
    test_data = [x[-test_cnt:] for x in data]

    return train_data, val_data, test_data, ground_truth_covs


def load_nglf_smooth_change(nv, m, nt, ns, snr=5.0, min_var=0.25, max_var=4.0):
    """ Generates data for the synthetic experiment with smooth varying NGLF model.

    :param nv:      Number of observed variables
    :param m:       Number of latent factors
    :param nt:      Number of time steps
    :param ns:      Number of test samples for each time step
    :param snr:     Signal to noise ratio
    :param min_var: Minimum variance of x_i
    :param max_var: Maximum variance of x_i
    :return: (data, ground_truth_cov)
    """
    random.seed(42)
    np.random.seed(42)

    assert nv % m == 0
    block_size = nv // m

    def generate_sufficient_params(nv, snr, min_var, max_var):
        # Generate parameters for p(x,z) joint model
        # NOTE: as z_std doesn't matter, we will set it 1.
        x_std = np.random.uniform(min_var, max_var, size=(nv,))
        cor_signs = np.sign(np.random.normal(size=(nv,)))
        mean_rho = np.sqrt(float(snr) / (snr + 1))
        cor = mean_rho * cor_signs
        print("Fixed SNR: {}".format(snr))
        return x_std, cor

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
    (x_std_1, cor_1) = generate_sufficient_params(nv, snr, min_var, max_var)
    (x_std_2, cor_2) = generate_sufficient_params(nv, snr, min_var, max_var)

    ground_truth = []
    data = np.zeros((nt, ns, nv))

    alphas = np.linspace(0, 1.0, nt)
    for i, alpha in enumerate(alphas):
        x_std = (1 - alpha) * x_std_1 + alpha * x_std_2
        cor = (1 - alpha) * cor_1 + alpha * cor_2
        sigma = construct_ground_truth(nv, x_std, cor)
        ground_truth.append(sigma)
        myu = np.zeros((nv,))
        data[i, :] = np.random.multivariate_normal(myu, sigma, size=(ns,))

    return data, ground_truth


def load_stock_data(nt, nv, train_cnt, val_cnt, test_cnt, data_type='stock_day',
                    start_date='2000-01-01', end_date='2018-01-01', stride='one'):
    random.seed(42)
    np.random.seed(42)

    print("Loading stock data ...")
    if data_type == 'stock_week':
        with open('../data/EOD_week.pkl', 'rb') as f:
            df = pd.DataFrame(pkl.load(f))
    elif data_type == 'stock_day':
        with open('../data/EOD_day.pkl', 'rb') as f:
            df = pd.DataFrame(pkl.load(f))
    else:
        raise ValueError("Unrecognized value '{}' for data_type variable".format(data_type))
    df = df[df.index >= start_date]
    df = df[df.index <= end_date]

    # shuffle the columns
    cols = sorted(list(df.columns))
    random.shuffle(cols)
    df = df[cols]

    train_data = []
    val_data = []
    test_data = []

    window = train_cnt + val_cnt + test_cnt
    if stride == 'one':
        indices = range(window, len(df) - window)
    if stride == 'full':
        indices = range(window, len(df) - window, window + 1)
    assert len(indices) >= nt

    for i in indices:
        start = i - window
        end = i + window + 1
        perm = range(2 * window + 1)
        random.shuffle(perm)

        part = np.array(df[start:end])
        assert len(part) == 2 * window + 1

        train_data.append(part[perm[:train_cnt]])
        val_data.append(part[perm[train_cnt:train_cnt + val_cnt]])
        test_data.append(part[perm[-test_cnt:]])

    # take last nt time steps
    train_data = np.array(train_data[-nt:])
    val_data = np.array(val_data[-nt:])
    test_data = np.array(test_data[-nt:])

    # add small gaussian noise
    noise_var = 1e-5
    noise_myu = np.zeros((train_data.shape[-1],))
    noise_cov = np.diag([noise_var] * train_data.shape[-1])
    train_data += np.random.multivariate_normal(noise_myu, noise_cov, size=train_data.shape[:-1])
    val_data += np.random.multivariate_normal(noise_myu, noise_cov, size=val_data.shape[:-1])
    test_data += np.random.multivariate_normal(noise_myu, noise_cov, size=test_data.shape[:-1])

    # find valid variables
    valid_stocks = []
    for i in range(train_data.shape[-1]):
        ok = True
        for t in range(train_data.shape[0]):
            if np.var(train_data[t, :, i]) > 1e-2:
                ok = False
                break
        if ok:
            valid_stocks.append(i)

    # select nv valid variables
    print("\tremained {} variables".format(len(valid_stocks)))
    assert len(valid_stocks) >= nv
    valid_stocks = valid_stocks[:nv]
    train_data = train_data[:, :, valid_stocks]
    val_data = val_data[:, :, valid_stocks]
    test_data = test_data[:, :, valid_stocks]

    # scale the data (this is needed for T-GLASSO to work)
    coef = np.sqrt(np.var(train_data, axis=0).mean())
    train_data = train_data / coef
    val_data = val_data / coef
    test_data = test_data / coef

    print('Stock data is loaded:')
    print('\ttrain shape:', train_data.shape)
    print('\tval   shape:', val_data.shape)
    print('\ttest  shape:', test_data.shape)

    return train_data, val_data, test_data


def load_stock_data_forecasting(nv, train_cnt, val_cnt, test_cnt, data_type='stock_day',
                                start_date='2000-01-01', end_date='2018-01-01'):
    random.seed(42)
    np.random.seed(42)

    print("Loading stock data ...")
    if data_type == 'stock_week':
        with open('../data/EOD_week.pkl', 'rb') as f:
            df = pd.DataFrame(pkl.load(f))
    elif data_type == 'stock_day':
        with open('../data/EOD_day.pkl', 'rb') as f:
            df = pd.DataFrame(pkl.load(f))
    else:
        raise ValueError("Unrecognized value '{}' for data_type variable".format(data_type))
    df = df[df.index >= start_date]
    df = df[df.index <= end_date]

    # shuffle the columns
    cols = sorted(list(df.columns))
    random.shuffle(cols)
    df = df[cols]

    n_samples = train_cnt + val_cnt + test_cnt
    assert n_samples <= len(df)
    df = df[-n_samples:]  # NOTE: this will give just one period. TODO: return multiple periods (sliding window).

    train_data = np.array(df[:train_cnt])
    val_data = np.array(df[train_cnt:train_cnt+val_cnt])
    test_data = np.array(df[-test_cnt:])

    # add small gaussian noise
    noise_var = 1e-5
    noise_myu = np.zeros((train_data.shape[-1],))
    noise_cov = np.diag([noise_var] * train_data.shape[-1])
    train_data += np.random.multivariate_normal(noise_myu, noise_cov, size=train_data.shape[:-1])
    val_data += np.random.multivariate_normal(noise_myu, noise_cov, size=val_data.shape[:-1])
    test_data += np.random.multivariate_normal(noise_myu, noise_cov, size=test_data.shape[:-1])

    # find valid variables
    valid_stocks = []
    for i in range(train_data.shape[-1]):
        if np.var(train_data[:, i]) < 1e-2:
            valid_stocks.append(i)

    # select nv valid variables
    print("\tremained {} variables".format(len(valid_stocks)))
    assert len(valid_stocks) >= nv
    valid_stocks = valid_stocks[:nv]
    train_data = train_data[:, valid_stocks]
    val_data = val_data[:, valid_stocks]
    test_data = test_data[:, valid_stocks]

    # scale the data (this is needed for T-GLASSO to work)
    coef = np.median(np.sqrt(np.var(train_data, axis=0)))  # median works better than mean because of large outliers
    train_data = train_data / coef
    val_data = val_data / coef
    test_data = test_data / coef

    print('Stock data is loaded:')
    print('\ttrain shape:', train_data.shape)
    print('\tval   shape:', val_data.shape)
    print('\ttest  shape:', test_data.shape)

    return train_data, val_data, test_data

# TODO: finalize loading stock data and write down a short documentation
