from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from sklearn.datasets import make_spd_matrix
from sklearn.preprocessing import StandardScaler

import numpy as np
import random
import pandas as pd
import os


def nglf_sufficient_params(nv, m, snr, min_std, max_std):
    """ Generate parameters for p(x,z) joint model.
    """
    # NOTE: as z_std doesn't matter, we will set it 1.
    x_std = np.random.uniform(min_std, max_std, size=(nv,))
    cor_signs = np.sign(np.random.normal(size=(nv,)))
    snrs = np.random.uniform(0, snr, size=(nv,))
    rhos = np.sqrt(snrs / (snrs + 1.0))
    cor = cor_signs * rhos
    par = [np.random.randint(0, m) for _ in range(nv)]
    return x_std, cor, par


def nglf_matrix_from_params(x_std, cor, par):
    """ Construct the covariance matrix corresponding to an NGLF model.
    """
    nv = len(x_std)
    S = np.zeros((nv, nv))
    for i in range(nv):
        for j in range(nv):
            if par[i] != par[j]:
                continue
            if i == j:
                S[i][j] = x_std[i] ** 2
            else:
                S[i][j] = x_std[i] * cor[i] * x_std[j] * cor[j]
    return S


def sample_from_nglf(nv, m, x_std, cor, par, ns, from_matrix=True):
    """ Sample ns from an NGLF model.
    """
    if from_matrix:
        sigma = nglf_matrix_from_params(x_std, cor, par)
        myu = np.zeros((nv,))
        return np.random.multivariate_normal(myu, sigma, size=(ns,)), sigma
    else:
        # generates following the probabilistic graphical model
        def generate_single():
            z = np.random.normal(0.0, 1.0, size=(m,))
            x = np.zeros((nv,))
            for i in range(nv):
                cond_mean = cor[i] * x_std[i] * z[par[i]]
                cond_var = (x_std[i] ** 2) * (1 - cor[i] ** 2)
                x[i] = np.random.normal(cond_mean, np.sqrt(cond_var))
            return x
        data = np.array([generate_single() for _ in range(ns)])
        return data, None


def generate_nglf(nv, m, ns, snr=5.0, min_std=0.25, max_std=4.0, shuffle=False, from_matrix=True):
    """ Generates data according to an NGLF model.

    :param nv:          Number of observed variables
    :param m:           Number of latent factors
    :param ns:          Number of samples
    :param snr:         Average signal to noise ratio (U[0, snr])
    :param min_std:     Minimum std of x_i
    :param max_std:     Maximum std of x_i
    :param shuffle:     Whether to shuffle to x_i's
    :param from_matrix: Whether to construct and return ground truth covariance matrices
    :return: (data, ground_truth_cov)
    """
    block_size = nv // m
    x_std, cor, par = nglf_sufficient_params(nv, m, snr, min_std, max_std)
    if not shuffle:
        par = [i // block_size for i in range(nv)]
    return sample_from_nglf(nv, m, x_std, cor, par, ns, from_matrix)


def generate_general(nv, m, ns, normalize=False, shuffle=False):
    """ Generate general data using make_spd_matrix() function.

    :param nv:        Number of observed variables
    :param m:         Number of latent factors
    :param ns:        Number of samples for each time step
    :param normalize: Whether to set Var[x] = 1
    :param shuffle:   Whether to shuffle to x_i's
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

    mu = np.zeros((nv,))
    return np.random.multivariate_normal(mu, sigma, size=(ns,)), sigma


def load_nglf_sudden_change(nv, m, nt, ns, snr=5.0, min_std=0.25, max_std=4.0, shuffle=False,
                            from_matrix=True, n_segments=2, seed=42):
    """ Generate data for the synthetic experiment with sudden change.

    :param nv:          Number of observed variables
    :param m:           Number of latent factors
    :param nt:          Number of time steps
    :param ns:      Number of samples for each time step
    :param snr:         Average signal to noise ratio (U[0, snr])
    :param min_std:     Minimum std of x_i
    :param max_std:     Maximum std of x_i
    :param shuffle:     Whether to shuffle to x_i's
    :param from_matrix: Whether to construct and return ground truth covariance matrices
                        Valid only when nglf=True
    :param n_segments:  Number of segments with constant cov. matrix
    :param seed:        Seed for np.random and random
    :return: (train_data, val_data, test_data, ground_truth_covs)
    """
    # find segment lengths
    segment_lens = [nt // n_segments for _ in range(n_segments)]
    segment_lens[-1] += nt - sum(segment_lens)
    assert (sum(segment_lens) == nt)

    # generate data
    data = []
    ground_truth_covs = []
    for seg_id in range(n_segments):
        # make sure each time we generate the same nglf model
        random.seed(seed + seg_id)
        np.random.seed(seed + seg_id)
        # generate for the current segment
        cur_ns = segment_lens[seg_id] * ns
        cur_data, cur_sigma = generate_nglf(nv=nv, m=m, ns=cur_ns, snr=snr, min_std=min_std, max_std=max_std,
                                            shuffle=shuffle, from_matrix=from_matrix)
        cur_data = cur_data.reshape((segment_lens[seg_id], ns, nv))
        data += list(cur_data)
        ground_truth_covs += [cur_sigma] * segment_lens[seg_id]

    return data, ground_truth_covs


def load_nglf_smooth_change(nv, m, nt, ns, snr=5.0, min_std=0.25, max_std=4.0, n_segments=2, seed=42):
    """ Generates data for the synthetic experiment with smooth varying NGLF model.

    :param nv:      Number of observed variables
    :param m:       Number of latent factors
    :param nt:      Number of time steps
    :param ns:      Number of samples for each time step
    :param snr:     Average signal to noise ratio (U[0, snr])
    :param min_std: Minimum std of x_i
    :param max_std: Maximum std of x_i
    :param n_segments: Number of segments where cov. matrix is changing smoothly
    :param seed:       Seed for np.random and random
    :return: (data, ground_truth_cov)
    """
    random.seed(seed)
    np.random.seed(seed)

    # find segment lengths and generate sets of sufficient parameters
    segment_lens = [nt // n_segments for _ in range(n_segments)]
    segment_lens[-1] += nt - sum(segment_lens)
    assert(sum(segment_lens) == nt)
    nglfs = [nglf_sufficient_params(nv, m, snr, min_std, max_std)
             for _ in range(n_segments + 1)]

    # generate the data
    ground_truth = []
    data = np.zeros((nt, ns, nv))
    t = 0
    for seg_id in range(n_segments):
        x_std_1, cor_1, par_1 = nglfs[seg_id]
        x_std_2, cor_2, par_2 = nglfs[seg_id + 1]
        L = segment_lens[seg_id]

        # choose where to change the parent of each x_i
        change_points = [np.random.randint(1, L) for _ in range(nv)]

        par = par_1
        for st in range(L):
            # change parents if needed
            for i in range(nv):
                if change_points[i] == st:
                    par[i] = par_2[i]

            # build new sufficient statistics
            alpha = np.float(st) / L
            x_std = (1 - alpha) * x_std_1 + alpha * x_std_2
            cor = (1 - alpha) * cor_1 + alpha * cor_2
            sigma = nglf_matrix_from_params(x_std, cor, par)

            # generate data for a single time step
            ground_truth.append(sigma)
            myu = np.zeros((nv,))
            data[t, :] = np.random.multivariate_normal(myu, sigma, size=(ns,))
            t += 1

    return data, ground_truth


def load_sp500(train_cnt, val_cnt, test_cnt, start_date='2000-01-01', end_date='2018-01-01',
               commodities=False, log_return=True, noise_var=1e-4, return_index=False, seed=42):
    """ Loads S&P 500 data (optionally with commodity prices).
    If nt is None, all time windows will be returned, otherwise only the last nt time windows will be returned.
    """
    np.random.seed(seed)
    random.seed(seed)

    # load S&P 500 data
    data_dir = os.path.join(os.path.dirname(__file__), 'data/trading_economics')
    assert('2000-01-01' <= start_date <= end_date <= '2018-06-01')
    df = pd.read_pickle(os.path.join(data_dir, 'sp500_2000-01-01-2018-06-01_raw.pkl'))

    # load the table containing sectors of stocks
    wiki_table = pd.read_csv(os.path.join(data_dir, 'sp500_components_wiki.csv'))
    symbol_to_category = wiki_table.set_index('Ticker symbol').to_dict()['GICS Sector']
    symbol_to_category = {k + ":US": v for k, v in symbol_to_category.items()}

    # load commodities if needed
    if commodities:
        commodity = pd.read_pickle(os.path.join(data_dir, 'commodity_prices.pkl'))
        df = pd.concat([df, commodity], axis=0)
        meta = pd.read_csv(os.path.join(data_dir, 'commodities_metadata.csv'))
        for (symbol, sector) in zip(meta['symbol'], meta['sector']):
            symbol_to_category[symbol] = "commodity/" + sector

    # make a table
    df = df.sort_index()
    df = df[['symbol', 'close']]
    df = df.pivot_table(index=df.index, columns='symbol', values='close')
    df = df[(df.index >= start_date) & (df.index <= end_date)]  # select the period
    # df = df.dropna(axis=1, how='all')  # eliminate blank columns
    df = df.dropna(axis=1, thresh=int(0.95 * len(df)))  # keep the columns that are mostly present
    df = df.fillna(method='ffill')  # forward fill missing dates

    if log_return:
        df = np.log(df).diff()[1:]
    else:
        df = df.pct_change()[1:]

    df = df.fillna(value=0)  # remaining missing values we treat as no trade, no change
    df = df.drop(df.columns[df.std(axis=0, skipna=True) < 1e-6], axis=1)  # drop columns with zero variance

    # sort columns by sector
    order = list(range(len(df.columns)))
    order = sorted(order, key=lambda x: symbol_to_category[df.columns[x]])
    df = df[np.array(df.columns)[order]]
    symbols = df.columns
    categories = [symbol_to_category[x] for x in df.columns]

    # standardize the raw data
    X = np.array(df)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # split the data into buckets, stride='full' otherwise the same sample can be both in train and val
    train_data = []
    val_data = []
    test_data = []
    window = train_cnt + val_cnt + test_cnt
    indices = range(0, len(df) - window + 1, window)

    for i in indices:
        start = i
        end = i + window
        perm = list(range(window))
        random.shuffle(perm)

        part = np.array(X[start:end])
        assert len(part) == window

        train_data.append(part[perm[:train_cnt]])
        val_data.append(part[perm[train_cnt:train_cnt + val_cnt]])
        test_data.append(part[perm[train_cnt + val_cnt:]])

    train_data = np.array(train_data)
    val_data = np.array(val_data)
    test_data = np.array(test_data)

    # add small Gaussian noise
    train_data += np.sqrt(noise_var) * np.random.normal(size=train_data.shape)
    val_data += np.sqrt(noise_var) * np.random.normal(size=val_data.shape)
    test_data += np.sqrt(noise_var) * np.random.normal(size=test_data.shape)

    print('S&P 500 data is loaded:')
    print('\ttrain shape:', train_data.shape)
    print('\tval   shape:', val_data.shape)
    print("\ttest  shape:", test_data.shape)

    if return_index:
        return train_data, val_data, test_data, symbols, categories, df.index
    return train_data, val_data, test_data, symbols, categories


def load_trading_economics(train_cnt, val_cnt, test_cnt, start_date='2000-01-01', end_date='2018-01-01',
                           log_return=True, noise_var=1e-4, return_index=False, seed=42):
    """ Loads full trading economics data.
    """
    np.random.seed(seed)
    random.seed(seed)

    # load trading economics data all stocks
    data_dir = os.path.join(os.path.dirname(__file__), 'data/trading_economics')
    assert('2000-01-01' <= start_date <= end_date <= '2018-06-01')
    df = pd.read_pickle(os.path.join(data_dir, 'trading_economics_all_stocks_raw.pkl'))

    # make a table
    df = df.sort_index()
    df = df[['symbol', 'close']]
    df = df.pivot_table(index=df.index, columns='symbol', values='close')
    df = df[(df.index >= start_date) & (df.index <= end_date)]  # select the period
    df = df.dropna(axis=1, thresh=int(0.05 * len(df)))  # require at least 5% present values
    df = df.fillna(method='ffill')  # forward fill missing dates
    df = df.drop(columns=df.columns[df.min() < 1e-6])

    if log_return:
        df = np.log(df).diff()[1:]
    else:
        df = df.pct_change()[1:]

    df = df.fillna(value=0)  # remaining missing values we treat as no trade, no change
    df = df.drop(df.columns[df.std(axis=0, skipna=True) < 1e-6], axis=1)  # drop columns with zero variance

    # sort columns by country
    countries = [x[x.find(":")+1:] for x in df.columns]
    order = list(range(len(df.columns)))
    order = sorted(order, key=lambda ind: countries[ind])
    df = df[np.array(df.columns)[order]]
    symbols = df.columns
    countries = [countries[ind] for ind in order]

    # standardize the raw data
    X = np.array(df)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # split the data into buckets, stride='full' otherwise the same sample can be both in train and val
    train_data = []
    val_data = []
    test_data = []
    window = train_cnt + val_cnt + test_cnt
    indices = range(0, len(df) - window + 1, window)

    for i in indices:
        start = i
        end = i + window
        perm = list(range(window))
        random.shuffle(perm)

        part = np.array(X[start:end])
        assert len(part) == window

        train_data.append(part[perm[:train_cnt]])
        val_data.append(part[perm[train_cnt:train_cnt + val_cnt]])
        test_data.append(part[perm[train_cnt + val_cnt:]])

    train_data = np.array(train_data)
    val_data = np.array(val_data)
    test_data = np.array(test_data)

    # add small Gaussian noise
    train_data += np.sqrt(noise_var) * np.random.normal(size=train_data.shape)
    val_data += np.sqrt(noise_var) * np.random.normal(size=val_data.shape)
    test_data += np.sqrt(noise_var) * np.random.normal(size=test_data.shape)

    print('Trading economics is loaded:')
    print('\ttrain shape:', train_data.shape)
    print('\tval   shape:', val_data.shape)
    print("\ttest  shape:", test_data.shape)

    if return_index:
        return train_data, val_data, test_data, symbols, countries, df.index
    return train_data, val_data, test_data, symbols, countries


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
