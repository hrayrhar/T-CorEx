from __future__ import print_function

from sklearn.preprocessing import StandardScaler
from nose.tools import nottest
import numpy as np
import pandas as pd
import time
import os
import random


@nottest
def load_sp500_with_commodities():
    """ Loads S%P 500 data with commodity prices from Feb, 2005 to March, 2016.
    """
    np.random.seed(42)
    random.seed(42)
    data_dir = 'data/trading_economics'
    df = pd.read_pickle(os.path.join(data_dir, 'sp500_2000-01-01-2018-06-01_raw.pkl'))
    
    # add commodity prices
    commodity = pd.read_pickle(os.path.join(data_dir, 'commodity_prices.pkl'))
    df = pd.concat([df, commodity], axis=0)
    
    # make a table
    df = df.sort_index()
    df = df[['symbol','close']]
    df = df.pivot_table(index=df.index, columns='symbol', values='close')
    df = df[(df.index >= '2004-02-06') & (df.index <= '2016-03-03')]

    df = df.dropna(axis=1, how='all')  # eliminate blank columns
    df = df.fillna(method='ffill')  # forward fill missing dates

    df = np.log(df).diff()[1:]  # calculate log-returns
    df = df.fillna(value=0)  # remaining missing values we treat as no trade, no change
    df = df.drop(df.columns[df.std(axis=0, skipna=True) == 0], axis=1)

    X = np.array(df)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # split the data into buckets
    train_data = []
    val_data = []
    test_data = []

    train_cnt = 15
    val_cnt = 5
    test_cnt = 0
    noise_var = 1e-4

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
        test_data.append(part[perm[train_cnt+val_cnt:]])

    train_data = np.array(train_data)
    val_data = np.array(val_data)
    test_data = np.array(test_data)

    # add small Gaussian noise
    train_data += np.sqrt(noise_var) * np.random.normal(size=train_data.shape)
    val_data   += np.sqrt(noise_var) * np.random.normal(size=val_data.shape)
    test_data  += np.sqrt(noise_var) * np.random.normal(size=test_data.shape)

    print('data is loaded:')
    print('\ttrain shape:', train_data.shape)
    print('\tval   shape:', val_data.shape)
    print("\ttest  shape:", test_data.shape)
    
    return train_data, val_data, test_data