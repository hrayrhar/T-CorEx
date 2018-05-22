from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sklearn.decomposition as sk_dec
import sklearn.covariance as sk_cov
import metric_utils
import linearcorex
import numpy as np
import time
import itertools

import sys
sys.path.append('../TVGL')
import TVGL


class Baseline(object):
    def __init__(self, name):
        self.name = name
        self._trained = False
        self._val_score = None
        self._params = None
        self._cov = None
        self._method = None

    def select(self, train_data, val_data, params, verbose=True):
        if verbose:
            print("\n{}\nSelecting the best parameter values for {} ...".format('-'*80, self.name))

        best_score = 1e18
        best_params = None

        const_params = dict()
        search_params = []
        for k, v in params.items():
            if isinstance(v, list):
                arr = [(k, x) for x in v]
                search_params.append(arr)
            elif isinstance(v, dict):
                arr = []
                for param_k, param_v in v.items():
                    arr += list([(param_k, x) for x in param_v])
                search_params.append(arr)
            else:
                const_params[k] = v

        # add a dummy variable if the grid is empty
        if len(search_params) == 0:
            search_params = [[('__dummy__', None)]]

        grid = list(itertools.product(*search_params))

        for index, cur_params in enumerate(grid):
            if verbose:
                print("done {} / {}".format(index, len(grid)), end='')
                print(" | running with ", end='')
                for k, v in cur_params:
                    if k != '__dummy__':
                        print('{}: {}\t'.format(k, v), end='')
                print('')

            cur_params = dict(cur_params)
            for k, v in const_params.items():
                cur_params[k] = v
            (cur_cov, cur_method) = self._train(train_data, cur_params, verbose)
            cur_score = metric_utils.calculate_nll_score(data=[val_data], covs=[cur_cov])

            if verbose:
                print('\tcurrent score: {}'.format(cur_score))

            if (best_params is None) or (not np.isnan(cur_score) and cur_score < best_score):
                best_score = cur_score
                best_params = cur_params

        if verbose:
            print('\nFinished with best validation score: {}'.format(best_score))
            print("Training the best method again with validation data included ...")

        self._trained = True
        self._val_score = best_score
        self._params = best_params
        (cur_cov, cur_method) = self._train(np.concatenate([train_data, val_data], axis=0), self._params, verbose)
        self._cov = cur_cov
        self._method = cur_method

        return (best_score, best_params, cur_cov, cur_method)

    def _train(self, train_data, params, verbose):
        # should return a pair: (cov, method)
        raise NotImplementedError()

    def evaluate(self, test_data, verbose=True):
        assert self._trained
        if verbose:
            print("Evaluating {} ...".format(self.name))
        nll = metric_utils.calculate_nll_score(data=[test_data], covs=[self._cov])
        if verbose:
            print("\tScore: {:.4f}".format(nll))
        return nll

    def get_covariance(self):
        assert self._trained
        return self._cov

    def timeit(self, train_data, params):
        start_time = time.time()
        dummy = self._train(train_data, params, verbose=False)
        finish_time = time.time()
        return finish_time - start_time


class GroundTruth(Baseline):
    def __init__(self, covs, test_data, **kwargs):
        super(GroundTruth, self).__init__(**kwargs)
        self._score = metric_utils.calculate_nll_score(data=test_data, covs=covs)
        self._covs = covs
        self._trained = True

    def _train(self, train_data, params, verbose):
        return (self._covs, None)


class Diagonal(Baseline):
    def __init__(self, **kwargs):
        super(Diagonal, self).__init__(**kwargs)

    def _train(self, train_data, params, verbose):
        X = train_data[-params['sample_cnt']:]
        if verbose:
            print("Training {} ...".format(self.name))
        start_time = time.time()
        cov = np.diag(np.var(X, axis=0))
        finish_time = time.time()
        if verbose:
            print("\tElapsed time {:.1f}s".format(finish_time - start_time))
        return (cov, None)


class LedoitWolf(Baseline):
    def __init__(self, **kwargs):
        super(LedoitWolf, self).__init__(**kwargs)

    def _train(self, train_data, params, verbose):
        X = train_data[-params['sample_cnt']:]
        if verbose:
            print("Training {} ...".format(self.name))
        start_time = time.time()
        est = sk_cov.LedoitWolf()
        est.fit(X)
        cov = est.covariance_
        finish_time = time.time()
        if verbose:
            print("\tElapsed time {:.1f}s".format(finish_time - start_time))
        return (cov, None)


class OAS(Baseline):
    def __init__(self, **kwargs):
        super(OAS, self).__init__(**kwargs)

    def _train(self, train_data, params, verbose):
        X = train_data[-params['sample_cnt']:]
        if verbose:
            print("Training {} ...".format(self.name))
        start_time = time.time()
        est = sk_cov.OAS()
        est.fit(X)
        cov = est.covariance_
        finish_time = time.time()
        if verbose:
            print("\tElapsed time {:.1f}s".format(finish_time - start_time))
        return (cov, None)


class PCA(Baseline):
    def __init__(self, **kwargs):
        super(PCA, self).__init__(**kwargs)

    def _train(self, train_data, params, verbose):
        X = train_data[-params['sample_cnt']:]
        if verbose:
            print("Training {} ...".format(self.name))
        start_time = time.time()
        try:
            est = sk_dec.PCA(n_components=params['n_components'])
            est.fit(X)
            cov = est.get_covariance()
        except Exception as e:
            cov = None
            if verbose:
                print("\t{} failed with message: {}".format(self.name, e.message))
        finish_time = time.time()
        if verbose:
            print("\tElapsed time {:.1f}s".format(finish_time - start_time))
        return (cov, None)


class FactorAnalysis(Baseline):
    def __init__(self, **kwargs):
        super(FactorAnalysis, self).__init__(**kwargs)

    def _train(self, train_data, params, verbose):
        X = train_data[-params['sample_cnt']:]
        if verbose:
            print("Training {} ...".format(self.name))
        start_time = time.time()
        try:
            est = sk_dec.FactorAnalysis(n_components=params['n_components'])
            est.fit(X)
            cov = est.get_covariance()
        except Exception as e:
            cov = None
            if verbose:
                print("\t{} failed with message: {}".format(self.name, e.message))
        finish_time = time.time()
        if verbose:
            print("\tElapsed time {:.1f}s".format(finish_time - start_time))
        return (cov, None)


class GraphLasso(Baseline):
    def __init__(self, **kwargs):
        super(GraphLasso, self).__init__(**kwargs)

    def _train(self, train_data, params, verbose):
        X = train_data[-params['sample_cnt']:]
        if verbose:
            print("Training {} ...".format(self.name))
        start_time = time.time()
        try:
            est = sk_cov.GraphLasso(alpha=params['alpha'],
                                    max_iter=params['max_iter'])
            est.fit(X)
            cov = est.covariance_
        except Exception as e:
            if verbose:
                print("\t{} failed with message: {}".format(self.name, e.message))
            cov = None
        finish_time = time.time()
        if verbose:
            print("\tElapsed time {:.1f}s".format(finish_time - start_time))
        return (cov, None)


class LinearCorex(Baseline):
    def __init__(self, **kwargs):
        super(LinearCorex, self).__init__(**kwargs)

    def _train(self, train_data, params, verbose):
        X = train_data[-params['sample_cnt']:]
        if verbose:
            print("Training {} ...".format(self.name))
        start_time = time.time()
        c = linearcorex.Corex(n_hidden=params['n_hidden'],
                              max_iter=params['max_iter'],
                              anneal=params['anneal'])
        c.fit(X)
        cov = c.get_covariance()
        finish_time = time.time()
        if verbose:
            print("\tElapsed time {:.1f}s".format(finish_time - start_time))
        return (cov, None)


def make_buckets(data, bucket_size):
    data_size = data.shape[0]
    rem = data_size % bucket_size
    buckets = []
    for start in range(rem, data_size, bucket_size):
        buckets.append(data[start:start + bucket_size])
    return buckets


class TimeVaryingGraphLasso(Baseline):
    def __init__(self, **kwargs):
        super(TimeVaryingGraphLasso, self).__init__(**kwargs)

    def _train(self, train_data, params, verbose):
        train_data = make_buckets(train_data, params['bucket_size'])
        train_data = np.concatenate(train_data, axis=0)  # make time-series
        if verbose:
            print("Training {} ...".format(self.name))
        start_time = time.time()
        inv_covs = TVGL.TVGL(data=train_data,
                             lengthOfSlice=params['bucket_size'],
                             lamb=params['lamb'],
                             beta=params['beta'],
                             indexOfPenalty=params['indexOfPenalty'],
                             max_iter=params['max_iter'])
        cov = np.linalg.inv(inv_covs[-1])
        finish_time = time.time()
        if verbose:
            print("\tElapsed time {:.1f}s".format(finish_time - start_time))
        return (cov, None)

    def timeit(self, train_data, params):
        # need to write special timeit() to exclude the time spent for linalg.inv()
        train_data = make_buckets(data, params['bucket_size'])
        train_data = np.concatenate(train_data, axis=0)  # make time-series
        start_time = time.time()
        inv_covs = TVGL.TVGL(data=train_data,
                             lengthOfSlice=params['bucket_size'],
                             lamb=params['lamb'],
                             beta=params['beta'],
                             indexOfPenalty=params['indexOfPenalty'],
                             max_iter=params['max_iter'])
        finish_time = time.time()
        return finish_time - start_time


class TCorex(Baseline):
    def __init__(self, tcorex, **kwargs):
        self.tcorex = tcorex
        super(TCorex, self).__init__(**kwargs)

    def _train(self, train_data, params, verbose):
        bucket_size = params['bucket_size']
        train_data = make_buckets(train_data, bucket_size)
        if verbose:
            print("Training {} ...".format(self.name))
        start_time = time.time()

        params['nt'] = len(train_data)
        del params['bucket_size']
        c = self.tcorex(**params)
        params['bucket_size'] = bucket_size
        c.fit(train_data)
        covs = c.get_covariance()

        finish_time = time.time()
        if verbose:
            print("\tElapsed time {:.1f}s".format(finish_time - start_time))
        return (covs, c)
