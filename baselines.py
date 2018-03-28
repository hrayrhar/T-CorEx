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

    def select(self, train_data, val_data, params):
        raise NotImplementedError()

    def evaluate(self, train_data, test_data, params, n_iter, verbose=True):
        raise NotImplementedError()

    def report_scores(self, scores, n_iter):
        if not isinstance(scores, list):
            scores = [scores] * n_iter
        return {"mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "scores": scores}


class GroundTruth(Baseline):
    def __init__(self, covs, test_data, **kwargs):
        super(GroundTruth, self).__init__(**kwargs)
        self.covs = covs
        self.test_data = test_data

    def select(self, train_data, val_data, params):
        print("Empty model selection for {}".format(self.name))
        score = self.evaluate(train_data, val_data, params, n_iter=1, verbose=False)
        return ({}, score)

    def evaluate(self, train_data, test_data, params, n_iter, verbose=True):
        if verbose:
            print("Evaluating {} ...".format(self.name))
        start_time = time.time()
        nll = metric_utils.calculate_nll_score(data=self.test_data, covs=self.covs)
        finish_time = time.time()
        print("\tElapsed time {:.1f}s".format(finish_time - start_time))
        return self.report_scores(nll, n_iter)


class Diagonal(Baseline):
    def __init__(self, **kwargs):
        super(Diagonal, self).__init__(**kwargs)

    def select(self, train_data, val_data, params):
        print("Empty model selection for {}".format(self.name))
        score = self.evaluate(train_data, val_data, params, n_iter=1, verbose=False)
        return ({}, score)

    def evaluate(self, train_data, test_data, params, n_iter, verbose=True):
        if verbose:
            print("Evaluating {} ...".format(self.name))
        start_time = time.time()
        covs = [np.diag(np.var(x, axis=0)) for x in train_data]
        nll = metric_utils.calculate_nll_score(data=test_data, covs=covs)
        finish_time = time.time()
        print("\tElapsed time {:.1f}s".format(finish_time - start_time))
        return self.report_scores(nll, n_iter)


class LedoitWolf(Baseline):
    def __init__(self, **kwargs):
        super(LedoitWolf, self).__init__(**kwargs)

    def select(self, train_data, val_data, params):
        print("Empty model selection for {}".format(self.name))
        score = self.evaluate(train_data, val_data, params, n_iter=1, verbose=False)
        return ({}, score)

    def evaluate(self, train_data, test_data, params, n_iter, verbose=True):
        if verbose:
            print("Evaluating {} ...".format(self.name))
        start_time = time.time()
        covs = []
        for x in train_data:
            est = sk_cov.LedoitWolf()
            est.fit(x)
            covs.append(est.covariance_)
        nll = metric_utils.calculate_nll_score(data=test_data, covs=covs)
        finish_time = time.time()
        print("\tElapsed time {:.1f}s".format(finish_time - start_time))
        return self.report_scores(nll, n_iter)


class OAS(Baseline):
    def __init__(self, **kwargs):
        super(OAS, self).__init__(**kwargs)

    def select(self, train_data, val_data, params):
        print("Empty model selection for {}".format(self.name))
        score = self.evaluate(train_data, val_data, params, n_iter=1, verbose=False)
        return ({}, score)

    def evaluate(self, train_data, test_data, params, n_iter, verbose=False):
        if verbose:
            print("Evaluating {} ...".format(self.name))
        start_time = time.time()
        covs = []
        for x in train_data:
            est = sk_cov.OAS()
            est.fit(x)
            covs.append(est.covariance_)
        nll = metric_utils.calculate_nll_score(data=test_data, covs=covs)
        finish_time = time.time()
        print("\tElapsed time {:.1f}s".format(finish_time - start_time))
        return self.report_scores(nll, n_iter)


class PCA(Baseline):
    def __init__(self, **kwargs):
        super(PCA, self).__init__(**kwargs)

    def select(self, train_data, val_data, params):
        print("Selecting the best parameter values for {} ...".format(self.name))
        best_score = 1e18
        best_params = None
        for n_components in params['n_components']:
            cur_params = {'n_components': n_components}
            if best_params is None:
                best_params = cur_params  # just to select one valid set of parameters
            cur_score = self.evaluate(train_data, val_data, cur_params, n_iter=1, verbose=False)['mean']
            if not np.isnan(cur_score) and cur_score < best_score:
                best_score = cur_score
                best_params = cur_params
        return (best_params, best_score)

    def evaluate(self, train_data, test_data, params, n_iter, verbose=True):
        if verbose:
            print("Evaluating {} ...".format(self.name))
        start_time = time.time()
        try:
            covs = []
            for x in train_data:
                est = sk_dec.PCA(n_components=params['n_components'])
                est.fit(x)
                covs.append(est.get_covariance())
            nll = metric_utils.calculate_nll_score(data=test_data, covs=covs)
        except Exception as e:
            if verbose:
                print("{} failed with message: {}".format(self.name, e.message))
            nll = np.nan
        finish_time = time.time()
        print("\tElapsed time {:.1f}s".format(finish_time - start_time))
        return self.report_scores(nll, n_iter)


class FactorAnalysis(Baseline):
    def __init__(self, **kwargs):
        super(FactorAnalysis, self).__init__(**kwargs)

    def select(self, train_data, val_data, params):
        print("Selecting the best parameter values for {} ...".format(self.name))
        best_score = 1e18
        best_params = None
        for n_components in params['n_components']:
            cur_params = {'n_components': n_components}
            if best_params is None:
                best_params = cur_params  # just to select one valid set of parameters
            cur_score = self.evaluate(train_data, val_data, cur_params, n_iter=1, verbose=False)['mean']
            if not np.isnan(cur_score) and cur_score < best_score:
                best_score = cur_score
                best_params = cur_params
        return (best_params, best_score)

    def evaluate(self, train_data, test_data, params, n_iter, verbose=True):
        if verbose:
            print("Evaluating {} ...".format(self.name))
        start_time = time.time()
        try:
            covs = []
            for x in train_data:
                est = sk_dec.FactorAnalysis(n_components=params['n_components'])
                est.fit(x)
                covs.append(est.get_covariance())
            nll = metric_utils.calculate_nll_score(data=test_data, covs=covs)
        except Exception as e:
            if verbose:
                print("{} failed with message: {}".format(self.name, e.message))
            nll = np.nan
        finish_time = time.time()
        print("\tElapsed time {:.1f}s".format(finish_time - start_time))
        return self.report_scores(nll, n_iter)


class GraphLasso(Baseline):
    def __init__(self, **kwargs):
        super(GraphLasso, self).__init__(**kwargs)

    def select(self, train_data, val_data, params):
        print("Selecting the best parameter values for {} ...".format(self.name))
        best_score = 1e18
        best_params = None
        for alpha in params['alpha']:
            cur_params = {'alpha': alpha, 'max_iter': params['max_iter'], 'mode': params['mode']}
            if best_params is None:
                best_params = cur_params  # just to select one valid set of parameters
            cur_score = self.evaluate(train_data, val_data, cur_params, n_iter=1, verbose=False)['mean']
            if not np.isnan(cur_score) and cur_score < best_score:
                best_score = cur_score
                best_params = cur_params
        return (best_params, best_score)

    def evaluate(self, train_data, test_data, params, n_iter, verbose=True):
        if verbose:
            print("Evaluating {} for {} iterations ...".format(self.name, n_iter))
        start_time = time.time()
        try:
            scores = []
            for iteration in range(n_iter):
                covs = []
                for x in train_data:
                    est = sk_cov.GraphLasso(alpha=params['alpha'],
                                            max_iter=params['max_iter'])
                    est.fit(x)
                    covs.append(est.covariance_)
                cur_nll = metric_utils.calculate_nll_score(data=test_data, covs=covs)
                scores.append(cur_nll)
        except Exception as e:
            if verbose:
                print("{} failed with message: {}".format(self.name, e.message))
            scores = np.nan
        finish_time = time.time()
        print("\tElapsed time {:.1f}s".format(finish_time - start_time))
        return self.report_scores(scores, n_iter)


class LinearCorex(Baseline):
    def __init__(self, **kwargs):
        super(LinearCorex, self).__init__(**kwargs)

    def select(self, train_data, val_data, params):
        print("Selecting the best parameter values for {} ...".format(self.name))
        best_score = 1e18
        best_params = None
        for n_hidden in params['n_hidden']:
            cur_params = {'n_hidden': n_hidden, 'max_iter': params['max_iter'], 'anneal': True}
            if best_params is None:
                best_params = cur_params  # just to select one valid set of parameters
            cur_score = self.evaluate(train_data, val_data, cur_params, n_iter=1, verbose=False)['mean']
            if not np.isnan(cur_score) and cur_score < best_score:
                best_score = cur_score
                best_params = cur_params
        return (best_params, best_score)

    def evaluate(self, train_data, test_data, params, n_iter, verbose=True):
        if verbose:
            print("Evaluating {} for {} iterations ...".format(self.name, n_iter))
        start_time = time.time()
        scores = []
        for iteration in range(n_iter):
            covs = []
            for x in train_data:
                c = linearcorex.Corex(n_hidden=params['n_hidden'],
                                      max_iter=params['max_iter'],
                                      anneal=params['anneal'])
                c.fit(x)
                covs.append(c.get_covariance())
            cur_nll = metric_utils.calculate_nll_score(data=test_data, covs=covs)
            scores.append(cur_nll)
        finish_time = time.time()
        print("\tElapsed time {:.1f}s".format(finish_time - start_time))
        return self.report_scores(scores, n_iter)


class LinearCorexWholeData(LinearCorex):
    def __init__(self, **kwargs):
        super(LinearCorexWholeData, self).__init__(**kwargs)

    def evaluate(self, train_data, test_data, params, n_iter, verbose=True):
        if verbose:
            print("Evaluating {} for {} iterations ...".format(self.name, n_iter))
        start_time = time.time()
        scores = []
        for iteration in range(n_iter):
            X = np.concatenate(train_data, axis=0)
            c = linearcorex.Corex(n_hidden=params['n_hidden'],
                                  max_iter=params['max_iter'],
                                  anneal=params['anneal'])
            c.fit(X)
            covs = [c.get_covariance() for t in range(len(train_data))]
            cur_nll = metric_utils.calculate_nll_score(data=test_data, covs=covs)
            scores.append(cur_nll)
        finish_time = time.time()
        print("\tElapsed time {:.1f}s".format(finish_time - start_time))
        return self.report_scores(scores, n_iter)


class TimeVaryingGraphLasso(Baseline):
    def __init__(self, **kwargs):
        super(TimeVaryingGraphLasso, self).__init__(**kwargs)

    def select(self, train_data, val_data, params):
        print("Selecting the best parameter values for {} ...".format(self.name))
        best_score = 1e18
        best_params = None
        grid_size = len(params['lamb']) * len(params['beta']) * len(params['indexOfPenalty'])
        done = 0
        for lamb in params['lamb']:
            for beta in params['beta']:
                for indexOfPenalty in params['indexOfPenalty']:
                    cur_params = dict({'lamb': lamb, 'beta': beta, 'indexOfPenalty': indexOfPenalty,
                                       'max_iter': params['max_iter']})
                    print("\rdone {} / {} {}".format(done, grid_size, ' ' * 10))
                    if best_params is None:
                        best_params = cur_params  # just to select one valid set of parameters
                    cur_score = self.evaluate(train_data, val_data, cur_params, n_iter=1, verbose=False)['mean']
                    if not np.isnan(cur_score) and cur_score < best_score:
                        best_score = cur_score
                        best_params = cur_params
                    done += 1
        print("\n")
        return (best_params, best_score)

    def evaluate(self, train_data, test_data, params, n_iter, verbose=True):
        if verbose:
            print("Evaluating {} for {} iterations ...".format(self.name, n_iter))
        start_time = time.time()
        # construct time-series
        train_data_ts = []
        for x in train_data:
            train_data_ts += list(x)
        train_data_ts = np.array(train_data_ts)
        scores = []
        for iteration in range(n_iter):
            inv_covs = TVGL.TVGL(data=train_data_ts,
                                 lengthOfSlice=len(train_data[0]),
                                 lamb=params['lamb'],
                                 beta=params['beta'],
                                 indexOfPenalty=params['indexOfPenalty'],
                                 max_iter=params['max_iter'])
            covs = [np.linalg.inv(x) for x in inv_covs]
            cur_nll = metric_utils.calculate_nll_score(data=test_data, covs=covs)
            scores.append(cur_nll)
        finish_time = time.time()
        print("\tElapsed time {:.1f}s, mean score = {}".format(finish_time - start_time, np.mean(scores)))
        return self.report_scores(scores, n_iter)


class TCorex(Baseline):
    def __init__(self, tcorex, **kwargs):
        self.tcorex = tcorex
        super(TCorex, self).__init__(**kwargs)

    def select(self, train_data, val_data, params):
        print("Selecting the best parameter values for {} ...".format(self.name))
        best_score = 1e18
        best_params = None

        const_params = dict()
        search_params = []
        for k, v in params.items():
            if k in ['l1', 'l2']:
                continue
            if isinstance(v, list):
                arr = [(k, x) for x in v]
                search_params.append(arr)
            else:
                const_params[k] = v

        # add L1/L2 regularization parameters
        reg_params = []
        if 'l1' in params:
            reg_params += [('l1', x) for x in params['l1']]
        if 'l2' in params:
            reg_params += [('l2', x) for x in params['l2']]
        if len(reg_params) > 0:
            search_params.append(reg_params)

        # add a dummy variable if the grid is empty
        if len(search_params) == 0:
            search_params = [[('dummy', None)]]

        grid = list(itertools.product(*search_params))

        for index, cur_params in enumerate(grid):
            print("done {} / {}".format(index, len(grid)), end='')
            print(" | running with ", end='')
            for k, v in cur_params:
                print('{}: {}\t'.format(k, v), end='')
            print('')

            cur_params = dict(cur_params)
            for k, v in const_params.items():
                cur_params[k] = v
            cur_params['nt'] = len(train_data)

            if best_params is None:
                best_params = cur_params  # just to select one valid set of parameters
            cur_score = self.evaluate(train_data, val_data, cur_params, n_iter=1, verbose=False)['mean']
            if not np.isnan(cur_score) and cur_score < best_score:
                best_score = cur_score
                best_params = cur_params
        print('\n')
        return (best_params, best_score)

    def evaluate(self, train_data, test_data, params, n_iter, verbose=True):
        if verbose:
            print("Evaluating {} for {} iterations ...".format(self.name, n_iter))
        start_time = time.time()
        scores = []
        for iteration in range(n_iter):
            c = self.tcorex(**params)
            c.fit(train_data)
            covs = c.get_covariance()
            cur_nll = metric_utils.calculate_nll_score(data=test_data, covs=covs)
            scores.append(cur_nll)
        finish_time = time.time()
        print("\tElapsed time {:.1f}s, mean score = {}".format(finish_time - start_time, np.mean(scores)))
        return self.report_scores(scores, n_iter)


""" The following method doesn't train new T-Corex for testing, it just saves the covariance estimate
of the best model and then calculates the scores on the test set.

class TCorex(Baseline):
    def __init__(self, tcorex, **kwargs):
        self.tcorex = tcorex
        super(TCorex, self).__init__(**kwargs)

    def select(self, train_data, val_data, params):
        print("Selecting the best parameter values for {} ...".format(self.name))
        best_score = 1e18
        best_params = None
        self.best_covs = None

        const_params = dict()
        search_params = []
        for k, v in params.items():
            if k in ['l1', 'l2']:
                continue
            if isinstance(v, list):
                arr = [(k, x) for x in v]
                search_params.append(arr)
            else:
                const_params[k] = v

        # add L1/L2 regularization parameters
        reg_params = []
        if 'l1' in params:
            reg_params += [('l1', x) for x in params['l1']]
        if 'l2' in params:
            reg_params += [('l2', x) for x in params['l2']]
        if len(reg_params) > 0:
            search_params.append(reg_params)

        # add a dummy variable if the grid is empty
        if len(search_params) == 0:
            search_params = [[('dummy', None)]]

        grid = list(itertools.product(*search_params))

        for index, cur_params in enumerate(grid):
            print("done {} / {}".format(index, len(grid)), end='')
            print(" | running with ", end='')
            for k, v in cur_params:
                print('{}: {}\t'.format(k, v), end='')
            print('')

            cur_params = dict(cur_params)
            for k, v in const_params.items():
                cur_params[k] = v
            cur_params['nt'] = len(train_data)

            cur_score, covs = self.evaluate(train_data, val_data, cur_params,
                                            n_iter=1, verbose=False, validation=True)
            cur_score = cur_score['mean']

            if (best_params is None) or (not np.isnan(cur_score) and cur_score < best_score):
                best_score = cur_score
                best_params = cur_params
                self.best_covs = covs

        print('\n')
        return (best_params, best_score)

    def evaluate(self, train_data, test_data, params, n_iter, verbose=True, validation=False):
        if verbose:
            print("Evaluating {} for {} iterations ...".format(self.name, n_iter))
        start_time = time.time()

        scores = []
        covs = None
        if validation:
            assert n_iter == 1
            for iteration in range(n_iter):
                c = self.tcorex(**params)
                c.fit(train_data)
                covs = c.get_covariance()
                cur_nll = metric_utils.calculate_nll_score(data=test_data, covs=covs)
                scores.append(cur_nll)
        else:
            scores = [metric_utils.calculate_nll_score(data=test_data, covs=self.best_covs)] * n_iter
        finish_time = time.time()
        print("\tElapsed time {:.1f}s, mean score = {}".format(finish_time - start_time, np.mean(scores)))
        ret = self.report_scores(scores, n_iter)
        if validation:
            ret = (ret, covs)
        return ret
"""