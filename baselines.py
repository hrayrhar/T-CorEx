import sklearn.decomposition as sk_dec
import sklearn.covariance as sk_cov
import metric_utils
import linearcorex
import theano_time_corex
import numpy as np

import sys
sys.path.append('../TVGL')
import TVGL


class Baseline(object):
    def __init__(self):
        pass

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
    def __init__(self, covs):
        super(GroundTruth, self).__init__()
        self.covs = covs

    def select(self, train_data, val_data, params):
        print "Empty model selection for ground truth baseline"
        return {}

    def evaluate(self, train_data, test_data, params, n_iter, verbose=True):
        if verbose:
            print "Evaluating ground truth baseline ..."
        nll = metric_utils.calculate_nll_score(data=test_data, covs=self.covs)
        return self.report_scores(nll, n_iter)


class Diagonal(Baseline):
    def __init__(self):
        super(Diagonal, self).__init__()

    def select(self, train_data, val_data, params):
        print "Empty model selection of diagonal baseline"
        return {}

    def evaluate(self, train_data, test_data, params, n_iter, verbose=True):
        if verbose:
            print "Evaluating diagonal baseline ..."
        covs = [np.diag(np.var(x, axis=0)) for x in train_data]
        nll = metric_utils.calculate_nll_score(data=test_data, covs=covs)
        return self.report_scores(nll, n_iter)


class LedoitWolf(Baseline):
    def __init__(self):
        super(LedoitWolf, self).__init__()

    def select(self, train_data, val_data, params):
        print "Empty model selection of Ledoit-Wolf baseline"
        return {}

    def evaluate(self, train_data, test_data, params, n_iter, verbose=True):
        if verbose:
            print "Evaluating Ledoit-Wolf baselines ..."
        covs = []
        for x in train_data:
            est = sk_cov.LedoitWolf()
            est.fit(x)
            covs.append(est.covariance_)
        nll = metric_utils.calculate_nll_score(data=test_data, covs=covs)
        return self.report_scores(nll, n_iter)


class OAS(Baseline):
    def __init__(self):
        super(OAS, self).__init__()

    def select(self, train_data, val_data, params):
        print "Empty model selection of oracle approximating shrinkage baseline"
        return {}

    def evaluate(self, train_data, test_data, params, n_iter, verbose=False):
        if verbose:
            print "Evaluating oracle approximating shrinkage baselines ..."
        covs = []
        for x in train_data:
            est = sk_cov.OAS()
            est.fit(x)
            covs.append(est.covariance_)
        nll = metric_utils.calculate_nll_score(data=test_data, covs=covs)
        return self.report_scores(nll, n_iter)


class PCA(Baseline):
    def __init__(self):
        super(PCA, self).__init__()

    def select(self, train_data, val_data, params):
        print "Selecting the best parameter values for PCA ..."
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
        return best_params

    def evaluate(self, train_data, test_data, params, n_iter, verbose=True):
        if verbose:
            print "Evaluating PCA ..."
        try:
            covs = []
            for x in train_data:
                est = sk_dec.PCA(n_components=params['n_components'])
                est.fit(x)
                covs.append(est.get_covariance())
            nll = metric_utils.calculate_nll_score(data=test_data, covs=covs)
        except Exception as e:
            if verbose:
                print "PCA failed with message: {}".format(e.message)
            nll = np.nan
        return self.report_scores(nll, n_iter)


class FactorAnalysis(Baseline):
    def __init__(self):
        super(FactorAnalysis, self).__init__()

    def select(self, train_data, val_data, params):
        print "Selecting the best parameter values for Factor Analysis ..."
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
        return best_params

    def evaluate(self, train_data, test_data, params, n_iter, verbose=True):
        if verbose:
            print "Evaluating Factor Analysis ..."
        try:
            covs = []
            for x in train_data:
                est = sk_dec.FactorAnalysis(n_components=params['n_components'])
                est.fit(x)
                covs.append(est.get_covariance())
            nll = metric_utils.calculate_nll_score(data=test_data, covs=covs)
        except Exception as e:
            if verbose:
                print "Factor analysis failed with message: {}".format(e.message)
            nll = np.nan
        return self.report_scores(nll, n_iter)


class GraphLasso(Baseline):
    def __init__(self):
        super(GraphLasso, self).__init__()

    def select(self, train_data, val_data, params):
        print "Selecting the best parameter values for Graphical Lasso ..."
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
        return best_params

    def evaluate(self, train_data, test_data, params, n_iter, verbose=True):
        if verbose:
            print "Evaluating grahical LASSO for {} iterations ...".format(n_iter)
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
                print "Graphical Lasso failed with message: {}".format(e.message)
            scores = np.nan
        return self.report_scores(scores, n_iter)


class LinearCorex(Baseline):
    def __init__(self):
        super(LinearCorex, self).__init__()

    def select(self, train_data, val_data, params):
        print "Selecting the best parameter values for Linear Corex ..."
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
        return best_params

    def evaluate(self, train_data, test_data, params, n_iter, verbose=True):
        if verbose:
            print "Evaluating linear corex for {} iterations ...".format(n_iter)
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
        return self.report_scores(scores, n_iter)


class TimeVaryingCorex(Baseline):
    def __init__(self):
        super(TimeVaryingCorex, self).__init__()

    def select(self, train_data, val_data, params):
        print "Selecting the best parameter values for Time Varying Linear Corex ..."
        best_score = 1e18
        best_params = None
        reg_params = [('l1', x) for x in params['l1']]
        reg_params += [('l2', x) for x in params['l2']]
        grid_size = len(params['n_hidden']) * len(reg_params)
        done = 0
        for n_hidden in params['n_hidden']:
            for reg_param in reg_params:
                print "\rdone {} / {} {}".format(done, grid_size, ' '*10)
                cur_params = dict({'nt': params['nt'], 'nv': params['nv'], 'n_hidden': n_hidden,
                                   'max_iter': params['max_iter'], 'anneal': True})
                cur_params['l1'] = 0
                cur_params['l2'] = 0
                cur_params[reg_param[0]] = reg_param[1]
                if best_params is None:
                    best_params = cur_params  # just to select one valid set of parameters
                cur_score = self.evaluate(train_data, val_data, cur_params, n_iter=1, verbose=False)['mean']
                if not np.isnan(cur_score) and cur_score < best_score:
                    best_score = cur_score
                    best_params = cur_params
                done += 1
        print "\n"
        return best_params

    def evaluate(self, train_data, test_data, params, n_iter, verbose=True):
        if verbose:
            print "Evaluating time-varying corex for {} iterations ...".format(n_iter)
        scores = []
        for iteration in range(n_iter):
            c = theano_time_corex.TimeCorexSigma(nt=params['nt'],
                                                 nv=params['nv'],
                                                 n_hidden=params['n_hidden'],
                                                 max_iter=params['max_iter'],
                                                 anneal=params['anneal'],
                                                 l1=params['l1'],
                                                 l2=params['l2'])
            c.fit(train_data)
            covs = c.get_covariance()
            cur_nll = metric_utils.calculate_nll_score(data=test_data, covs=covs)
            scores.append(cur_nll)
        return self.report_scores(scores, n_iter)


class TimeVaryingCorexW(Baseline):
    def __init__(self):
        super(TimeVaryingCorexW, self).__init__()

    def select(self, train_data, val_data, params):
        print "Selecting the best parameter values for Time Varying Linear Corex (W) ..."
        best_score = 1e18
        best_params = None
        reg_params = [('l1', x) for x in params['l1']]
        reg_params += [('l2', x) for x in params['l2']]
        grid_size = len(params['n_hidden']) * len(reg_params)
        done = 0
        for n_hidden in params['n_hidden']:
            for reg_param in reg_params:
                print "\rdone {} / {} {}".format(done, grid_size, ' '*10)
                cur_params = dict({'nt': params['nt'], 'nv': params['nv'], 'n_hidden': n_hidden,
                                   'max_iter': params['max_iter'], 'anneal': True})
                cur_params['l1'] = 0
                cur_params['l2'] = 0
                cur_params[reg_param[0]] = reg_param[1]
                if best_params is None:
                    best_params = cur_params  # just to select one valid set of parameters
                cur_score = self.evaluate(train_data, val_data, cur_params, n_iter=1, verbose=False)['mean']
                if not np.isnan(cur_score) and cur_score < best_score:
                    best_score = cur_score
                    best_params = cur_params
                done += 1
        print "\n"
        return best_params

    def evaluate(self, train_data, test_data, params, n_iter, verbose=True):
        if verbose:
            print "Evaluating time-varying corex (W) for {} iterations ...".format(n_iter)
        scores = []
        for iteration in range(n_iter):
            c = theano_time_corex.TimeCorexW(nt=params['nt'],
                                             nv=params['nv'],
                                             n_hidden=params['n_hidden'],
                                             max_iter=params['max_iter'],
                                             anneal=params['anneal'],
                                             l1=params['l1'],
                                             l2=params['l2'])
            c.fit(train_data)
            covs = c.get_covariance()
            cur_nll = metric_utils.calculate_nll_score(data=test_data, covs=covs)
            scores.append(cur_nll)
        return self.report_scores(scores, n_iter)


class TimeVaryingCorexWWT(Baseline):
    def __init__(self):
        super(TimeVaryingCorexWWT, self).__init__()

    def select(self, train_data, val_data, params):
        print "Selecting the best parameter values for Time Varying Linear Corex (WWT) ..."
        best_score = 1e18
        best_params = None
        reg_params = [('l1', x) for x in params['l1']]
        reg_params += [('l2', x) for x in params['l2']]
        grid_size = len(params['n_hidden']) * len(reg_params)
        done = 0
        for n_hidden in params['n_hidden']:
            for reg_param in reg_params:
                print "\rdone {} / {} {}".format(done, grid_size, ' '*10)
                cur_params = dict({'nt': params['nt'], 'nv': params['nv'], 'n_hidden': n_hidden,
                                   'max_iter': params['max_iter'], 'anneal': True})
                cur_params['l1'] = 0
                cur_params['l2'] = 0
                cur_params[reg_param[0]] = reg_param[1]
                if best_params is None:
                    best_params = cur_params  # just to select one valid set of parameters
                cur_score = self.evaluate(train_data, val_data, cur_params, n_iter=1, verbose=False)['mean']
                if not np.isnan(cur_score) and cur_score < best_score:
                    best_score = cur_score
                    best_params = cur_params
                done += 1
        print "\n"
        return best_params

    def evaluate(self, train_data, test_data, params, n_iter, verbose=True):
        if verbose:
            print "Evaluating time-varying corex (WWT) for {} iterations ...".format(n_iter)
        scores = []
        for iteration in range(n_iter):
            c = theano_time_corex.TimeCorexWWT(nt=params['nt'],
                                               nv=params['nv'],
                                               n_hidden=params['n_hidden'],
                                               max_iter=params['max_iter'],
                                               anneal=params['anneal'],
                                               l1=params['l1'],
                                               l2=params['l2'])
            c.fit(train_data)
            covs = c.get_covariance()
            cur_nll = metric_utils.calculate_nll_score(data=test_data, covs=covs)
            scores.append(cur_nll)
        return self.report_scores(scores, n_iter)


class TimeVaryingCorexMI(Baseline):
    def __init__(self):
        super(TimeVaryingCorexMI, self).__init__()

    def select(self, train_data, val_data, params):
        print "Selecting the best parameter values for Time Varying Linear Corex (MI) ..."
        best_score = 1e18
        best_params = None
        reg_params = [('l1', x) for x in params['l1']]
        reg_params += [('l2', x) for x in params['l2']]
        grid_size = len(params['n_hidden']) * len(reg_params)
        done = 0
        for n_hidden in params['n_hidden']:
            for reg_param in reg_params:
                print "\rdone {} / {} {}".format(done, grid_size, ' '*10)
                cur_params = dict({'nt': params['nt'], 'nv': params['nv'], 'n_hidden': n_hidden,
                                   'max_iter': params['max_iter'], 'anneal': True})
                cur_params['l1'] = 0
                cur_params['l2'] = 0
                cur_params[reg_param[0]] = reg_param[1]
                if best_params is None:
                    best_params = cur_params  # just to select one valid set of parameters
                cur_score = self.evaluate(train_data, val_data, cur_params, n_iter=1, verbose=False)['mean']
                if not np.isnan(cur_score) and cur_score < best_score:
                    best_score = cur_score
                    best_params = cur_params
                done += 1
        print "\n"
        return best_params

    def evaluate(self, train_data, test_data, params, n_iter, verbose=True):
        if verbose:
            print "Evaluating time-varying corex (MI) for {} iterations ...".format(n_iter)
        scores = []
        for iteration in range(n_iter):
            c = theano_time_corex.TimeCorexGlobalMI(nt=params['nt'],
                                                    nv=params['nv'],
                                                    n_hidden=params['n_hidden'],
                                                    max_iter=params['max_iter'],
                                                    anneal=params['anneal'],
                                                    l1=params['l1'],
                                                    l2=params['l2'])
            c.fit(train_data)
            covs = c.get_covariance()
            cur_nll = metric_utils.calculate_nll_score(data=test_data, covs=covs)
            scores.append(cur_nll)
        return self.report_scores(scores, n_iter)


class TimeVaryingGraphLasso(Baseline):
    def __init__(self):
        super(TimeVaryingGraphLasso, self).__init__()

    def select(self, train_data, val_data, params):
        print "Selecting the best parameter values for TVGL ..."
        best_score = 1e18
        best_params = None
        grid_size = len(params['lamb']) * len(params['beta']) * len(params['indexOfPenalty'])
        done = 0
        for lamb in params['lamb']:
            for beta in params['beta']:
                for indexOfPenalty in params['indexOfPenalty']:
                    cur_params = dict({'lamb': lamb, 'beta': beta, 'indexOfPenalty': indexOfPenalty,
                                       'max_iter': params['max_iter']})
                    print "\rdone {} / {} {}".format(done, grid_size, ' ' * 10)
                    if best_params is None:
                        best_params = cur_params  # just to select one valid set of parameters
                    cur_score = self.evaluate(train_data, val_data, cur_params, n_iter=1, verbose=False)['mean']
                    if not np.isnan(cur_score) and cur_score < best_score:
                        best_score = cur_score
                        best_params = cur_params
                    done += 1
        print "\n"
        return best_params

    def evaluate(self, train_data, test_data, params, n_iter, verbose=True):
        if verbose:
            print "Evaluating time-varying graphical LASSO for {} iterations ...".format(n_iter)
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
        return self.report_scores(scores, n_iter)
