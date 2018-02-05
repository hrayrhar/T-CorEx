import sklearn.decomposition as sk_dec
import sklearn.covariance as sk_cov
import metric_utils
import linearcorex
import theano_time_corex

import sys
sys.path.append('../TVGL')
import TVGL


class Baseline(Object):
    def __init__(self):
        super(Baseline, self).__init__()

    def select(self, train_data, val_data, params):
        raise NotImplementedError()

    def evaluate(self, train_data, test_data, params, n_iter):
        raise NotImplementedError()

    def report_scores(self, scores, n_iter):
        if not isinstance(scores, list):
            scores = [scores] * n_iter
        return {"mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "scores": scores}


class LinearCorex(Baseline):
    def __init__(self):
        super(LinearCorex, self).__init__()

    def select(self, train_data, val_data, params):
        pass

    def evaluate(self, train_data, test_data, params, n_iter):
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
        pass

    def evaluate(self, train_data, test_data, params, n_iter):
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


class Diagonal(Baseline):
    def __init__(self):
        super(Diagonal, self).__init__()

    def select(self, train_data, val_data, params):
        return params

    def evaluate(self, train_data, test_data, params, n_iter):
        print "Evaluating diagonal baseline ..."
        covs = [np.diag(np.var(x, axis=0)) for x in train_data]
        nll = metric_utils.calculate_nll_score(data=test_data, covs=covs)
        return self.report_scores(nll, n_iter)


class LedoitWolf(Baseline):

    def __init__(self):
        super(LedoitWolf, self).__init__()

    def select(self, train_data, val_data, params):
        pass

    def evaluate(self, train_data, test_data, params, n_iter):
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
        pass

    def evaluate(self, train_data, test_data, params, n_iter):
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
        pass

    def evaluate(self, train_data, test_data, params, n_iter):
        print "Evaluating PCA ..."
        covs = []
        for x in train_data:
            est = sk_dec.PCA(n_components=params['n_components'])
            est.fit(x)
            covs.append(est.get_covariance())
        nll = metric_utils.calculate_nll_score(data=test_data, covs=covs)
        return self.report_scores(nll, n_iter)


class FactorAnalysis(Baseline):

    def __init__(self):
        super(FactorAnalysis, self).__init__()

    def select(self, train_data, val_data, params):
        pass

    def evaluate(self, train_data, test_data, params, n_iter):
        print "Evaluating Factor Analysis ..."
        covs = []
        for x in train_data:
            est = sk_dec.FactorAnalysis(n_components=params['n_components'])
            est.fit(x)
            covs.append(est.get_covariance())
        nll = metric_utils.calculate_nll_score(data=test_data, covs=covs)
        return self.report_scores(nll, n_iter)


class GraphLasso(Baseline):

    def __init__(self):
        super(GraphLasso, self).__init__()

    def select(self, train_data, val_data, params):
        pass

    def evaluate(self, train_data, test_data, params, n_iter):
        print "Evaluating grahical LASSO for {} iterations ...".format(n_iter)
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
        return self.report_scores(scores, n_iter)


class TimeVaryingGraphLasso(Baseline):

    def __init__(self):
        super(TimeVaryingGraphLasso, self).__init__()

    def select(self, train_data, val_data, params):
        pass

    def evaluate(self, train_data, test_data, params, n_iter):
        print "Evaluating time-varying graphical LASSO for {} iterations ...".format(n_iter)
        # construct time-series
        train_data_ts = []
        for x in train_data:
            train_data_ts += x
        train_data_ts = np.array(train_data_ts)
        scores = []
        for iteration in range(n_iter):
            inv_covs = TVGL(data=train_data_ts,
                            lengthOfSlice=params['lengthOfSlice'],
                            lamb=params['lamb'],
                            beta=params['beta'],
                            indexOfPenalty=params['indexOfPenalty'])
            covs = [np.linalg.inv(x) for x in inv_covs]
            cur_nll = metric_utils.calculate_nll_score(data=test_data, covs=covs)
            scores.append(cur_nll)
        return self.report_scores(scores, n_iter)
