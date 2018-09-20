from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np
from scipy.stats import norm, rankdata
import time
import random

import lasagne


def g(x, t=4):
    """A transformation that suppresses outliers for a standard normal."""
    xp = np.clip(x, -t, t)
    diff = np.tanh(x - xp)
    return xp + diff


def g_inv(x, t=4):
    """Inverse of g transform."""
    xp = np.clip(x, -t, t)
    diff = np.arctanh(np.clip(x - xp, -1 + 1e-10, 1 - 1e-10))
    return xp + diff


def mean_impute(x, v):
    """Missing values in the data, x, are indicated by v. Wherever this value appears in x, it is replaced by the
    mean value taken from the marginal distribution of that column."""
    if not np.isnan(v):
        x = np.where(x == v, np.nan, x)
    x_new = []
    n_obs = []
    for i, xi in enumerate(x.T):
        missing_locs = np.where(np.isnan(xi))[0]
        xi_nm = xi[np.isfinite(xi)]
        xi[missing_locs] = np.mean(xi_nm)
        x_new.append(xi)
        n_obs.append(len(xi_nm))
    return np.array(x_new).T, np.array(n_obs)


def get_w_from_corex(corex):
    u = corex.ws
    z2 = corex.moments['Y_j^2']
    return u * np.sqrt(z2).reshape((-1, 1))


class Corex:
    def __init__(self, nv, n_hidden=10, max_iter=10000, tol=1e-5, anneal=True, missing_values=None, update_iter=15,
                 gaussianize='standard', y_scale=1.0, l1=0.0, verbose=0, seed=None):

        self.nv = nv  # Number of variables
        self.m = n_hidden  # Number of latent factors to learn
        self.max_iter = max_iter  # Number of iterations to try
        self.tol = tol  # Threshold for convergence
        self.anneal = anneal
        self.eps = 0  # If anneal is True, it's adjusted during optimization to avoid local minima
        self.missing_values = missing_values
        self.update_iter = update_iter  # Compute statistics every update_iter

        self.gaussianize = gaussianize  # Preprocess data: 'standard' scales to zero mean and unit variance

        self.y_scale = y_scale  # Can be arbitrary, but sets the scale of Y
        np.random.seed(seed)  # Set seed for deterministic results
        self.l1 = l1  # L1 on W regularization coefficient
        self.verbose = verbose
        if verbose:
            np.set_printoptions(precision=3, suppress=True, linewidth=160)
            print('Linear CorEx with {:d} latent factors'.format(n_hidden))

        self.history = {}  # Keep track of values for each iteration
        self.rng = RandomStreams(seed)

    def _define_model(self, anneal_eps):
        self.x_wno = T.matrix('X')
        ns = self.x_wno.shape[0]
        self.anneal_noise = self.rng.normal(size=(ns, self.nv))
        self.x = np.sqrt(1 - anneal_eps ** 2) * self.x_wno + anneal_eps * self.anneal_noise
        self.z_noise = self.rng.normal(avg=0.0, std=self.y_scale, size=(ns, self.m))
        self.ws = theano.shared(1.0 / np.sqrt(self.nv) * np.random.randn(self.m, self.nv), name='W')
        self.z_mean = T.dot(self.x, self.ws.T)
        self.z = self.z_mean + self.z_noise

        epsilon = 1e-8
        z2 = (self.z ** 2).mean(axis=0)  # (m,)
        R = T.dot(self.z.T, self.x) / ns  # m, nv
        R = R / T.sqrt(z2).reshape((self.m, 1))  # as <x^2_i> == 1 we don't divide by it
        ri = ((R ** 2) / T.clip(1 - R ** 2, epsilon, 1 - epsilon)).sum(axis=0)  # (nv,)

        # v_xi | z conditional mean
        outer_term = (1 / (1 + ri)).reshape((1, self.nv))
        inner_term_1 = R / T.clip(1 - R ** 2, epsilon, 1) / T.sqrt(z2).reshape((self.m, 1))  # (m, nv)
        inner_term_2 = self.z  # (ns, m)
        cond_mean = outer_term * T.dot(inner_term_2, inner_term_1)  # (ns, nv)

        # objective
        obj_part_1 = 0.5 * T.log(T.clip(((self.x - cond_mean) ** 2).mean(axis=0), epsilon, np.inf)).sum(axis=0)
        obj_part_2 = 0.5 * T.log(z2).sum(axis=0)
        reg_obj = T.constant(0)
        if self.l1 > 0:
            reg_obj = T.sum(self.l1 * T.abs_(self.ws))

        self.obj = obj_part_1 + obj_part_2 + reg_obj

        # optimizer
        updates = lasagne.updates.adam(self.obj, [self.ws])

        # functions
        self.train_step = theano.function(inputs=[self.x_wno],
                                          outputs=[self.obj, self.z_mean],
                                          updates=updates)

    def fit(self, x):
        x = np.asarray(x, dtype=np.float32)
        x = self.preprocess(x, fit=True)  # Fit a transform for each marginal
        assert x.shape[1] == self.nv
        self.x_std = x  # to have access to the standardized version of input

        anneal_schedule = [0.]
        if self.anneal:
            anneal_schedule = [0.6 ** k for k in range(1, 7)] + [0]

        for i_eps, eps in enumerate(anneal_schedule):
            start_time = time.time()
            self.eps = eps
            if i_eps > 0:
                old_ws = self.ws.get_value()
                self._define_model(eps)
                self.ws.set_value(old_ws)
            else:
                self._define_model(eps)

            self.moments = self._calculate_moments(x, self.ws, quick=True)
            self._update_u(x)

            for i_loop in range(self.max_iter):
                # TODO: write a stopping condition
                if self.verbose:
                    print("annealing eps: {}, iter: {} / {}".format(eps, i_loop, self.max_iter), end='\r')

                (obj, _) = self.train_step(x.astype(np.float32))

                if self.verbose > 1 and i_loop % self.update_iter == 0:
                    self.moments = self._calculate_moments(x, self.ws, quick=True)
                    self._update_u(x)
                    print("tc = {}, obj = {}, eps = {}".format(self.tc, obj, eps))

            print("Annealing iteration finished, time = {}".format(time.time() - start_time))

        self.moments = self._calculate_moments(x, self.ws, quick=False)
        order = np.argsort(-self.moments["TCs"])  # Largest TC components first.
        ws = self.ws.get_value()
        ws = ws[order]
        self.ws.set_value(ws)
        self._update_u(x)
        self.moments = self._calculate_moments(x, self.ws, quick=False)  # Update moments based on sorted weights.
        return self

    def _update_u(self, x):
        self.us = self.getus(self.ws.get_value(), x)

    @property
    def tc(self):
        """This actually returns the lower bound on TC that is optimized. The lower bound assumes a constraint that
         would be satisfied by a non-overlapping model.
         Check "moments" for two other estimates of TC that may be useful."""
        return self.moments["TC"]

    @property
    def tcs(self):
        """TCs for each individual latent factor. They DO NOT sum to TC in this case, because of overlaps."""
        return self.moments["TCs"]

    @property
    def mis(self):
        return - 0.5 * np.log1p(-self.moments["rho"] ** 2)

    def clusters(self):
        return np.argmax(np.abs(self.us), axis=0)

    def _sig(self, x, u):
        """Multiple the matrix u by the covariance matrix of x. We are interested in situations where
        n_variables >> n_samples, so we do this without explicitly constructing the covariance matrix."""
        n_samples = x.shape[0]
        y = x.dot(u.T)
        tmp_dot = x.T.dot(y)
        prod = (1 - self.eps ** 2) * tmp_dot.T / n_samples + self.eps ** 2 * u  # nv by m,  <X_i Y_j> / std Y_j
        return prod

    def getus(self, w, x):
        # U_{ji} = \frac{W_{ji}}{\sqrt{E[Z_j^2]}}
        """
        U = np.zeros(W.shape)
        for j in range(W.shape[0]):
            U[j, :] = W[j, :] / np.sqrt(self.z2_fromW(j, W, x))
        return U
        """
        tmp_dot = np.dot(self._sig(x, w), w.T)
        z2 = self.y_scale ** 2 + np.einsum("ii->i", tmp_dot)
        return w / np.sqrt(z2).reshape((-1, 1))

    def _calculate_moments(self, x, ws, quick=False):
        us = self.getus(ws.get_value(), x)
        return self._calculate_moments_ns(x, us, quick=quick)

    def _calculate_moments_ns(self, x, ws, quick=False):
        """Calculate moments based on the weights and samples. We also calculate and save MI, TC, additivity, and
        the value of the objective. Note it is assumed that <X_i^2> = 1! """
        n_samples = x.shape[0]
        m = {}  # Dictionary of moments
        y = x.dot(ws.T)
        tmp_sum = np.einsum('lj,lj->j', y, y)
        m["uj"] = (1 - self.eps ** 2) * tmp_sum / n_samples + self.eps ** 2 * np.sum(ws ** 2, axis=1)

        if np.max(m["uj"]) > 1.0:
            print(np.max(m["uj"]))
            assert False

        tmp_dot = x.T.dot(y)
        m["rho"] = (1 - self.eps ** 2) * tmp_dot.T / n_samples + self.eps ** 2 * ws  # m by nv
        m["ry"] = ws.dot(m["rho"].T)  # normalized covariance of Y
        m["Y_j^2"] = self.y_scale ** 2 / (1. - m["uj"])
        np.fill_diagonal(m["ry"], 1)
        m["invrho"] = 1. / (1. - m["rho"] ** 2)
        m["rhoinvrho"] = m["rho"] * m["invrho"]
        m["Qij"] = np.dot(m['ry'], m["rhoinvrho"])
        m["Qi"] = np.einsum('ki,ki->i', m["rhoinvrho"], m["Qij"])
        # m["Qi-Si^2"] = np.einsum('ki,ki->i', m["rhoinvrho"], m["Qij"])
        m["Si"] = np.sum(m["rho"] * m["rhoinvrho"], axis=0)

        # This is the objective, a lower bound for TC
        m["TC"] = np.sum(np.log(1 + m["Si"])) \
                  - 0.5 * np.sum(np.log(1 - m["Si"] ** 2 + m["Qi"])) \
                  + 0.5 * np.sum(np.log(1 - m["uj"]))

        if not quick:
            m["MI"] = - 0.5 * np.log1p(-m["rho"] ** 2)
            m["X_i Y_j"] = m["rho"].T * np.sqrt(m["Y_j^2"])
            m["X_i Z_j"] = np.linalg.solve(m["ry"], m["rho"]).T
            m["X_i^2 | Y"] = (1. - np.einsum('ij,ji->i', m["X_i Z_j"], m["rho"])).clip(1e-6)
            m['I(Y_j ; X)'] = 0.5 * np.log(m["Y_j^2"]) - 0.5 * np.log(self.y_scale ** 2)
            m['I(X_i ; Y)'] = - 0.5 * np.log(m["X_i^2 | Y"])
            m["TCs"] = m["MI"].sum(axis=1) - m['I(Y_j ; X)']
            m["TC_no_overlap"] = m["MI"].max(axis=0).sum() - m[
                'I(Y_j ; X)'].sum()  # A direct calculation of TC where each variable is in exactly one group.
            m["TC_direct"] = m['I(X_i ; Y)'].sum() - m[
                'I(Y_j ; X)']  # A direct calculation of TC. Should be upper bound for "TC", "TC_no_overlap"
            m["additivity"] = (m["MI"].sum(axis=0) - m['I(X_i ; Y)']).sum()
        return m

    def transform(self, x, details=False):
        """Transform an array of inputs, x, into an array of k latent factors, Y.
            Optionally, you can get the remainder information and/or stop at a specified level."""
        x = self.preprocess(x)
        ns, nv = x.shape
        assert self.nv == nv, "Incorrect number of variables in input, %d instead of %d" % (nv, self.nv)
        if details:
            moments = self._calculate_moments(x, self.us)
            return x.dot(self.us.T), moments
        return x.dot(self.us.T)

    def preprocess(self, x, fit=False):
        """Transform each marginal to be as close to a standard Gaussian as possible.
        'standard' (default) just subtracts the mean and scales by the std.
        'empirical' does an empirical gaussianization (but this cannot be inverted).
        'outliers' tries to squeeze in the outliers
        Any other choice will skip the transformation."""
        if self.missing_values is not None:
            x, self.n_obs = mean_impute(x, self.missing_values)  # Creates a copy
        else:
            self.n_obs = len(x)
        if self.gaussianize == 'none':
            pass
        elif self.gaussianize == 'standard':
            if fit:
                mean = np.mean(x, axis=0)
                # std = np.std(x, axis=0, ddof=0).clip(1e-10)
                std = np.sqrt(np.sum((x - mean) ** 2, axis=0) / self.n_obs).clip(1e-10)
                self.theta = (mean, std)
            x = ((x - self.theta[0]) / self.theta[1])
            if np.max(np.abs(x)) > 6 and self.verbose:
                print("Warning: outliers more than 6 stds away from mean. Consider using gaussianize='outliers'")
        elif self.gaussianize == 'outliers':
            if fit:
                mean = np.mean(x, axis=0)
                std = np.std(x, axis=0, ddof=0).clip(1e-10)
                self.theta = (mean, std)
            x = g((x - self.theta[0]) / self.theta[1])  # g truncates long tails
        elif self.gaussianize == 'empirical':
            print("Warning: correct inversion/transform of empirical gauss transform not implemented.")
            x = np.array([norm.ppf((rankdata(x_i) - 0.5) / len(x_i)) for x_i in x.T]).T
        return x

    def invert(self, x):
        """Invert the preprocessing step to get x's in the original space."""
        if self.gaussianize == 'standard':
            return self.theta[1] * x + self.theta[0]
        elif self.gaussianize == 'outliers':
            return self.theta[1] * g_inv(x) + self.theta[0]
        else:
            return x

    def predict(self, y):
        return self.invert(np.dot(self.moments["X_i Z_j"], y.T).T)

    def get_covariance(self):
        # This uses E(Xi|Y) formula for non-synergistic relationships
        m = self.moments
        z = m['rhoinvrho'] / (1 + m['Si'])
        cov = np.dot(z.T, z)
        cov /= (1. - self.eps ** 2)
        np.fill_diagonal(cov, 1)
        return self.theta[1][:, np.newaxis] * self.theta[1] * cov


class TCorexBase(object):
    def __init__(self, nt, nv, n_hidden=10, max_iter=10000, tol=1e-5, anneal=True, missing_values=None,
                 gaussianize='standard', y_scale=1.0, update_iter=10, pretrained_weights=None, verbose=0,
                 seed=None, ignore_sigma=False):

        self.nt = nt  # Number of timesteps
        self.nv = nv  # Number of variables
        self.m = n_hidden  # Number of latent factors to learn
        self.max_iter = max_iter  # Number of iterations to try
        self.tol = tol  # Threshold for convergence
        self.anneal = anneal
        self.eps = 0  # If anneal is True, it's adjusted during optimization to avoid local minima
        self.missing_values = missing_values

        self.gaussianize = gaussianize  # Preprocess data: 'standard' scales to zero mean and unit variance

        self.y_scale = y_scale  # Can be arbitrary, but sets the scale of Y
        self.update_iter = update_iter  # Compute statistics every update_iter
        self.pretrained_weights = pretrained_weights
        np.random.seed(seed)  # Set seed for deterministic results
        self.verbose = verbose
        self.ignore_sigma = ignore_sigma
        if verbose:
            np.set_printoptions(precision=3, suppress=True, linewidth=160)
            print('Linear CorEx with {:d} latent factors'.format(n_hidden))

        self.history = [{} for t in range(self.nt)]  # Keep track of values for each iteration
        self.rng = RandomStreams(seed)

    def _define_model(self):
        raise NotImplementedError("_define_model function should be specified for all child classes")

    def _train_loop(self, x):
        """
        :param x: is the standardized input (mean ~=0, std ~= 1).
        """
        if self.verbose:
            print("Starting the training loop ...")
        # set the annealing schedule
        anneal_schedule = [0.]
        if self.anneal:
            anneal_schedule = [0.6 ** k for k in range(1, 7)] + [0]

        # initialize the weights if pre-trained weights are specified
        if self.pretrained_weights is not None:
            for cur_w, pre_w in zip(self.ws, self.pretrained_weights):
                cur_w.set_value(pre_w.astype(np.float32))

        for eps in anneal_schedule:
            start_time = time.time()
            self.eps = eps
            self.anneal_eps.set_value(np.float32(eps))
            self.moments = self._calculate_moments(x, self.ws, quick=True)
            self._update_u(x)
            for i_loop in range(self.max_iter):
                if self.verbose:
                    print("annealing eps: {}, iter: {} / {}".format(eps, i_loop, self.max_iter), end='\r')
                ret = self.train_step(*x)
                obj = ret[0]
                reg_obj = ret[2]
                if i_loop % self.update_iter == 0 and self.verbose > 1:
                    self.moments = self._calculate_moments(x, self.ws, quick=True)
                    self._update_u(x)
                    print("tc = {}, obj = {}, reg = {}, eps = {}".format(np.sum(self.tc),
                                                                         obj, reg_obj, eps))
            print("Annealing iteration finished, time = {}".format(time.time() - start_time))

        self.moments = self._calculate_moments(x, self.ws, quick=False)  # Update moments with details
        self._update_u(x)
        return self

    def fit(self, x):
        self._define_model()
        # TODO: write a stopping condition
        x = [np.array(xt, dtype=np.float32) for xt in x]
        x = self.preprocess(x, fit=True)  # fit a transform for each marginal
        self.x_std = x  # to have an access to standardized x
        return self._train_loop(x)

    def _update_u(self, x):
        self.us = [self.getus(w.get_value(), xt) for w, xt in zip(self.ws, x)]

    @property
    def tc(self):
        """This actually returns the lower bound on TC that is optimized. The lower bound assumes a constraint that
         would be satisfied by a non-overlapping model.
         Check "moments" for two other estimates of TC that may be useful."""
        return [m["TC"] for m in self.moments]

    @property
    def tcs(self):
        """TCs for each individual latent factor. They DO NOT sum to TC in this case, because of overlaps."""
        return [m["TCs"] for m in self.moments]

    @property
    def mis(self):
        return [-0.5 * np.log1p(-m["rho"] ** 2) for m in self.moments]

    def clusters(self):
        return [np.argmax(np.abs(u), axis=0) for u in self.us]

    def _sig(self, x, u):
        """Multiple the matrix u by the covariance matrix of x. We are interested in situations where
        n_variables >> n_samples, so we do this without explicitly constructing the covariance matrix."""
        n_samples = x.shape[0]
        y = x.dot(u.T)
        tmp_dot = x.T.dot(y)
        prod = (1 - self.eps ** 2) * tmp_dot.T / n_samples + self.eps ** 2 * u  # nv by m,  <X_i Y_j> / std Y_j
        return prod

    def getus(self, w, x):
        # U_{ji} = \frac{W_{ji}}{\sqrt{E[Z_j^2]}}
        """
        U = np.zeros(W.shape)
        for j in range(W.shape[0]):
            U[j, :] = W[j, :] / np.sqrt(self.z2_fromW(j, W, x))
        return U
        """
        tmp_dot = np.dot(self._sig(x, w), w.T)
        z2 = self.y_scale ** 2 + np.einsum("ii->i", tmp_dot)
        return w / np.sqrt(z2).reshape((-1, 1))

    def _calculate_moments(self, x, ws, quick=False):
        us = [self.getus(w.get_value(), xt) for w, xt in zip(ws, x)]
        ret = [None] * self.nt
        for t in range(self.nt):
            ret[t] = self._calculate_moments_ns(x[t], us[t], quick=quick)
        return ret

    def _calculate_moments_ns(self, x, ws, quick=False):
        """Calculate moments based on the weights and samples. We also calculate and save MI, TC, additivity, and
        the value of the objective. Note it is assumed that <X_i^2> = 1! """
        m = {}  # Dictionary of moments
        n_samples = x.shape[0]
        y = x.dot(ws.T)
        tmp_sum = np.einsum('lj,lj->j', y, y)
        m["uj"] = (1 - self.eps ** 2) * tmp_sum / n_samples + self.eps ** 2 * np.sum(ws ** 2, axis=1)

        tmp_dot = x.T.dot(y)
        m["rho"] = (1 - self.eps ** 2) * tmp_dot.T / n_samples + self.eps ** 2 * ws  # m by nv
        m["ry"] = ws.dot(m["rho"].T)  # normalized covariance of Y
        m["Y_j^2"] = self.y_scale ** 2 / (1. - m["uj"])
        np.fill_diagonal(m["ry"], 1)
        m["invrho"] = 1. / (1. - m["rho"] ** 2)
        m["rhoinvrho"] = m["rho"] * m["invrho"]
        m["Qij"] = np.dot(m['ry'], m["rhoinvrho"])
        m["Qi"] = np.einsum('ki,ki->i', m["rhoinvrho"], m["Qij"])
        # m["Qi-Si^2"] = np.einsum('ki,ki->i', m["rhoinvrho"], m["Qij"])
        m["Si"] = np.sum(m["rho"] * m["rhoinvrho"], axis=0)

        # This is the objective, a lower bound for TC
        m["TC"] = np.sum(np.log(1 + m["Si"])) \
                  - 0.5 * np.sum(np.log(1 - m["Si"] ** 2 + m["Qi"])) \
                  + 0.5 * np.sum(np.log(1 - m["uj"]))

        if not quick:
            m["MI"] = - 0.5 * np.log1p(-m["rho"] ** 2)
            m["X_i Y_j"] = m["rho"].T * np.sqrt(m["Y_j^2"])
            m["X_i Z_j"] = np.linalg.solve(m["ry"], m["rho"]).T
            m["X_i^2 | Y"] = (1. - np.einsum('ij,ji->i', m["X_i Z_j"], m["rho"])).clip(1e-6)
            m['I(Y_j ; X)'] = 0.5 * np.log(m["Y_j^2"]) - 0.5 * np.log(self.y_scale ** 2)
            m['I(X_i ; Y)'] = - 0.5 * np.log(m["X_i^2 | Y"])
            m["TCs"] = m["MI"].sum(axis=1) - m['I(Y_j ; X)']
            m["TC_no_overlap"] = m["MI"].max(axis=0).sum() - m[
                'I(Y_j ; X)'].sum()  # A direct calculation of TC where each variable is in exactly one group.
            m["TC_direct"] = m['I(X_i ; Y)'].sum() - m[
                'I(Y_j ; X)']  # A direct calculation of TC. Should be upper bound for "TC", "TC_no_overlap"
            m["additivity"] = (m["MI"].sum(axis=0) - m['I(X_i ; Y)']).sum()
        return m

    def transform(self, x, details=False):
        """Transform an array of inputs, x, into an array of k latent factors, Y.
            Optionally, you can get the remainder information and/or stop at a specified level."""
        x = self.preprocess(x)
        ret = [a.dot(w.get_value().T) for (a, w) in zip(x, self.ws)]
        if details:
            moments = self._calculate_moments(x, self.us)
            return ret, moments
        return ret

    def preprocess(self, X, fit=False):
        """Transform each marginal to be as close to a standard Gaussian as possible.
        'standard' (default) just subtracts the mean and scales by the std.
        'empirical' does an empirical gaussianization (but this cannot be inverted).
        'outliers' tries to squeeze in the outliers
        Any other choice will skip the transformation."""
        warnings = []
        ret = [None] * len(X)
        if fit:
            self.theta = []
        for t in range(len(X)):
            x = X[t]
            if self.missing_values is not None:
                x, n_obs = mean_impute(x, self.missing_values)  # Creates a copy
            else:
                n_obs = len(x)
            if self.gaussianize == 'none':
                pass
            elif self.gaussianize == 'standard':
                if fit:
                    mean = np.mean(x, axis=0)
                    # std = np.std(x, axis=0, ddof=0).clip(1e-10)
                    std = np.sqrt(np.sum((x - mean) ** 2, axis=0) / n_obs).clip(1e-10)
                    self.theta.append((mean, std))
                x = ((x - self.theta[t][0]) / self.theta[t][1])
                if np.max(np.abs(x)) > 6 and self.verbose:
                    warnings.append("Warning: outliers more than 6 stds away from mean. "
                                    "Consider using gaussianize='outliers'")
            elif self.gaussianize == 'outliers':
                if fit:
                    mean = np.mean(x, axis=0)
                    std = np.std(x, axis=0, ddof=0).clip(1e-10)
                    self.theta.append((mean, std))
                x = g((x - self.theta[t][0]) / self.theta[t][1])  # g truncates long tails
            elif self.gaussianize == 'empirical':
                warnings.append("Warning: correct inversion/transform of empirical gauss transform not implemented.")
                x = np.array([norm.ppf((rankdata(x_i) - 0.5) / len(x_i)) for x_i in x.T]).T
            ret[t] = x
        for w in set(warnings):
            print(w)
        return ret

    def invert(self, x):
        # TODO: consider timesteps
        """Invert the preprocessing step to get x's in the original space."""
        if self.gaussianize == 'standard':
            return self.theta[1] * x + self.theta[0]
        elif self.gaussianize == 'outliers':
            return self.theta[1] * g_inv(x) + self.theta[0]
        else:
            return x

    def predict(self, y):
        # NOTE: not sure what does this function do
        ret = [None] * self.nt
        for t in range(self.nt):
            ret[t] = self.invert(np.dot(self.moments[t]["X_i Z_j"], y[t].T).T)
        return ret

    """
    def get_covariance(self):
        # This uses E(Xi|Y) formula for non-synergistic relationships
        m = self.moments
        ret = [None] * self.nt
        for t in range(self.nt):
            z = m[t]['rhoinvrho'] / (1 + m[t]['Si'])
            cov = np.dot(z.T, z)
            cov /= (1. - self.eps ** 2)
            np.fill_diagonal(cov, 1)
            ret[t] = self.theta[t][1][:, np.newaxis] * self.theta[t][1] * cov
        return ret
    """

    def get_covariance(self):
        norm_cov = self.get_norm_covariance(*list(np.array(self.x_std).astype(np.float32)))
        ret = [None] * self.nt
        for t in range(self.nt):
            ret[t] = self.theta[t][1][:, np.newaxis] * self.theta[t][1] * norm_cov[t]
        return ret


class TCorexSimple(TCorexBase):
    def __init__(self, l1=0.0, l2=0.0, reg_type='W', **kwargs):
        super(TCorexSimple, self).__init__(**kwargs)
        self.l1 = l1
        self.l2 = l2
        self.reg_type = reg_type

    def _define_model(self):
        self.x_wno = [None] * self.nt
        self.x = [None] * self.nt
        self.ws = [None] * self.nt
        self.z_mean = [None] * self.nt
        self.z = [None] * self.nt
        self.anneal_eps = theano.shared(np.float32(0))

        for t in range(self.nt):
            self.x_wno[t] = T.matrix('X')
            ns = self.x_wno[t].shape[0]
            anneal_noise = self.rng.normal(size=(ns, self.nv))
            self.x[t] = np.sqrt(1 - self.anneal_eps ** 2) * self.x_wno[t] + self.anneal_eps * anneal_noise
            z_noise = self.rng.normal(avg=0.0, std=self.y_scale, size=(ns, self.m))
            self.ws[t] = theano.shared(1.0 / np.sqrt(self.nv) * np.random.randn(self.m, self.nv), name='W{}'.format(t))
            self.z_mean[t] = T.dot(self.x[t], self.ws[t].T)
            self.z[t] = self.z_mean[t] + z_noise

        epsilon = 1e-8
        self.objs = [None] * self.nt
        self.sigma = [None] * self.nt
        mi_xz = [None] * self.nt

        for t in range(self.nt):
            z2 = (self.z[t] ** 2).mean(axis=0)  # (m,)
            ns = self.x_wno[t].shape[0]
            R = T.dot(self.z[t].T, self.x[t]) / ns  # m, nv
            R = R / T.sqrt(z2).reshape((self.m, 1))  # as <x^2_i> == 1 we don't divide by it
            ri = ((R ** 2) / T.clip(1 - R ** 2, epsilon, 1 - epsilon)).sum(axis=0)  # (nv,)
            if self.reg_type == 'MI':
                mi_xz[t] = -0.5 * T.log1p(-T.clip(R ** 2, 0, 1 - epsilon))

            # v_xi | z conditional mean
            outer_term = (1 / (1 + ri)).reshape((1, self.nv))
            inner_term_1 = R / T.clip(1 - R ** 2, epsilon, 1) / T.sqrt(z2).reshape((self.m, 1))  # (m, nv)
            inner_term_2 = self.z[t]  # (ns, m)
            cond_mean = outer_term * T.dot(inner_term_2, inner_term_1)  # (ns, nv)

            if not self.ignore_sigma:
                inner_mat = 1.0 / (1 + ri).reshape((1, self.nv)) * R / T.clip(1 - R ** 2, epsilon, 1)
                self.sigma[t] = T.dot(inner_mat.T, inner_mat)
                self.sigma[t] = self.sigma[t] * (1 - T.eye(self.nv)) + T.eye(self.nv)

            # objective
            obj_part_1 = 0.5 * T.log(T.clip(((self.x[t] - cond_mean) ** 2).mean(axis=0), epsilon, np.inf)).sum(axis=0)
            obj_part_2 = 0.5 * T.log(z2).sum(axis=0)
            self.objs[t] = obj_part_1 + obj_part_2

        # experiments show that main_obj scales approximately linearly with nv
        # also it is a sum of over time steps, so we divide on (nt * nv)
        self.main_obj = 1.0 / (self.nt * self.nv) * T.sum(self.objs)

        # regularization
        reg_matrices = [None] * self.nt
        if self.reg_type == 'W':
            reg_matrices = self.ws
        if self.reg_type == 'WWT':
            for t in range(self.nt):
                reg_matrices[t] = T.dot(self.ws[t].T, self.ws[t])
        if self.reg_type == 'Sigma':
            reg_matrices = self.sigma
        if self.reg_type == 'MI':
            reg_matrices = mi_xz

        self.reg_obj = T.constant(0)

        # experiments show that L1 and L2 regularizations scale approximately linearly with nv
        if self.l1 > 0:
            l1_reg = T.sum([T.abs_(reg_matrices[t + 1] - reg_matrices[t]).sum() for t in range(self.nt - 1)])
            l1_reg = 1.0 / (self.nt * self.nv) * l1_reg
            self.reg_obj = self.reg_obj + self.l1 * l1_reg

        if self.l2 > 0:
            l2_reg = T.sum([T.square(reg_matrices[t + 1] - reg_matrices[t]).sum() for t in range(self.nt - 1)])
            l2_reg = 1.0 / (self.nt * self.nv) * l2_reg
            self.reg_obj = self.reg_obj + self.l2 * l2_reg

        self.total_obj = self.main_obj + self.reg_obj

        # optimizer
        updates = lasagne.updates.adam(self.total_obj, self.ws)

        # functions
        self.train_step = theano.function(inputs=self.x_wno,
                                          outputs=[self.total_obj, self.main_obj, self.reg_obj] + self.objs,
                                          updates=updates)

        if not self.ignore_sigma:
            self.get_norm_covariance = theano.function(inputs=self.x_wno, outputs=self.sigma)


class TCorex(TCorexBase):
    def __init__(self, l1=0.0, l2=0.0, reg_type='W', init=True, gamma=2,
                 max_sample_cnt=2 ** 30, **kwargs):
        """
        :param gamma: parameter that controls the decay of weights.
                      weight of a sample whose distance from the current time-step is d will have 1.0/(gamma^d).
                      If gamma is equal to 1 all samples will be used with weight 1.
        :parm max_sample_cnt: maximum number of samples to use. Setting this to smaller values will give some speed up.
        """
        super(TCorex, self).__init__(**kwargs)
        self.l1 = l1
        self.l2 = l2
        self.reg_type = reg_type
        self.init = init
        self.gamma = np.float(gamma)
        self.max_sample_count = max_sample_cnt
        self.window_len = None  # this depends on x and will be computed in fit()

    def _define_model(self):
        self.x_wno = [None] * self.nt
        self.x = [None] * self.nt
        self.ws = [None] * self.nt
        self.z_mean = [None] * self.nt
        self.z = [None] * self.nt
        self.anneal_eps = theano.shared(np.float32(0))

        for t in range(self.nt):
            self.x_wno[t] = T.matrix('X')
            ns = self.x_wno[t].shape[0]
            anneal_noise = self.rng.normal(size=(ns, self.nv))
            self.x[t] = np.sqrt(1 - self.anneal_eps ** 2) * self.x_wno[t] + self.anneal_eps * anneal_noise
            z_noise = self.rng.normal(avg=0.0, std=self.y_scale, size=(ns, self.m))
            self.ws[t] = theano.shared(1.0 / np.sqrt(self.nv) * np.random.randn(self.m, self.nv), name='W{}'.format(t))
            self.z_mean[t] = T.dot(self.x[t], self.ws[t].T)
            self.z[t] = self.z_mean[t] + z_noise

        epsilon = 1e-8
        self.objs = [None] * self.nt
        self.sigma = [None] * self.nt
        mi_xz = [None] * self.nt

        for t in range(self.nt):
            weights = []
            l = max(0, t - self.window_len[t])
            r = min(self.nt, t + self.window_len[t] + 1)
            x_all = []
            for i in range(l, r):
                cur_ns = self.x_wno[i].shape[0]
                coef = np.power(self.gamma, np.abs(i - t))
                # skip if the importance is too low
                if coef < 1e-6:
                    continue
                weights.append(coef * T.ones((cur_ns,)))
                x_all.append(self.x[i])
            weights = T.concatenate(weights, axis=0)
            weights = weights / T.sum(weights)
            weights = weights.reshape((-1, 1))
            x_all = T.concatenate(x_all, axis=0)
            ns_tot = x_all.shape[0]

            z_all_mean = T.dot(x_all, self.ws[t].T)
            z_all_noise = self.rng.normal(avg=0.0, std=self.y_scale, size=(ns_tot, self.m))
            z_all = z_all_mean + z_all_noise
            z2_all = ((z_all ** 2) * weights).sum(axis=0)  # (m,)
            R_all = T.dot((z_all * weights).T, x_all)  # m, nv
            R_all = R_all / T.sqrt(z2_all).reshape((self.m, 1))  # as <x^2_i> == 1 we don't divide by it

            z2 = z2_all
            R = R_all

            if self.reg_type == 'MI':
                mi_xz[t] = -0.5 * T.log1p(-T.clip(R ** 2, 0, 1 - epsilon))

            # the rest depends on R and z2 only
            ri = ((R ** 2) / T.clip(1 - R ** 2, epsilon, 1 - epsilon)).sum(axis=0)  # (nv,)

            # v_xi | z conditional mean
            outer_term = (1 / (1 + ri)).reshape((1, self.nv))
            inner_term_1 = R / T.clip(1 - R ** 2, epsilon, 1) / T.sqrt(z2).reshape((self.m, 1))  # (m, nv)
            # NOTE: we use z[t], but seems only for objective not for the covariance estimate
            inner_term_2 = self.z[t]  # (ns, m)
            cond_mean = outer_term * T.dot(inner_term_2, inner_term_1)  # (ns, nv)

            # calculate normed covariance matrix
            if not self.ignore_sigma:
                inner_mat = 1.0 / (1 + ri).reshape((1, self.nv)) * R / T.clip(1 - R ** 2, epsilon, 1)
                self.sigma[t] = T.dot(inner_mat.T, inner_mat)
                self.sigma[t] = self.sigma[t] * (1 - T.eye(self.nv)) + T.eye(self.nv)

            # objective
            obj_part_1 = 0.5 * T.log(T.clip(((self.x[t] - cond_mean) ** 2).mean(axis=0), epsilon, np.inf)).sum(axis=0)
            obj_part_2 = 0.5 * T.log(z2).sum(axis=0)
            self.objs[t] = obj_part_1 + obj_part_2

        # experiments show that main_obj scales approximately linearly with nv
        # also it is a sum of over time steps, so we divide on (nt * nv)
        self.main_obj = 1.0 / (self.nt * self.nv) * T.sum(self.objs)

        # regularization
        reg_matrices = [None] * self.nt
        if self.reg_type == 'W':
            reg_matrices = self.ws
        if self.reg_type == 'WWT':
            for t in range(self.nt):
                reg_matrices[t] = T.dot(self.ws[t].T, self.ws[t])
        if self.reg_type == 'Sigma':
            reg_matrices = self.sigma
        if self.reg_type == 'MI':
            reg_matrices = mi_xz

        self.reg_obj = T.constant(0)

        # experiments show that L1 and L2 regularizations scale approximately linearly with nv
        if self.l1 > 0:
            l1_reg = T.sum([T.abs_(reg_matrices[t + 1] - reg_matrices[t]).sum() for t in range(self.nt - 1)])
            l1_reg = 1.0 / (self.nt * self.nv) * l1_reg
            self.reg_obj = self.reg_obj + self.l1 * l1_reg

        if self.l2 > 0:
            l2_reg = T.sum([T.square(reg_matrices[t + 1] - reg_matrices[t]).sum() for t in range(self.nt - 1)])
            l2_reg = 1.0 / (self.nt * self.nv) * l2_reg
            self.reg_obj = self.reg_obj + self.l2 * l2_reg

        self.total_obj = self.main_obj + self.reg_obj

        # optimizer
        updates = lasagne.updates.adam(self.total_obj, self.ws)

        # functions
        self.train_step = theano.function(inputs=self.x_wno,
                                          outputs=[self.total_obj, self.main_obj, self.reg_obj] + self.objs,
                                          updates=updates)

        if not self.ignore_sigma:
            self.get_norm_covariance = theano.function(inputs=self.x_wno, outputs=self.sigma)

    def fit(self, x):
        # Compute the window lengths for each time step and define the model
        n_samples = [len(xt) for xt in x]
        window_len = [0] * self.nt
        for i in range(self.nt):
            k = 0
            while k <= self.nt and sum(n_samples[max(0, i - k):min(self.nt, i + k + 1)]) < self.max_sample_count:
                k += 1
            window_len[i] = k
        self.window_len = window_len

        self._define_model()

        # Use the priors to estimate <X_i> and std(x_i) better
        self.theta = [None] * self.nt
        for t in range(self.nt):
            sum_weights = 0.0
            l = max(0, t - window_len[t])
            r = min(self.nt, t + window_len[t] + 1)
            for i in range(l, r):
                w = np.power(self.gamma, np.abs(i - t))
                sum_weights += x[i].shape[0] * w

            mean_prior = np.zeros((self.nv,))
            for i in range(l, r):
                w = np.power(self.gamma, np.abs(i - t))
                for sample in x[i]:
                    mean_prior += w * np.array(sample)
            mean_prior /= sum_weights

            var_prior = np.zeros((self.nv,))
            for i in range(l, r):
                w = np.power(self.gamma, np.abs(i - t))
                for sample in x[i]:
                    var_prior += w * ((np.array(sample) - mean_prior) ** 2)
            var_prior /= sum_weights
            std_prior = np.sqrt(var_prior)

            self.theta[t] = (mean_prior, std_prior)

        x = self.preprocess(x, fit=False)  # standardize the data using the better estimates
        x = [np.array(xt, dtype=np.float32) for xt in x]  # convert to np.float32
        self.x_std = x  # to have an access to standardized x

        # initialize weights using the weighs of the linear CorEx trained on all data
        if self.init:
            # NOTE: we simple linear CorExes in sliding window fashion and initialize using their weights
            #       but we prefer learning just one linear CorEx on whole data for simplicity and speed.
            if self.verbose:
                print("Initializing with weights of a linear CorEx learned on whole data")
            init_start = time.time()
            lin_corex = Corex(nv=self.nv,
                              n_hidden=self.m,
                              max_iter=self.max_iter,
                              anneal=self.anneal,
                              verbose=self.verbose,
                              gaussianize=self.gaussianize)
            # take maximum self.max_sample_cnt samples
            data_concat = np.concatenate(x, axis=0)
            if data_concat.shape[0] > self.max_sample_count:
                random.shuffle(data_concat)
                data_concat = data_concat[:self.max_sample_count]
            lin_corex.fit(data_concat)
            self.pretrained_weights = [lin_corex.ws.get_value()] * self.nt
            if self.verbose:
                print("Initialization took {:.2f} seconds".format(time.time() - init_start))

        return self._train_loop(x)
