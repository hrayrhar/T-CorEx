from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np
from scipy.stats import norm, rankdata
import time

import lasagne
import theano_linear_corex


import gc

try:
    import cudamat as cm

    GPU_SUPPORT = True
except:
    print("Install CUDA and cudamat (for python) to enable GPU speedups.")
    GPU_SUPPORT = False


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


class TimeCorex(object):
    def __init__(self, nt, nv, n_hidden=10, max_iter=10000, tol=1e-5, anneal=True, missing_values=None,
                 discourage_overlap=True, gaussianize='standard', gpu=False, y_scale=1.0, update_iter=10,
                 pretrained_weights=None, verbose=False, seed=None):

        self.nt = nt  # Number of timesteps
        self.nv = nv  # Number of variables
        self.m = n_hidden  # Number of latent factors to learn
        self.max_iter = max_iter  # Number of iterations to try
        self.tol = tol  # Threshold for convergence
        self.anneal = anneal
        self.eps = 0  # If anneal is True, it's adjusted during optimization to avoid local minima
        self.missing_values = missing_values

        self.discourage_overlap = discourage_overlap  # Whether or not to discourage overlapping latent factors
        self.gaussianize = gaussianize  # Preprocess data: 'standard' scales to zero mean and unit variance
        self.gpu = gpu  # Enable GPU support for some large matrix multiplications.
        if self.gpu:
            cm.cublas_init()

        self.y_scale = y_scale  # Can be arbitrary, but sets the scale of Y
        self.update_iter = update_iter  # Compute statistics every update_iter
        self.pretrained_weights = pretrained_weights
        np.random.seed(seed)  # Set seed for deterministic results
        self.verbose = verbose
        if verbose:
            np.set_printoptions(precision=3, suppress=True, linewidth=160)
            print('Linear CorEx with {:d} latent factors'.format(n_hidden))

        self.history = [{} for t in range(self.nt)]  # Keep track of values for each iteration
        self.rng = RandomStreams(seed)

    def fit(self, x):
        # TODO: write a stopping condition
        x = [np.array(xt, dtype=np.float32) for xt in x]
        x = self.preprocess(x, fit=True)  # Fit a transform for each marginal
        self.x_std = x  # to have an access to standardized x

        anneal_schedule = [0.]
        if self.anneal:
            anneal_schedule = [0.6 ** k for k in range(1, 7)] + [0]

        if self.pretrained_weights is not None:
            for cur_w, pre_w in zip(self.ws, self.pretrained_weights):
                cur_w.set_value(pre_w)

        for i_eps, eps in enumerate(anneal_schedule):
            start_time = time.time()

            self.eps = eps
            self.anneal_eps.set_value(np.float32(eps))
            self.moments = self._calculate_moments(x, self.ws, quick=True)
            self._update_u(x)

            for i_loop in range(self.max_iter):
                ret = self.train_step(*x)
                obj = ret[0]
                reg_obj = ret[2]

                if i_loop % self.update_iter == 0 and self.verbose:
                    self.moments = self._calculate_moments(x, self.ws, quick=True)
                    self._update_u(x)
                    print("tc = {}, obj = {}, reg = {}, eps = {}".format(np.sum(self.tc),
                                                                         obj, reg_obj, eps))

            print("Annealing iteration finished, time = {}".format(time.time() - start_time))

        self.moments = self._calculate_moments(x, self.ws, quick=False)  # Update moments with details
        return self

    def _update_u(self, x):
        self.us = [self.getus(w.get_value(), xt) for w, xt in zip(self.ws, x)]

    def update_records(self, moments, delta):
        """Print and store some statistics about each iteration."""
        gc.disable()  # There's a bug that slows when appending, fixed by temporarily disabling garbage collection
        for t in range(self.nt):
            self.history[t]["TC"] = self.history[t].get("TC", []) + [moments[t]["TC"]]
        if self.verbose > 1:
            tc_sum = sum([m["TC"] for m in moments])
            add_sum = sum([m.get("additivity", 0) for m in moments])
            print("TC={:.3f}\tadd={:.3f}\tdelta={:.6f}".format(tc_sum, add_sum, delta))
        if self.verbose:
            for t in range(self.nt):
                self.history[t]["additivity"] = self.history[t].get("additivity", []) + [
                    moments[t].get("additivity", 0)]
                self.history[t]["TCs"] = self.history[t].get("TCs", []) + [moments[t].get("TCs", np.zeros(self.m))]
        gc.enable()

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
        return np.argmax(np.abs(self.us), axis=0)

    def _sig(self, x, u):
        """Multiple the matrix u by the covariance matrix of x. We are interested in situations where
        n_variables >> n_samples, so we do this without explicitly constructing the covariance matrix."""
        n_samples = x.shape[0]
        if self.gpu:
            y = cm.empty((n_samples, self.m))
            uc = cm.CUDAMatrix(u)
            cm.dot(x, uc.T, target=y)
            del uc
            tmp = cm.empty((self.nv, self.m))
            cm.dot(x.T, y, target=tmp)
            tmp_dot = tmp.asarray()
            del y
            del tmp
        else:
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
            if self.discourage_overlap:
                ret[t] = self._calculate_moments_ns(x[t], us[t], quick=quick)
            else:
                ret[t] = self._calculate_moments_syn(x[t], us[t], quick=quick)
        return ret

    def _calculate_moments_ns(self, x, ws, quick=False):
        """Calculate moments based on the weights and samples. We also calculate and save MI, TC, additivity, and
        the value of the objective. Note it is assumed that <X_i^2> = 1! """
        m = {}  # Dictionary of moments
        n_samples = x.shape[0]
        if self.gpu:
            y = cm.empty((n_samples, self.m))
            wc = cm.CUDAMatrix(ws)
            cm.dot(x, wc.T, target=y)  # + noise, but it is included analytically
            del wc
            tmp_sum = np.einsum('lj,lj->j', y.asarray(), y.asarray())  # TODO: Should be able to do on gpu...
        else:
            y = x.dot(ws.T)
            tmp_sum = np.einsum('lj,lj->j', y, y)
        m["uj"] = (1 - self.eps ** 2) * tmp_sum / n_samples + self.eps ** 2 * np.sum(ws ** 2, axis=1)

        if self.gpu:
            tmp = cm.empty((self.nv, self.m))
            cm.dot(x.T, y, target=tmp)
            tmp_dot = tmp.asarray()
            del tmp
            del y
        else:
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

    def _calculate_moments_syn(self, x, ws, quick=False):
        return None

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
                    print("Warning: outliers more than 6 stds away from mean. Consider using gaussianize='outliers'")
            elif self.gaussianize == 'outliers':
                if fit:
                    mean = np.mean(x, axis=0)
                    std = np.std(x, axis=0, ddof=0).clip(1e-10)
                    self.theta.append((mean, std))
                x = g((x - self.theta[t][0]) / self.theta[t][1])  # g truncates long tails
            elif self.gaussianize == 'empirical':
                print("Warning: correct inversion/transform of empirical gauss transform not implemented.")
                x = np.array([norm.ppf((rankdata(x_i) - 0.5) / len(x_i)) for x_i in x.T]).T
            if self.gpu and fit:  # Don't return GPU matrices when only transforming
                x = cm.CUDAMatrix(x)
            ret[t] = x
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

    def get_covariance(self):
        # This uses E(Xi|Y) formula for non-synergistic relationships
        m = self.moments
        ret = [None] * self.nt
        for t in range(self.nt):
            if self.discourage_overlap:
                z = m[t]['rhoinvrho'] / (1 + m[t]['Si'])
                cov = np.dot(z.T, z)
                cov /= (1. - self.eps ** 2)
                np.fill_diagonal(cov, 1)
                ret[t] = self.theta[t][1][:, np.newaxis] * self.theta[t][1] * cov
            else:
                cov = np.einsum('ij,kj->ik', m[t]["X_i Z_j"], mp[t]["X_i Y_j"])
                np.fill_diagonal(cov, 1)
                ret[t] = self.theta[t][1][:, np.newaxis] * self.theta[t][1] * cov
        return ret


class TimeCorexW(TimeCorex):
    def __init__(self, l1=0.0, l2=0.0, **kwargs):
        super(TimeCorexW, self).__init__(**kwargs)
        self.l1 = l1
        self.l2 = l2
        self._define_model()

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

        EPS = 1e-5
        self.objs = [None] * self.nt
        self.sigma = [None] * self.nt

        for t in range(self.nt):
            z2 = (self.z[t] ** 2).mean(axis=0)  # (m,)
            ns = self.x_wno[t].shape[0]
            R = T.dot(self.z[t].T, self.x[t]) / ns  # m, nv
            R = R / T.sqrt(z2).reshape((self.m, 1))  # as <x^2_i> == 1 we don't divide by it
            ri = ((R ** 2) / T.clip(1 - R ** 2, EPS, 1 - EPS)).sum(axis=0)  # (nv,)

            # v_xi | z conditional mean
            outer_term = (1 / (1 + ri)).reshape((1, self.nv))
            inner_term_1 = (R / T.clip(1 - R ** 2, EPS, 1) / T.sqrt(z2).reshape((self.m, 1))).reshape(
                (1, self.m, self.nv))
            inner_term_2 = self.z[t].reshape((ns, self.m, 1))
            cond_mean = outer_term * ((inner_term_1 * inner_term_2).sum(axis=1))  # (ns, nv)

            inner_mat = 1.0 / (1 + ri).reshape((1, self.nv)) * R / T.clip(1 - R ** 2, EPS, 1)
            self.sigma[t] = T.dot(inner_mat.T, inner_mat)
            self.sigma[t] = self.sigma[t] * (1 - T.eye(self.nv)) + T.eye(self.nv)

            # objective
            obj_part_1 = 0.5 * T.log(T.clip(((self.x[t] - cond_mean) ** 2).mean(axis=0), EPS, np.inf)).sum(axis=0)
            obj_part_2 = 0.5 * T.log(z2).sum(axis=0)
            self.objs[t] = obj_part_1 + obj_part_2

        self.main_obj = T.sum(self.objs)

        # regularization
        self.reg_obj = T.constant(0)

        if self.l1 > 0:
            l1_reg = T.sum([T.abs_(self.ws[t + 1] - self.ws[t]).sum() for t in range(self.nt - 1)])
            self.reg_obj = self.reg_obj + self.l1 * l1_reg

        if self.l2 > 0:
            l2_reg = T.sum([T.square(self.ws[t + 1] - self.ws[t]).sum() for t in range(self.nt - 1)])
            self.reg_obj = self.reg_obj + self.l2 * l2_reg

        self.total_obj = self.main_obj + self.reg_obj

        # optimizer
        updates = lasagne.updates.adam(self.total_obj, self.ws)

        # functions
        self.train_step = theano.function(inputs=self.x_wno,
                                          outputs=[self.total_obj, self.main_obj, self.reg_obj] + self.objs,
                                          updates=updates)

        self.get_norm_covariance = theano.function(inputs=self.x_wno,
                                                   outputs=self.sigma)


class TimeCorexWWT(TimeCorex):
    def __init__(self, l1=0.0, l2=0.0, **kwargs):
        super(TimeCorexWWT, self).__init__(**kwargs)
        self.l1 = l1
        self.l2 = l2
        self._define_model()

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

        EPS = 1e-5
        self.objs = [None] * self.nt
        self.sigma = [None] * self.nt

        for t in range(self.nt):
            z2 = (self.z[t] ** 2).mean(axis=0)  # (m,)
            ns = self.x_wno[t].shape[0]
            R = T.dot(self.z[t].T, self.x[t]) / ns  # m, nv
            R = R / T.sqrt(z2).reshape((self.m, 1))  # as <x^2_i> == 1 we don't divide by it
            ri = ((R ** 2) / T.clip(1 - R ** 2, EPS, 1 - EPS)).sum(axis=0)  # (nv,)

            # v_xi | z conditional mean
            outer_term = (1 / (1 + ri)).reshape((1, self.nv))
            inner_term_1 = (R / T.clip(1 - R ** 2, EPS, 1) / T.sqrt(z2).reshape((self.m, 1))).reshape(
                (1, self.m, self.nv))
            inner_term_2 = self.z[t].reshape((ns, self.m, 1))
            cond_mean = outer_term * ((inner_term_1 * inner_term_2).sum(axis=1))  # (ns, nv)

            inner_mat = 1.0 / (1 + ri).reshape((1, self.nv)) * R / T.clip(1 - R ** 2, EPS, 1)
            self.sigma[t] = T.dot(inner_mat.T, inner_mat)
            self.sigma[t] = self.sigma[t] * (1 - T.eye(self.nv)) + T.eye(self.nv)

            # objective
            obj_part_1 = 0.5 * T.log(T.clip(((self.x[t] - cond_mean) ** 2).mean(axis=0), EPS, np.inf)).sum(axis=0)
            obj_part_2 = 0.5 * T.log(z2).sum(axis=0)
            self.objs[t] = obj_part_1 + obj_part_2

        self.main_obj = T.sum(self.objs)

        # regularization
        self.reg_obj = T.constant(0)

        self.S = [None] * self.nt
        for t in range(self.nt):
            self.S[t] = T.dot(self.ws[t].T, self.ws[t])

        if self.l1 > 0:
            l1_reg = T.sum([T.abs_(self.S[t + 1] - self.S[t]).sum() for t in range(self.nt - 1)])
            self.reg_obj = self.reg_obj + self.l1 * l1_reg

        if self.l2 > 0:
            l2_reg = T.sum([T.square(self.S[t + 1] - self.S[t]).sum() for t in range(self.nt - 1)])
            self.reg_obj = self.reg_obj + self.l2 * l2_reg

        self.total_obj = self.main_obj + self.reg_obj

        # optimizer
        updates = lasagne.updates.adam(self.total_obj, self.ws)

        # functions
        self.train_step = theano.function(inputs=self.x_wno,
                                          outputs=[self.total_obj, self.main_obj, self.reg_obj] + self.objs,
                                          updates=updates)

        self.get_norm_covariance = theano.function(inputs=self.x_wno,
                                                   outputs=self.sigma)


class TimeCorexGlobalMI(TimeCorex):
    def __init__(self, l1=0.0, l2=0.0, **kwargs):
        super(TimeCorexGlobalMI, self).__init__(**kwargs)
        self.l1 = l1
        self.l2 = l2
        self._define_model()

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

        EPS = 1e-5
        self.objs = [None] * self.nt
        self.mizx = [None] * self.nt

        for t in range(self.nt):
            z2 = (self.z[t] ** 2).mean(axis=0)  # (m,)
            ns = self.x_wno[t].shape[0]
            R = T.dot(self.z[t].T, self.x[t]) / ns  # m, nv
            R = R / T.sqrt(z2).reshape((self.m, 1))  # as <x^2_i> == 1 we don't divide by it
            ri = ((R ** 2) / T.clip(1 - R ** 2, EPS, 1 - EPS)).sum(axis=0)  # (nv,)

            self.mizx[t] = -0.5 * T.log1p(-T.clip(R ** 2, 0, 1 - EPS))

            # v_xi | z conditional mean
            outer_term = (1 / (1 + ri)).reshape((1, self.nv))
            inner_term_1 = (R / T.clip(1 - R ** 2, EPS, 1) / T.sqrt(z2).reshape((self.m, 1))).reshape(
                (1, self.m, self.nv))
            inner_term_2 = self.z[t].reshape((ns, self.m, 1))
            cond_mean = outer_term * ((inner_term_1 * inner_term_2).sum(axis=1))  # (ns, nv)

            # objective
            obj_part_1 = 0.5 * T.log(T.clip(((self.x[t] - cond_mean) ** 2).mean(axis=0), EPS, np.inf)).sum(axis=0)
            obj_part_2 = 0.5 * T.log(z2).sum(axis=0)
            self.objs[t] = obj_part_1 + obj_part_2

        self.main_obj = T.sum(self.objs)

        # regularization
        self.reg_obj = T.constant(0)

        if self.l1 > 0:
            l1_reg = T.sum([T.abs_(self.mizx[t + 1] - self.mizx[t]).sum() for t in range(self.nt - 1)])
            self.reg_obj = self.reg_obj + self.l1 * l1_reg

        if self.l2 > 0:
            l2_reg = T.sum([T.square(self.mizx[t + 1] - self.mizx[t]).sum() for t in range(self.nt - 1)])
            self.reg_obj = self.reg_obj + self.l2 * l2_reg

        self.total_obj = self.main_obj + self.reg_obj

        # optimizer
        updates = lasagne.updates.adam(self.total_obj, self.ws)

        # functions
        self.train_step = theano.function(inputs=self.x_wno,
                                          outputs=[self.total_obj, self.main_obj, self.reg_obj] + self.objs,
                                          updates=updates)


class TimeCorexSigma(TimeCorex):
    def __init__(self, l1=0.0, l2=0.0, **kwargs):
        super(TimeCorexSigma, self).__init__(**kwargs)
        self.l1 = l1
        self.l2 = l2
        self._define_model()

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

        EPS = 1e-5
        self.objs = [None] * self.nt
        self.sigma = [None] * self.nt
        self.R = [None] * self.nt

        for t in range(self.nt):
            z2 = (self.z[t] ** 2).mean(axis=0)  # (m,)
            ns = self.x_wno[t].shape[0]
            R = T.dot(self.z[t].T, self.x[t]) / ns  # m, nv
            R = R / T.sqrt(z2).reshape((self.m, 1))  # as <x^2_i> == 1 we don't divide by it
            ri = ((R ** 2) / T.clip(1 - R ** 2, EPS, 1 - EPS)).sum(axis=0)  # (nv,)

            # v_xi | z conditional mean
            outer_term = (1 / (1 + ri)).reshape((1, self.nv))
            inner_term_1 = (R / T.clip(1 - R ** 2, EPS, 1) / T.sqrt(z2).reshape((self.m, 1))).reshape(
                (1, self.m, self.nv))
            inner_term_2 = self.z[t].reshape((ns, self.m, 1))
            cond_mean = outer_term * ((inner_term_1 * inner_term_2).sum(axis=1))  # (ns, nv)

            inner_mat = 1.0 / (1 + ri).reshape((1, self.nv)) * R / T.clip(1 - R ** 2, EPS, 1)
            self.sigma[t] = T.dot(inner_mat.T, inner_mat)
            self.sigma[t] = self.sigma[t] * (1 - T.eye(self.nv)) + T.eye(self.nv)

            self.R[t] = R

            # objective
            obj_part_1 = 0.5 * T.log(T.clip(((self.x[t] - cond_mean) ** 2).mean(axis=0), EPS, np.inf)).sum(axis=0)
            obj_part_2 = 0.5 * T.log(z2).sum(axis=0)
            self.objs[t] = obj_part_1 + obj_part_2

        self.main_obj = T.sum(self.objs)

        # regularization
        self.reg_obj = T.constant(0)

        if self.l1 > 0:
            l1_reg = T.sum([T.abs_(self.sigma[t + 1] - self.sigma[t]).sum() for t in range(self.nt - 1)])
            self.reg_obj = self.reg_obj + self.l1 * l1_reg

        if self.l2 > 0:
            l2_reg = T.sum([T.square(self.sigma[t + 1] - self.sigma[t]).sum() for t in range(self.nt - 1)])
            self.reg_obj = self.reg_obj + self.l2 * l2_reg

        self.total_obj = self.main_obj + self.reg_obj

        # optimizer
        updates = lasagne.updates.adam(self.total_obj, self.ws)

        # functions
        self.train_step = theano.function(inputs=self.x_wno,
                                          outputs=[self.total_obj, self.main_obj, self.reg_obj] + self.objs,
                                          updates=updates)

        self.get_norm_covariance = theano.function(inputs=self.x_wno,
                                                   outputs=self.sigma)

        self.get_R = theano.function(inputs=self.x_wno,
                                     outputs=self.R)


class TimeCorexWPrior(TimeCorex):
    def __init__(self, l1=0.0, l2=0.0, lamb=0.5, **kwargs):
        super(TimeCorexWPrior, self).__init__(**kwargs)
        self.l1 = l1
        self.l2 = l2
        self.lamb = lamb
        self._define_model()

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

        EPS = 1e-5
        self.objs = [None] * self.nt
        self.sigma = [None] * self.nt

        self.z2_prior = [None] * self.nt
        self.R_prior = [None] * self.nt

        for t in range(self.nt):
            self.z2_prior[t] = theano.shared(np.zeros((self.m,), dtype=np.float32), name='z2p_{}'.format(t))
            self.R_prior[t] = theano.shared(np.zeros((self.m, self.nv), dtype=np.float32), name='Rp_{}'.format(t))

            z2 = (self.z[t] ** 2).mean(axis=0)  # (m,)
            ns = self.x_wno[t].shape[0]
            R = T.dot(self.z[t].T, self.x[t]) / ns  # m, nv
            R = R / T.sqrt(z2).reshape((self.m, 1))  # as <x^2_i> == 1 we don't divide by it

            # add priors
            z2 = (1 - self.lamb) * z2 + self.lamb * self.z2_prior[t]
            R = (1 - self.lamb) * R + self.lamb * self.R_prior[t]

            # the rest depends on R and z2 only
            ri = ((R ** 2) / T.clip(1 - R ** 2, EPS, 1 - EPS)).sum(axis=0)  # (nv,)

            # v_xi | z conditional mean
            outer_term = (1 / (1 + ri)).reshape((1, self.nv))
            inner_term_1 = (R / T.clip(1 - R ** 2, EPS, 1) / T.sqrt(z2).reshape((self.m, 1))).reshape(
                (1, self.m, self.nv))
            # NOTE: we use z[t], but seems only for objective not for the covariance estimate
            inner_term_2 = self.z[t].reshape((ns, self.m, 1))
            cond_mean = outer_term * ((inner_term_1 * inner_term_2).sum(axis=1))  # (ns, nv)

            inner_mat = 1.0 / (1 + ri).reshape((1, self.nv)) * R / T.clip(1 - R ** 2, EPS, 1)
            self.sigma[t] = T.dot(inner_mat.T, inner_mat)
            self.sigma[t] = self.sigma[t] * (1 - T.eye(self.nv)) + T.eye(self.nv)

            # objective
            obj_part_1 = 0.5 * T.log(T.clip(((self.x[t] - cond_mean) ** 2).mean(axis=0), EPS, np.inf)).sum(axis=0)
            obj_part_2 = 0.5 * T.log(z2).sum(axis=0)
            self.objs[t] = obj_part_1 + obj_part_2

        self.main_obj = T.sum(self.objs)

        # regularization
        self.reg_obj = T.constant(0)

        if self.l1 > 0:
            l1_reg = T.sum([T.abs_(self.ws[t + 1] - self.ws[t]).sum() for t in range(self.nt - 1)])
            self.reg_obj = self.reg_obj + self.l1 * l1_reg

        if self.l2 > 0:
            l2_reg = T.sum([T.square(self.ws[t + 1] - self.ws[t]).sum() for t in range(self.nt - 1)])
            self.reg_obj = self.reg_obj + self.l2 * l2_reg

        self.total_obj = self.main_obj + self.reg_obj

        # optimizer
        updates = lasagne.updates.adam(self.total_obj, self.ws)

        # functions
        self.train_step = theano.function(inputs=self.x_wno,
                                          outputs=[self.total_obj, self.main_obj, self.reg_obj] + self.objs,
                                          updates=updates)

        self.get_norm_covariance = theano.function(inputs=self.x_wno,
                                                   outputs=self.sigma)

    def fit(self, x):
        # TODO: write a stopping condition

        # fit a linear corex to get the priors
        lin_corex = theano_linear_corex.Corex(nv=self.nv,
                                              n_hidden=self.m,
                                              max_iter=500,
                                              anneal=True,
                                              verbose=self.verbose)
        lin_corex.fit(np.concatenate(x, axis=0))

        x = [np.array(xt, dtype=np.float32) for xt in x]
        dummy = self.preprocess(x, fit=True)  # Fit a transform for each marginal

        # use the priors to estimate <X_i> and std(x_i) better
        for t in range(self.nt):
            cur_mean = (1 - self.lamb) * self.theta[t][0] + self.lamb * lin_corex.theta[0]
            cur_std = (1 - self.lamb) * self.theta[t][1] + self.lamb * lin_corex.theta[1]
            self.theta[t] = (cur_mean, cur_std)

        x = self.preprocess(x, fit=False)  # standardize the data using the better estimates
        self.x_std = x  # to have an access to standardized x

        # initialize W matrices with lin_corex matrix
        self.pretrained_weights = [lin_corex.ws.get_value()] * self.nt

        # initialize priors
        for t in range(self.nt):
            self.z2_prior[t].set_value(lin_corex.moments['Y_j^2'].astype(np.float32))
            self.R_prior[t].set_value(lin_corex.moments['rho'].astype(np.float32))

        anneal_schedule = [0.]
        if self.anneal:
            anneal_schedule = [0.6 ** k for k in range(1, 7)] + [0]

        if self.pretrained_weights is not None:
            for cur_w, pre_w in zip(self.ws, self.pretrained_weights):
                cur_w.set_value(pre_w.astype(np.float32))

        for i_eps, eps in enumerate(anneal_schedule):
            start_time = time.time()

            self.eps = eps
            self.anneal_eps.set_value(np.float32(eps))
            self.moments = self._calculate_moments(x, self.ws, quick=True)
            self._update_u(x)

            for i_loop in range(self.max_iter):
                ret = self.train_step(*x)
                obj = ret[0]
                reg_obj = ret[2]

                if i_loop % self.update_iter == 0 and self.verbose:
                    self.moments = self._calculate_moments(x, self.ws, quick=True)
                    self._update_u(x)
                    print("tc = {}, obj = {}, reg = {}, eps = {}".format(np.sum(self.tc),
                                                                         obj, reg_obj, eps))

            print("Annealing iteration finished, time = {}".format(time.time() - start_time))

        self.moments = self._calculate_moments(x, self.ws, quick=False)  # Update moments with details
        return self

    def get_covariance(self):
        norm_cov = self.get_norm_covariance(*list(np.array(self.x_std).astype(np.float32)))
        ret = [None] * self.nt
        for t in range(self.nt):
            ret[t] = self.theta[t][1][:, np.newaxis] * self.theta[t][1] * norm_cov[t]
        return ret


class TimeCorexWPriorLearnable(TimeCorex):
    def __init__(self, l1=0.0, l2=0.0, **kwargs):
        super(TimeCorexWPriorLearnable, self).__init__(**kwargs)
        self.l1 = l1
        self.l2 = l2
        self._define_model()

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

        EPS = 1e-5
        self.objs = [None] * self.nt
        self.sigma = [None] * self.nt

        self.z2_prior = [None] * self.nt
        self.R_prior = [None] * self.nt

        self.lamb_logit = theano.shared(np.float32(4.0), name='lamb_logit')
        self.lamb = T.nnet.sigmoid(self.lamb_logit)

        for t in range(self.nt):
            self.z2_prior[t] = theano.shared(np.zeros((self.m,), dtype=np.float32), name='z2p_{}'.format(t))
            self.R_prior[t] = theano.shared(np.zeros((self.m, self.nv), dtype=np.float32), name='Rp_{}'.format(t))

            z2 = (self.z[t] ** 2).mean(axis=0)  # (m,)
            ns = self.x_wno[t].shape[0]
            R = T.dot(self.z[t].T, self.x[t]) / ns  # m, nv
            R = R / T.sqrt(z2).reshape((self.m, 1))  # as <x^2_i> == 1 we don't divide by it

            # add priors
            z2 = (1 - self.lamb) * z2 + self.lamb * self.z2_prior[t]
            R = (1 - self.lamb) * R + self.lamb * self.R_prior[t]

            # the rest depends on R and z2 only
            ri = ((R ** 2) / T.clip(1 - R ** 2, EPS, 1 - EPS)).sum(axis=0)  # (nv,)

            # v_xi | z conditional mean
            outer_term = (1 / (1 + ri)).reshape((1, self.nv))
            inner_term_1 = (R / T.clip(1 - R ** 2, EPS, 1) / T.sqrt(z2).reshape((self.m, 1))).reshape(
                (1, self.m, self.nv))
            # NOTE: we use z[t], but seems only for objective not for the covariance estimate
            inner_term_2 = self.z[t].reshape((ns, self.m, 1))
            cond_mean = outer_term * ((inner_term_1 * inner_term_2).sum(axis=1))  # (ns, nv)

            inner_mat = 1.0 / (1 + ri).reshape((1, self.nv)) * R / T.clip(1 - R ** 2, EPS, 1)
            self.sigma[t] = T.dot(inner_mat.T, inner_mat)
            self.sigma[t] = self.sigma[t] * (1 - T.eye(self.nv)) + T.eye(self.nv)

            # objective
            obj_part_1 = 0.5 * T.log(T.clip(((self.x[t] - cond_mean) ** 2).mean(axis=0), EPS, np.inf)).sum(axis=0)
            obj_part_2 = 0.5 * T.log(z2).sum(axis=0)
            self.objs[t] = obj_part_1 + obj_part_2

        self.main_obj = T.sum(self.objs)

        # regularization
        self.reg_obj = T.constant(0)

        if self.l1 > 0:
            l1_reg = T.sum([T.abs_(self.ws[t + 1] - self.ws[t]).sum() for t in range(self.nt - 1)])
            self.reg_obj = self.reg_obj + self.l1 * l1_reg

        if self.l2 > 0:
            l2_reg = T.sum([T.square(self.ws[t + 1] - self.ws[t]).sum() for t in range(self.nt - 1)])
            self.reg_obj = self.reg_obj + self.l2 * l2_reg

        self.total_obj = self.main_obj + self.reg_obj

        # optimizer
        # updates = lasagne.updates.adam(self.total_obj, self.ws + [self.lamb_logit])
        updates = lasagne.updates.adam(self.total_obj, self.ws)

        # functions
        self.train_step = theano.function(inputs=self.x_wno,
                                          outputs=[self.total_obj, self.main_obj, self.reg_obj] + self.objs,
                                          updates=updates)

        self.get_obj= theano.function(inputs=self.x_wno,
                                      outputs=[self.total_obj, self.main_obj, self.reg_obj] + self.objs)

        self.get_norm_covariance = theano.function(inputs=self.x_wno,
                                                   outputs=self.sigma)

    def fit(self, x):
        # TODO: write a stopping condition

        # fit a linear corex to get the priors
        lin_corex = theano_linear_corex.Corex(nv=self.nv,
                                              n_hidden=self.m,
                                              max_iter=500,
                                              anneal=True,
                                              verbose=self.verbose)
        lin_corex.fit(np.concatenate(x, axis=0))

        x = [np.array(xt, dtype=np.float32) for xt in x]
        x = self.preprocess(x, fit=True)

        self.x_std = x  # to have an access to standardized x

        # initialize W matrices with lin_corex matrix
        self.pretrained_weights = [lin_corex.ws.get_value()] * self.nt

        # initialize priors
        for t in range(self.nt):
            self.z2_prior[t].set_value(lin_corex.moments['Y_j^2'].astype(np.float32))
            self.R_prior[t].set_value(lin_corex.moments['rho'].astype(np.float32))

        anneal_schedule = [0.]
        if self.anneal:
            anneal_schedule = [0.6 ** k for k in range(1, 7)] + [0]

        if self.pretrained_weights is not None:
            for cur_w, pre_w in zip(self.ws, self.pretrained_weights):
                cur_w.set_value(pre_w.astype(np.float32))

        # self.lamb_logit.set_value(np.float32(-10))
        # objs = self.get_obj(*x)
        # print("Pretrained lamb = 0, main_obj  = {}, reg_obj = {}".format(objs[1], objs[2]))
        #
        # self.lamb_logit.set_value(np.float32(10))
        # objs = self.get_obj(*x)
        # print("Pretrained lamb = 1, main_obj  = {}, reg_obj = {}".format(objs[1], objs[2]))
        # return

        for i_eps, eps in enumerate(anneal_schedule):
            start_time = time.time()

            self.eps = eps
            self.anneal_eps.set_value(np.float32(eps))
            self.moments = self._calculate_moments(x, self.ws, quick=True)
            self._update_u(x)

            for i_loop in range(self.max_iter):
                ret = self.train_step(*x)
                obj = ret[0]
                reg_obj = ret[2]

                if i_loop % 50 == 0:
                    lambdas = np.linspace(1e-2, 1.0 - 1e-2, 20)
                    best_obj = 1e18
                    best_logit = 0
                    best_lamb = 0
                    for lamb in lambdas:
                        lamb_logit = np.log(lamb / (1.0 - lamb))
                        self.lamb_logit.set_value(np.float32(lamb_logit))
                        cur_obj = self.get_obj(*x)[0]  # take total_obj
                        # print("lamb = {}, obj = {}".format(lamb, cur_obj))
                        if cur_obj < best_obj:
                            best_obj = cur_obj
                            best_logit = lamb_logit
                            best_lamb = lamb
                    self.lamb_logit.set_value(np.float32(best_logit))
                    print('Tuned lambda = {}, logit = {}'.format(best_lamb, best_logit))

                if i_loop % self.update_iter == 0 and self.verbose:
                    self.moments = self._calculate_moments(x, self.ws, quick=True)
                    self._update_u(x)
                    lamb = 1.0 / (1.0 + np.exp(-self.lamb_logit.get_value()))
                    print("tc = {}, obj = {}, reg = {}, eps = {}, lamb = {}".format(
                        np.sum(self.tc), obj, reg_obj, eps, lamb))

            print("Annealing iteration finished, time = {}".format(time.time() - start_time))

        self.moments = self._calculate_moments(x, self.ws, quick=False)  # Update moments with details

        # use the priors to estimate <X_i> and std(x_i) better
        lamb = 1.0 / (1.0 + np.exp(-self.lamb_logit.get_value()))
        print("Learned lamb = {}".format(lamb))
        for t in range(self.nt):

            cur_mean = (1 - lamb) * self.theta[t][0] + lamb * lin_corex.theta[0]
            cur_std = (1 - lamb) * self.theta[t][1] + lamb * lin_corex.theta[1]
            self.theta[t] = (cur_mean, cur_std)

        return self

    def get_covariance(self):
        norm_cov = self.get_norm_covariance(*list(np.array(self.x_std).astype(np.float32)))
        ret = [None] * self.nt
        for t in range(self.nt):
            ret[t] = self.theta[t][1][:, np.newaxis] * self.theta[t][1] * norm_cov[t]
        return ret


class TimeCorexWPrior2(TimeCorex):
    def __init__(self, l1=0.0, l2=0.0, lamb=0.5, **kwargs):
        super(TimeCorexWPrior2, self).__init__(**kwargs)
        self.l1 = l1
        self.l2 = l2
        self.lamb = lamb
        self._define_model()

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

        EPS = 1e-5
        self.objs = [None] * self.nt
        self.sigma = [None] * self.nt

        x_all = T.concatenate(self.x, axis=0)
        ns_tot = x_all.shape[0]

        for t in range(self.nt):
            z2 = (self.z[t] ** 2).mean(axis=0)  # (m,)
            ns = self.x_wno[t].shape[0]
            R = T.dot(self.z[t].T, self.x[t]) / ns  # m, nv
            R = R / T.sqrt(z2).reshape((self.m, 1))  # as <x^2_i> == 1 we don't divide by it

            # calculate priors
            z_all_mean = T.dot(x_all, self.ws[t].T)
            z_all_noise = self.rng.normal(avg=0.0, std=self.y_scale, size=(ns_tot, self.m))
            z_all = z_all_mean + z_all_noise
            z2_all = (z_all ** 2).mean(axis=0)  # (m,)
            R_all = T.dot(z_all.T, x_all) / ns_tot  # m, nv
            R_all = R_all / T.sqrt(z2_all).reshape((self.m, 1))  # as <x^2_i> == 1 we don't divide by it

            # add priors
            z2 = (1 - self.lamb) * z2 + self.lamb * z2_all
            R = (1 - self.lamb) * R + self.lamb * R_all

            # the rest depends on R and z2 only
            ri = ((R ** 2) / T.clip(1 - R ** 2, EPS, 1 - EPS)).sum(axis=0)  # (nv,)

            # v_xi | z conditional mean
            outer_term = (1 / (1 + ri)).reshape((1, self.nv))
            inner_term_1 = (R / T.clip(1 - R ** 2, EPS, 1) / T.sqrt(z2).reshape((self.m, 1))).reshape(
                (1, self.m, self.nv))
            # NOTE: we use z[t], but seems only for objective not for the covariance estimate
            inner_term_2 = self.z[t].reshape((ns, self.m, 1))
            cond_mean = outer_term * ((inner_term_1 * inner_term_2).sum(axis=1))  # (ns, nv)

            inner_mat = 1.0 / (1 + ri).reshape((1, self.nv)) * R / T.clip(1 - R ** 2, EPS, 1)
            self.sigma[t] = T.dot(inner_mat.T, inner_mat)
            self.sigma[t] = self.sigma[t] * (1 - T.eye(self.nv)) + T.eye(self.nv)

            # objective
            obj_part_1 = 0.5 * T.log(T.clip(((self.x[t] - cond_mean) ** 2).mean(axis=0), EPS, np.inf)).sum(axis=0)
            obj_part_2 = 0.5 * T.log(z2).sum(axis=0)
            self.objs[t] = obj_part_1 + obj_part_2

        self.main_obj = T.sum(self.objs)

        # regularization
        self.reg_obj = T.constant(0)

        if self.l1 > 0:
            l1_reg = T.sum([T.abs_(self.ws[t + 1] - self.ws[t]).sum() for t in range(self.nt - 1)])
            self.reg_obj = self.reg_obj + self.l1 * l1_reg

        if self.l2 > 0:
            l2_reg = T.sum([T.square(self.ws[t + 1] - self.ws[t]).sum() for t in range(self.nt - 1)])
            self.reg_obj = self.reg_obj + self.l2 * l2_reg

        self.total_obj = self.main_obj + self.reg_obj

        # optimizer
        updates = lasagne.updates.adam(self.total_obj, self.ws)

        # functions
        self.train_step = theano.function(inputs=self.x_wno,
                                          outputs=[self.total_obj, self.main_obj, self.reg_obj] + self.objs,
                                          updates=updates)

        self.get_norm_covariance = theano.function(inputs=self.x_wno,
                                                   outputs=self.sigma)

    def fit(self, x):
        # TODO: write a stopping condition

        # fit a linear corex to get the priors
        lin_corex = theano_linear_corex.Corex(nv=self.nv,
                                              n_hidden=self.m,
                                              max_iter=500,
                                              anneal=True,
                                              verbose=self.verbose)
        lin_corex.fit(np.concatenate(x, axis=0))

        x = [np.array(xt, dtype=np.float32) for xt in x]
        dummy = self.preprocess(x, fit=True)  # Fit a transform for each marginal

        # use the priors to estimate <X_i> and std(x_i) better
        for t in range(self.nt):
            cur_mean = (1 - self.lamb) * self.theta[t][0] + self.lamb * lin_corex.theta[0]
            cur_std = (1 - self.lamb) * self.theta[t][1] + self.lamb * lin_corex.theta[1]
            self.theta[t] = (cur_mean, cur_std)

        x = self.preprocess(x, fit=False)  # standardize the data using the better estimates
        self.x_std = x  # to have an access to standardized x

        # initialize W matrices with lin_corex matrix
        self.pretrained_weights = [lin_corex.ws.get_value()] * self.nt

        if self.pretrained_weights is not None:
            for cur_w, pre_w in zip(self.ws, self.pretrained_weights):
                cur_w.set_value(pre_w.astype(np.float32))

        anneal_schedule = [0.]
        if self.anneal:
            anneal_schedule = [0.6 ** k for k in range(1, 7)] + [0]

        for i_eps, eps in enumerate(anneal_schedule):
            start_time = time.time()

            self.eps = eps
            self.anneal_eps.set_value(np.float32(eps))
            self.moments = self._calculate_moments(x, self.ws, quick=True)
            self._update_u(x)

            for i_loop in range(self.max_iter):
                ret = self.train_step(*x)
                obj = ret[0]
                reg_obj = ret[2]

                if i_loop % self.update_iter == 0 and self.verbose:
                    self.moments = self._calculate_moments(x, self.ws, quick=True)
                    self._update_u(x)
                    print("tc = {}, obj = {}, reg = {}, eps = {}".format(np.sum(self.tc),
                                                                         obj, reg_obj, eps))

            print("Annealing iteration finished, time = {}".format(time.time() - start_time))

        self.moments = self._calculate_moments(x, self.ws, quick=False)  # Update moments with details
        return self

    def get_covariance(self):
        norm_cov = self.get_norm_covariance(*list(np.array(self.x_std).astype(np.float32)))
        ret = [None] * self.nt
        for t in range(self.nt):
            ret[t] = self.theta[t][1][:, np.newaxis] * self.theta[t][1] * norm_cov[t]
        return ret


class TimeCorexWPriorWeights(TimeCorex):
    def __init__(self, l1=0.0, l2=0.0, lamb=0.5, gamma=2, **kwargs):
        """
        :param gamma: parameter that controls the decay of weights.
                      weight of a sample whose distance from the current time-step is d will have 1.0/(gamma^d).
                      If gamma is equal to 1 all samples will be used with weight 1.
        """
        super(TimeCorexWPriorWeights, self).__init__(**kwargs)
        self.l1 = l1
        self.l2 = l2
        self.lamb = lamb
        self.gamma = gamma
        self._define_model()

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

        EPS = 1e-5
        self.objs = [None] * self.nt
        self.sigma = [None] * self.nt

        x_all = T.concatenate(self.x, axis=0)
        ns_tot = x_all.shape[0]

        for t in range(self.nt):
            z2 = (self.z[t] ** 2).mean(axis=0)  # (m,)
            ns = self.x_wno[t].shape[0]
            R = T.dot(self.z[t].T, self.x[t]) / ns  # m, nv
            R = R / T.sqrt(z2).reshape((self.m, 1))  # as <x^2_i> == 1 we don't divide by it

            # calculate priors
            weights = []
            for i in range(self.nt):
                cur_ns = self.x_wno[i].shape[0]
                weights.append(T.ones((cur_ns,)) / np.power(self.gamma, np.abs(i - t)))
            weights = T.concatenate(weights, axis=0)
            weights = weights / T.sum(weights)
            weights = weights.reshape((-1, 1))

            z_all_mean = T.dot(x_all, self.ws[t].T)
            z_all_noise = self.rng.normal(avg=0.0, std=self.y_scale, size=(ns_tot, self.m))
            z_all = z_all_mean + z_all_noise
            z2_all = ((z_all ** 2) * weights).sum(axis=0)  # (m,)
            R_all = T.dot((z_all * weights).T, x_all)  # m, nv
            R_all = R_all / T.sqrt(z2_all).reshape((self.m, 1))  # as <x^2_i> == 1 we don't divide by it

            # add priors
            z2 = (1 - self.lamb) * z2 + self.lamb * z2_all
            R = (1 - self.lamb) * R + self.lamb * R_all

            # the rest depends on R and z2 only
            ri = ((R ** 2) / T.clip(1 - R ** 2, EPS, 1 - EPS)).sum(axis=0)  # (nv,)

            # v_xi | z conditional mean
            outer_term = (1 / (1 + ri)).reshape((1, self.nv))
            inner_term_1 = (R / T.clip(1 - R ** 2, EPS, 1) / T.sqrt(z2).reshape((self.m, 1))).reshape(
                (1, self.m, self.nv))
            # NOTE: we use z[t], but seems only for objective not for the covariance estimate
            inner_term_2 = self.z[t].reshape((ns, self.m, 1))
            cond_mean = outer_term * ((inner_term_1 * inner_term_2).sum(axis=1))  # (ns, nv)

            inner_mat = 1.0 / (1 + ri).reshape((1, self.nv)) * R / T.clip(1 - R ** 2, EPS, 1)
            self.sigma[t] = T.dot(inner_mat.T, inner_mat)
            self.sigma[t] = self.sigma[t] * (1 - T.eye(self.nv)) + T.eye(self.nv)

            # objective
            obj_part_1 = 0.5 * T.log(T.clip(((self.x[t] - cond_mean) ** 2).mean(axis=0), EPS, np.inf)).sum(axis=0)
            obj_part_2 = 0.5 * T.log(z2).sum(axis=0)
            self.objs[t] = obj_part_1 + obj_part_2

        self.main_obj = T.sum(self.objs)

        # regularization
        self.reg_obj = T.constant(0)

        if self.l1 > 0:
            l1_reg = T.sum([T.abs_(self.ws[t + 1] - self.ws[t]).sum() for t in range(self.nt - 1)])
            self.reg_obj = self.reg_obj + self.l1 * l1_reg

        if self.l2 > 0:
            l2_reg = T.sum([T.square(self.ws[t + 1] - self.ws[t]).sum() for t in range(self.nt - 1)])
            self.reg_obj = self.reg_obj + self.l2 * l2_reg

        self.total_obj = self.main_obj + self.reg_obj

        # optimizer
        updates = lasagne.updates.adam(self.total_obj, self.ws)

        # functions
        self.train_step = theano.function(inputs=self.x_wno,
                                          outputs=[self.total_obj, self.main_obj, self.reg_obj] + self.objs,
                                          updates=updates)

        self.get_norm_covariance = theano.function(inputs=self.x_wno,
                                                   outputs=self.sigma)

    def fit(self, x):
        # TODO: write a stopping condition

        # fit a linear corex
        lin_corex = theano_linear_corex.Corex(nv=self.nv,
                                              n_hidden=self.m,
                                              max_iter=500,
                                              anneal=True,
                                              verbose=self.verbose)
        lin_corex.fit(np.concatenate(x, axis=0))

        dummy = self.preprocess(x, fit=True)  # Fit a transform for each marginal

        # use the priors to estimate <X_i> and std(x_i) better
        for t in range(self.nt):
            sum_weights = 0.0
            for i in range(self.nt):
                w = 1.0 / np.power(self.gamma, np.abs(i - t))
                sum_weights += x[i].shape[0] * w

            mean_prior = np.zeros((self.nv,))
            for i in range(self.nt):
                w = 1.0 / np.power(self.gamma, np.abs(i - t))
                for sample in x[i]:
                    mean_prior += w * np.array(sample)
            mean_prior /= sum_weights

            var_prior = np.zeros((self.nv,))
            for i in range(self.nt):
                w = 1.0 / np.power(self.gamma, np.abs(i - t))
                for sample in x[i]:
                    var_prior += w * ((np.array(sample) - mean_prior) ** 2)
            var_prior /= sum_weights
            std_prior = np.sqrt(var_prior)

            cur_mean = (1 - self.lamb) * self.theta[t][0] + self.lamb * mean_prior
            cur_std = (1 - self.lamb) * self.theta[t][1] + self.lamb * std_prior
            self.theta[t] = (cur_mean, cur_std)

        x = self.preprocess(x, fit=False)  # standardize the data using the better estimates
        x = [np.array(xt, dtype=np.float32) for xt in x]  # conver to np.float32
        self.x_std = x  # to have an access to standardized x

        # initialize W matrices with lin_corex matrix
        self.pretrained_weights = [lin_corex.ws.get_value()] * self.nt

        if self.pretrained_weights is not None:
            for cur_w, pre_w in zip(self.ws, self.pretrained_weights):
                cur_w.set_value(pre_w.astype(np.float32))

        anneal_schedule = [0.]
        if self.anneal:
            anneal_schedule = [0.6 ** k for k in range(1, 7)] + [0]

        for i_eps, eps in enumerate(anneal_schedule):
            start_time = time.time()

            self.eps = eps
            self.anneal_eps.set_value(np.float32(eps))
            self.moments = self._calculate_moments(x, self.ws, quick=True)
            self._update_u(x)

            for i_loop in range(self.max_iter):
                ret = self.train_step(*x)
                obj = ret[0]
                reg_obj = ret[2]

                if i_loop % self.update_iter == 0 and self.verbose:
                    self.moments = self._calculate_moments(x, self.ws, quick=True)
                    self._update_u(x)
                    print("tc = {}, obj = {}, reg = {}, eps = {}".format(np.sum(self.tc),
                                                                         obj, reg_obj, eps))

            print("Annealing iteration finished, time = {}".format(time.time() - start_time))

        self.moments = self._calculate_moments(x, self.ws, quick=False)  # Update moments with details
        return self

    def get_covariance(self):
        norm_cov = self.get_norm_covariance(*list(np.array(self.x_std).astype(np.float32)))
        ret = [None] * self.nt
        for t in range(self.nt):
            ret[t] = self.theta[t][1][:, np.newaxis] * self.theta[t][1] * norm_cov[t]
        return ret


class TimeCorexWPriorOnlyWeights(TimeCorex):
    def __init__(self, l1=0.0, l2=0.0, gamma=2, **kwargs):
        """
        :param gamma: parameter that controls the decay of weights.
                      weight of a sample whose distance from the current time-step is d will have 1.0/(gamma^d).
                      If gamma is equal to 1 all samples will be used with weight 1.
        """
        super(TimeCorexWPriorOnlyWeights, self).__init__(**kwargs)
        self.l1 = l1
        self.l2 = l2
        self.gamma = gamma
        self._define_model()

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

        EPS = 1e-5
        self.objs = [None] * self.nt
        self.sigma = [None] * self.nt

        x_all = T.concatenate(self.x, axis=0)
        ns_tot = x_all.shape[0]

        for t in range(self.nt):
            weights = []
            for i in range(self.nt):
                cur_ns = self.x_wno[i].shape[0]
                weights.append(T.ones((cur_ns,)) / np.power(self.gamma, np.abs(i - t)))
            weights = T.concatenate(weights, axis=0)
            weights = weights / T.sum(weights)
            weights = weights.reshape((-1, 1))

            z_all_mean = T.dot(x_all, self.ws[t].T)
            z_all_noise = self.rng.normal(avg=0.0, std=self.y_scale, size=(ns_tot, self.m))
            z_all = z_all_mean + z_all_noise
            z2_all = ((z_all ** 2) * weights).sum(axis=0)  # (m,)
            R_all = T.dot((z_all * weights).T, x_all)  # m, nv
            R_all = R_all / T.sqrt(z2_all).reshape((self.m, 1))  # as <x^2_i> == 1 we don't divide by it

            z2 = z2_all
            R = R_all

            # the rest depends on R and z2 only
            ri = ((R ** 2) / T.clip(1 - R ** 2, EPS, 1 - EPS)).sum(axis=0)  # (nv,)

            # v_xi | z conditional mean
            outer_term = (1 / (1 + ri)).reshape((1, self.nv))
            inner_term_1 = (R / T.clip(1 - R ** 2, EPS, 1) / T.sqrt(z2).reshape((self.m, 1))).reshape(
                (1, self.m, self.nv))
            # NOTE: we use z[t], but seems only for objective not for the covariance estimate
            # TODO: try write the loss function using all samples
            inner_term_2 = self.z[t].reshape((ns, self.m, 1))
            cond_mean = outer_term * ((inner_term_1 * inner_term_2).sum(axis=1))  # (ns, nv)

            inner_mat = 1.0 / (1 + ri).reshape((1, self.nv)) * R / T.clip(1 - R ** 2, EPS, 1)
            self.sigma[t] = T.dot(inner_mat.T, inner_mat)
            self.sigma[t] = self.sigma[t] * (1 - T.eye(self.nv)) + T.eye(self.nv)

            # objective
            obj_part_1 = 0.5 * T.log(T.clip(((self.x[t] - cond_mean) ** 2).mean(axis=0), EPS, np.inf)).sum(axis=0)
            obj_part_2 = 0.5 * T.log(z2).sum(axis=0)
            self.objs[t] = obj_part_1 + obj_part_2

        self.main_obj = T.sum(self.objs)

        # regularization
        self.reg_obj = T.constant(0)

        if self.l1 > 0:
            l1_reg = T.sum([T.abs_(self.ws[t + 1] - self.ws[t]).sum() for t in range(self.nt - 1)])
            self.reg_obj = self.reg_obj + self.l1 * l1_reg

        if self.l2 > 0:
            l2_reg = T.sum([T.square(self.ws[t + 1] - self.ws[t]).sum() for t in range(self.nt - 1)])
            self.reg_obj = self.reg_obj + self.l2 * l2_reg

        self.total_obj = self.main_obj + self.reg_obj

        # optimizer
        updates = lasagne.updates.adam(self.total_obj, self.ws)

        # functions
        self.train_step = theano.function(inputs=self.x_wno,
                                          outputs=[self.total_obj, self.main_obj, self.reg_obj] + self.objs,
                                          updates=updates)

        self.get_norm_covariance = theano.function(inputs=self.x_wno,
                                                   outputs=self.sigma)

    def fit(self, x):
        # TODO: write a stopping condition

        # fit a linear corex
        lin_corex = theano_linear_corex.Corex(nv=self.nv,
                                              n_hidden=self.m,
                                              max_iter=500,
                                              anneal=True,
                                              verbose=self.verbose)
        lin_corex.fit(np.concatenate(x, axis=0))

        dummy = self.preprocess(x, fit=True)  # Fit a transform for each marginal

        # use the priors to estimate <X_i> and std(x_i) better
        for t in range(self.nt):
            sum_weights = 0.0
            for i in range(self.nt):
                w = 1.0 / np.power(self.gamma, np.abs(i - t))
                sum_weights += x[i].shape[0] * w

            mean_prior = np.zeros((self.nv,))
            for i in range(self.nt):
                w = 1.0 / np.power(self.gamma, np.abs(i - t))
                for sample in x[i]:
                    mean_prior += w * np.array(sample)
            mean_prior /= sum_weights

            var_prior = np.zeros((self.nv,))
            for i in range(self.nt):
                w = 1.0 / np.power(self.gamma, np.abs(i - t))
                for sample in x[i]:
                    var_prior += w * ((np.array(sample) - mean_prior) ** 2)
            var_prior /= sum_weights
            std_prior = np.sqrt(var_prior)

            self.theta[t] = (mean_prior, std_prior)

        x = self.preprocess(x, fit=False)  # standardize the data using the better estimates
        x = [np.array(xt, dtype=np.float32) for xt in x]  # conver to np.float32
        self.x_std = x  # to have an access to standardized x

        # initialize W matrices with lin_corex matrix
        self.pretrained_weights = [lin_corex.ws.get_value()] * self.nt
        self.pretrained_weights = None  # TODO: remove

        if self.pretrained_weights is not None:
            for cur_w, pre_w in zip(self.ws, self.pretrained_weights):
                cur_w.set_value(pre_w.astype(np.float32))

        anneal_schedule = [0.]
        if self.anneal:
            anneal_schedule = [0.6 ** k for k in range(1, 7)] + [0]

        for i_eps, eps in enumerate(anneal_schedule):
            start_time = time.time()

            self.eps = eps
            self.anneal_eps.set_value(np.float32(eps))
            self.moments = self._calculate_moments(x, self.ws, quick=True)
            self._update_u(x)

            for i_loop in range(self.max_iter):
                ret = self.train_step(*x)
                obj = ret[0]
                reg_obj = ret[2]

                if i_loop % self.update_iter == 0 and self.verbose:
                    self.moments = self._calculate_moments(x, self.ws, quick=True)
                    self._update_u(x)
                    print("tc = {}, obj = {}, reg = {}, eps = {}".format(np.sum(self.tc),
                                                                         obj, reg_obj, eps))

            print("Annealing iteration finished, time = {}".format(time.time() - start_time))

        self.moments = self._calculate_moments(x, self.ws, quick=False)  # Update moments with details
        return self

    def get_covariance(self):
        norm_cov = self.get_norm_covariance(*list(np.array(self.x_std).astype(np.float32)))
        ret = [None] * self.nt
        for t in range(self.nt):
            ret[t] = self.theta[t][1][:, np.newaxis] * self.theta[t][1] * norm_cov[t]
        return ret
