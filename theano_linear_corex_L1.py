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


class Corex:

    def __init__(self, nv, n_hidden=10, max_iter=10000, tol=1e-5, anneal=True, missing_values=None,
                 discourage_overlap=True, gaussianize='standard', gpu=False, y_scale=1.0, l1=0.0,
                 verbose=False, seed=None):

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

        EPS = 1e-6

        z2 = (self.z ** 2).mean(axis=0)  # (m,)
        R = T.dot(self.z.T, self.x) / ns  # m, nv
        R = R / T.sqrt(z2).reshape((self.m, 1))  # as <x^2_i> == 1 we don't divide by it
        ri = ((R ** 2) / T.clip(1 - R ** 2, EPS, 1 - EPS)).sum(axis=0)  # (nv,)

        # v_xi | z conditional mean
        outer_term = (1 / (1 + ri)).reshape((1, self.nv))
        inner_term_1 = (R / T.clip(1 - R ** 2, EPS, 1) / T.sqrt(z2).reshape((self.m, 1))).reshape((1, self.m, self.nv))
        inner_term_2 = self.z.reshape((ns, self.m, 1))
        # TODO: use discourage overlap ?
        cond_mean = outer_term * ((inner_term_1 * inner_term_2).sum(axis=1))  # (ns, nv)

        # objective
        obj_part_1 = 0.5 * T.log(T.clip(((self.x - cond_mean) ** 2).mean(axis=0), EPS, np.inf)).sum(axis=0)
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

        anneal_schedule = [0.]
        if self.anneal:
            anneal_schedule = [0.6 ** k for k in range(1, 7)] + [0]

        for i_eps, eps in enumerate(anneal_schedule):
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
                (obj, _) = self.train_step(x.astype(np.float32))

                if i_loop % 15 == 0:
                    self.moments = self._calculate_moments(x, self.ws, quick=True)
                    self._update_u(x)
                    print("tc = {}, obj = {}, eps = {}".format(self.tc, obj, eps))

        self.moments = self._calculate_moments(x, self.ws, quick=False)  # Update moments with details
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
        return np.argmax(np.abs(self.us), axis=0)  # TODO: understand this

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
        us = self.getus(ws.get_value(), x)
        if self.discourage_overlap:
            return self._calculate_moments_ns(x, us, quick=quick)
        else:
            return self._calculate_moments_syn(x, us, quick=quick)

    def _calculate_moments_ns(self, x, ws, quick=False):
        """Calculate moments based on the weights and samples. We also calculate and save MI, TC, additivity, and
        the value of the objective. Note it is assumed that <X_i^2> = 1! """
        n_samples = x.shape[0]
        m = {}  # Dictionary of moments
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

        if np.max(m["uj"]) > 1.0:
            print(np.max(m["uj"]))
            assert False

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
        if self.gpu and fit:  # Don't return GPU matrices when only transforming
            x = cm.CUDAMatrix(x)
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
        if self.discourage_overlap:
            z = m['rhoinvrho'] / (1 + m['Si'])
            cov = np.dot(z.T, z)
            cov /= (1. - self.eps ** 2)
            np.fill_diagonal(cov, 1)
            return self.theta[1][:, np.newaxis] * self.theta[1] * cov
        else:
            cov = np.einsum('ij,kj->ik', m["X_i Z_j"], m["X_i Y_j"])
            np.fill_diagonal(cov, 1)
            return self.theta[1][:, np.newaxis] * self.theta[1] * cov


def main():
    import pandas as pd
    import cPickle as pkl

    with open('../data/EOD_week.pkl', 'rb') as f:
        df = pd.DataFrame(pkl.load(f))
    print("Data.shape = {}".format(df.shape))

    corex = Corex(nv=df.shape[1], n_hidden=10, max_iter=300, verbose=True, anneal=True)
    corex.fit(df[10:20])


if __name__ == '__main__':
    main()
