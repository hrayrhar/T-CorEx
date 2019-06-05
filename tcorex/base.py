""" Base functions/classes for T-CorEx methods.
Some parts of the code are borrowed from https://github.com/gregversteeg/LinearCorex.
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from .experiments.misc import make_sure_path_exists
from scipy.stats import norm, rankdata
import numpy as np
import time
import torch
import os


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


def to_numpy(x):
    if x.requires_grad:
        x = x.detach()
    if x.device.type != 'cpu':
        x = x.cpu()
    return x.numpy()


def save(model, path, verbose=False):
    save_dir = os.path.dirname(path)
    make_sure_path_exists(save_dir)
    if verbose:
        print('Saving into {}'.format(path))
    torch.save(model, path)


def load(path):
    return torch.load(path)


class TCorexBase(object):
    """ Base class for all T-CorEx methods.
    """
    def __init__(self, nt, nv, n_hidden=10, max_iter=1000, tol=1e-5, anneal=True, missing_values=None,
                 gaussianize='standard', pretrained_weights=None, device='cpu', stopping_len=50, verbose=0,
                 optimizer_class=torch.optim.Adam, optimizer_params={}):
        """
        :param nt: int, number of time periods
        :param nv: int, number of observed variables
        :param n_hidden: int, number of latent factors
        :param max_iter: int, maximum number of iterations to train in each annealing step
        :param tol: float, threshold for checking convergence
        :param anneal: boolean, whether to use annealing or not
        :param missing_values: float or None, value used for imputing missing values. None indicates imputing means.
        :param gaussianize: str, 'none', 'standard', 'outliers', or 'empirical'. Specifies to normalize the data.
        :param pretrained_weights: None or list of numpy arrays. Pretrained weights.
        :param device: str, 'cpu' or 'cuda'. The device parameter passed to PyTorch.
        :param stopping_len: int, the length of history used for detecting convergence.
        :param verbose: 0, 1, or 2. Specifies the verbosity level.
        :param optimizer_class: optimizer class like torch.optim.Adam
        :param optimizer_params: dictionary listing parameters of the optimizer
        """
        self.nt = nt
        self.nv = nv
        self.m = n_hidden
        self.max_iter = max_iter
        self.tol = tol
        self.anneal = anneal
        self.missing_values = missing_values
        self.gaussianize = gaussianize
        self.pretrained_weights = pretrained_weights
        self.device = torch.device(device)
        self.stopping_len = stopping_len
        self.verbose = verbose
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        if verbose > 0:
            np.set_printoptions(precision=3, suppress=True, linewidth=160)
            print('Linear CorEx with {:d} latent factors'.format(n_hidden))
        self.add_params = []  # used in _train_loop()

        # initialize later
        self.x_input = None
        self.theta = None
        self.ws = None

    def forward(self, x_wno, anneal_eps, indices=None, return_factorization=False, **kwargs):
        raise NotImplementedError("forward function should be specified for all child classes")

    def _train_loop(self):
        """ train loop expects self have one variable x_input
        """
        if self.verbose > 0:
            print("Starting the training loop ...")
        # set the annealing schedule
        anneal_schedule = [0.]
        if self.anneal:
            anneal_schedule = [0.6 ** k for k in range(1, 7)] + [0]

        # initialize the weights if pre-trained weights are specified
        if self.pretrained_weights is not None:
            self.load_weights(self.pretrained_weights)

        # set up the optimizer
        optimizer = self.optimizer_class(self.ws, **self.optimizer_params)

        for eps in anneal_schedule:
            start_time = time.time()

            history = []
            last_iter = 0

            for i_loop in range(self.max_iter):
                ret = self.forward(self.x_input, eps)
                obj = ret['total_obj']
                history.append(to_numpy(obj))
                last_iter = i_loop

                optimizer.zero_grad()
                obj.backward()
                optimizer.step()

                # Stopping criterion
                # NOTE: this parameter can be tuned probably
                delta = 1.0
                if len(history) >= 2 * self.stopping_len:
                    prev_mean = np.mean(history[-2 * self.stopping_len:-self.stopping_len])
                    cur_mean = np.mean(history[-self.stopping_len:])
                    delta = np.abs(prev_mean - cur_mean) / np.abs(prev_mean + 1e-6)
                if delta < self.tol:
                    break

                main_obj = ret['main_obj']
                reg_obj = ret['reg_obj']
                if self.verbose > 1:
                    print("eps: {:.4f}, iter: {} / {}, obj: {:.4f}, main: {:.4f}, reg: {:.4f}, delta: {:.6f} ".format(
                        eps, i_loop, self.max_iter, obj, main_obj, reg_obj, delta), end='\r')

            if self.verbose > 0:
                print("Annealing iteration finished, iters: {}, time: {:.2f}s".format(
                    last_iter + 1, time.time() - start_time))

        # clear cache to free some GPU memory
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return self

    def fit(self, x):
        x = [np.array(xt, dtype=np.float32) for xt in x]
        x = self.preprocess(x, fit=True)  # fit a transform for each marginal
        self.x_input = x  # to have an access to input
        return self._train_loop()

    def get_weights(self):
        return [to_numpy(w) for w in self.ws]

    def mis(self):
        """ Returns I (Z_j : X_i) for each time period. """
        R = self.forward(self.x_input, 0, return_R=True)['R']
        R = [to_numpy(rho) for rho in R]
        eps = 1e-6
        R = [np.clip(rho, -1 + eps, 1 - eps) for rho in R]
        return [-0.5 * np.log1p(-rho ** 2) for rho in R]

    def clusters(self, cluster_type='MI'):
        """ Get clusters of variables for each time period. """
        return [mi.argmax(axis=0) for mi in self.mis()]

    def transform(self, x):
        """ Transform an array of inputs, x, into an array of k latent factors, Y. """
        x = self.preprocess(x)
        ws = self.get_weights()
        ret = [a.dot(w.T) for (a, w) in zip(x, ws)]
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
                if np.max(np.abs(x)) > 6 and self.verbose > 0:
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

    def get_covariance(self, indices=None, normed=False):
        if indices is None:
            indices = range(self.nt)
        cov = self.forward(self.x_input, anneal_eps=0, indices=indices)['sigma']
        for t in range(self.nt):
            if cov[t] is None:
                continue
            cov[t] = to_numpy(cov[t])
            if not normed:
                cov[t] = self.theta[t][1][:, np.newaxis] * self.theta[t][1] * cov[t]
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        return cov

    def get_factorization(self):
        ret = self.forward(self.x_input, anneal_eps=0, return_factorization=True)['factorization']
        for t in range(self.nt):
            ret[t] = to_numpy(ret[t])
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        return ret

    def load_weights(self, weights):
        self.ws = [torch.tensor(w, dtype=torch.float, device=self.device, requires_grad=True)
                   for w in weights]
