from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from scipy.stats import norm, rankdata
from tcorex import base
import numpy as np
import time
import torch


def get_w_from_u(corex):
    u = corex.ws
    z2 = corex.moments['Y_j^2']
    return u * np.sqrt(z2).reshape((-1, 1))


def get_u_from_w(corex):
    w = corex.get_weights()
    z2 = corex.forward(corex.x_input, 0)['z2']
    z2 = base.to_numpy(z2)
    return w / np.sqrt(z2).reshape((-1, 1))


class Corex:
    def __init__(self, nv, n_hidden=10, max_iter=1000, tol=1e-5, anneal=True, missing_values=None,
                 gaussianize='standard', y_scale=1.0, l1=0.0, device='cpu', verbose=0):
        self.nv = nv  # Number of variables
        self.m = n_hidden  # Number of latent factors to learn
        self.max_iter = max_iter  # Number of iterations to try
        self.tol = tol  # Threshold for convergence
        self.anneal = anneal
        self.missing_values = missing_values
        self.gaussianize = gaussianize  # Preprocess data: 'standard' scales to zero mean and unit variance
        self.y_scale = y_scale  # Can be arbitrary, but sets the scale of Y
        self.l1 = l1  # L1 on W regularization coefficient
        self.device = torch.device(device)
        self.verbose = verbose
        if verbose:
            np.set_printoptions(precision=3, suppress=True, linewidth=160)
            print('Linear CorEx with {:d} latent factors'.format(n_hidden))

        # define the weights of the model
        self.ws = np.random.normal(loc=0, scale=1.0 / np.sqrt(self.nv), size=(self.m, self.nv))
        self.ws = torch.tensor(self.ws, dtype=torch.float, device=self.device, requires_grad=True)
        # TODO: decide whether to use orthogonal initialization; do it for T-CorEx too
        # self.ws = torch.zeros((self.m, self.nv), dtype=torch.float, device=self.device, requires_grad=True)
        # self.ws = torch.nn.init.orthogonal_(self.ws, gain=1.0/np.sqrt(self.nv))

    def forward(self, x_wno, anneal_eps, compute_sigma=False):
        x_wno = torch.tensor(x_wno, dtype=torch.float, device=self.device)
        anneal_eps = torch.tensor(anneal_eps, dtype=torch.float, device=self.device)

        ns = x_wno.shape[0]
        anneal_noise = torch.randn((ns, self.nv), dtype=torch.float, device=self.device)
        x = torch.sqrt(1 - anneal_eps ** 2) * x_wno + anneal_eps * anneal_noise
        z_noise = self.y_scale * torch.randn((ns, self.m), dtype=torch.float, device=self.device)
        z_mean = torch.mm(x, self.ws.t())
        z = z_mean + z_noise

        epsilon = 1e-8
        z2 = (z ** 2).mean(dim=0)  # (m,)
        R = torch.mm(z.t(), x) / ns  # m, nv
        R = R / torch.sqrt(z2).reshape((self.m, 1))  # as <x^2_i> == 1 we don't divide by it
        ri = ((R ** 2) / torch.clamp(1 - R ** 2, epsilon, 1 - epsilon)).sum(dim=0)  # (nv,)

        # v_xi | z conditional mean
        outer_term = (1 / (1 + ri)).reshape((1, self.nv))
        inner_term_1 = R / torch.clamp(1 - R ** 2, epsilon, 1) / torch.sqrt(z2).reshape((self.m, 1))  # (m, nv)
        inner_term_2 = z  # (ns, m)
        cond_mean = outer_term * torch.mm(inner_term_2, inner_term_1)  # (ns, nv)

        sigma = None
        if compute_sigma:
            inner_mat = 1.0 / (1 + ri).reshape((1, self.nv)) * R / torch.clamp(1 - R ** 2, epsilon, 1)
            sigma = torch.mm(inner_mat.t(), inner_mat)
            identity_matrix = torch.eye(self.nv, dtype=torch.float, device=self.device)
            sigma = sigma * (1 - identity_matrix) + identity_matrix

        # objective
        obj_part_1 = 0.5 * torch.log(torch.clamp(((x - cond_mean) ** 2).mean(dim=0), epsilon, np.inf)).sum(dim=0)
        obj_part_2 = 0.5 * torch.log(z2).sum(dim=0)
        reg_obj = torch.tensor(0, dtype=torch.float, device=self.device)
        if self.l1 > 0:
            reg_obj = torch.sum(self.l1 * torch.abs(self.ws))

        obj = obj_part_1 + obj_part_2 + reg_obj

        return {'obj': obj,
                'main_obj': obj_part_1 + obj_part_2,
                'reg_obj': reg_obj,
                'z2': z2,
                'R': R,
                'sigma': sigma}

    def fit(self, x):
        x = np.asarray(x, dtype=np.float32)
        x = self.preprocess(x, fit=True)  # Fit a transform for each marginal
        assert x.shape[1] == self.nv
        self.x_input = x  # to have access to the standardized version of input

        anneal_schedule = [0.]
        if self.anneal:
            anneal_schedule = [0.6 ** k for k in range(1, 7)] + [0]

        # set up the optimizer
        optimizer = torch.optim.Adam([self.ws])

        for i_eps, eps in enumerate(anneal_schedule):
            start_time = time.time()

            history = []
            last_iter = 0

            for i_loop in range(self.max_iter):
                obj = self.forward(x, eps)['obj']
                history.append(base.to_numpy(obj))
                last_iter = i_loop

                optimizer.zero_grad()
                obj.backward()
                optimizer.step()

                # Stopping criterion
                # NOTE: this parameter can be tuned probably
                K = 50
                delta = 1.0
                if len(history) >= 2*K:
                    prev_mean = np.mean(history[-2*K:-K])
                    cur_mean = np.mean(history[-K:])
                    delta = np.abs(prev_mean - cur_mean) / np.abs(prev_mean + 1e-6)
                if delta < self.tol:
                    break

                if self.verbose > 0:
                    print("eps: {}, iter: {} / {}, obj: {:.4f}, delta: {:.6f}".format(
                        eps, i_loop, self.max_iter, history[-1], delta), end='\r')

            print("Annealing iteration finished, iters: {}, time: {:.2f}s".format(last_iter+1,
                                                                                  time.time() - start_time))

        # clear cache to free some GPU memory
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return self

    def get_weights(self):
        return base.to_numpy(self.ws)

    def mis(self):
        """ Returns I (Z_j : X_i) """
        R = self.forward(self.x_input, 0)['R']
        R = base.to_numpy(R)
        return -0.5 * np.log1p(R ** 2)

    def clusters(self, type='MI'):
        """ Get clusters of variables.
        :param type: MI or W. In case of MI, the cluster is defined as argmax_j I(x_i : z_j).
                     In case of W, the cluster is defined as argmax_j |W_{j,i}|
        """
        if type == 'W':
            return np.abs(self.get_weights()).argmax(axis=0)
        return self.mis.argmax(axis=0)

    def transform(self, x):
        """ Transform an array of inputs, x, into an array of k latent factors, Y. """
        x = self.preprocess(x)
        return np.dot(x, self.get_weights().T)

    def preprocess(self, x, fit=False):
        """Transform each marginal to be as close to a standard Gaussian as possible.
        'standard' (default) just subtracts the mean and scales by the std.
        'empirical' does an empirical gaussianization (but this cannot be inverted).
        'outliers' tries to squeeze in the outliers
        Any other choice will skip the transformation."""
        if self.missing_values is not None:
            x, self.n_obs = base.mean_impute(x, self.missing_values)  # Creates a copy
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
            x = base.g((x - self.theta[0]) / self.theta[1])  # g truncates long tails
        elif self.gaussianize == 'empirical':
            print("Warning: correct inversion/transform of empirical gauss transform not implemented.")
            x = np.array([norm.ppf((rankdata(x_i) - 0.5) / len(x_i)) for x_i in x.T]).T
        return x

    def get_covariance(self, normed=False):
        sigma = self.forward(self.x_input, 0)['sigma']
        sigma = base.to_numpy(sigma)
        if normed:
            return sigma
        return self.theta[1][:, np.newaxis] * self.theta[1] * sigma

    def load_weights(self, w):
        self.ws = torch.tensor(w, dtype=torch.float, device=self.device, requires_grad=True)
