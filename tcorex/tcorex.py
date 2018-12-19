""" T-CorEx - Linear Time Covariance Estimation method based on Linear CorEx.
Given time-series data returns the covariance matrix for each time period.
More details coming soon ...
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from .base import TCorexBase
from .corex import Corex
import numpy as np
import time
import random
import torch


class TCorex(TCorexBase):
    def __init__(self, l1=0.0, l2=0.0, reg_type='W', init=True, gamma=0.5,
                 max_sample_cnt=2 ** 30, weighted_obj=False, **kwargs):
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
        self.weighted_obj = weighted_obj
        self.window_len = None  # this depends on x and will be computed in fit()

        # define the weights of the model
        if (not self.init) and (self.pretrained_weights is None):
            self.ws = []
            for t in range(self.nt):
                wt = np.random.normal(loc=0, scale=1.0 / np.sqrt(self.nv), size=(self.m, self.nv))
                wt = torch.tensor(wt, dtype=torch.float, device=self.device, requires_grad=True)
                self.ws.append(wt)

    def forward(self, x_wno, anneal_eps, indices=None, return_factorization=False, return_R=False):
        # copy x_wno
        x_wno = [xt.copy() for xt in x_wno]
        # add annealing noise
        for t in range(self.nt):
            ns = x_wno[t].shape[0]
            anneal_noise = np.random.normal(size=(ns, self.nv))
            x_wno[t] = np.sqrt(1 - anneal_eps ** 2) * x_wno[t] + anneal_eps * anneal_noise
        x = [torch.tensor(xt, dtype=torch.float, device=self.device) for xt in x_wno]

        z = [None] * self.nt
        for t in range(self.nt):
            ns = x_wno[t].shape[0]
            z_noise = self.y_scale * torch.randn((ns, self.m), dtype=torch.float, device=self.device)
            z_mean = torch.mm(x[t], self.ws[t].t())
            z[t] = z_mean + z_noise

        epsilon = 1e-8
        objs = [None] * self.nt
        sigma = [None] * self.nt
        mi_xz = [None] * self.nt
        Rs = [None] * self.nt
        factorization = [None] * self.nt

        # store all concatenations here for better memory usage
        concats = dict()

        for t in range(self.nt):
            l = max(0, t - self.window_len[t])
            r = min(self.nt, t + self.window_len[t] + 1)
            weights = []
            left_t = t
            right_t = t
            for i in range(l, r):
                cur_ns = x_wno[i].shape[0]
                coef = np.power(self.gamma, np.abs(i - t))
                # skip if the importance is too low
                if coef < 1e-6:
                    continue
                left_t = min(left_t, i)
                right_t = max(right_t, i)
                weights.append(torch.tensor(coef * np.ones((cur_ns,)), dtype=torch.float, device=self.device))

            weights = torch.cat(weights, dim=0)
            weights = weights / torch.sum(weights)
            weights = weights.reshape((-1, 1))

            t_range = (left_t, right_t)
            if t_range in concats:
                x_all = concats[t_range]
            else:
                x_all = torch.cat(x[left_t:right_t+1], dim=0)
                concats[t_range] = x_all
            ns_tot = x_all.shape[0]

            z_all_mean = torch.mm(x_all, self.ws[t].t())
            z_all_noise = self.y_scale * torch.randn((ns_tot, self.m), dtype=torch.float, device=self.device)
            z_all = z_all_mean + z_all_noise
            z2_all = ((z_all ** 2) * weights).sum(dim=0)  # (m,)
            R_all = torch.mm((z_all * weights).t(), x_all)  # m, nv
            R_all = R_all / torch.sqrt(z2_all).reshape((self.m, 1))  # as <x^2_i> == 1 we don't divide by it

            z2 = z2_all
            R = R_all

            if return_R:
                Rs[t] = R

            if self.weighted_obj:
                X = x_all
                Z = z_all
            else:
                X = x[t]
                Z = z[t]

            if self.reg_type == 'MI':
                mi_xz[t] = -0.5 * torch.log1p(-torch.clamp(R ** 2, 0, 1 - epsilon))

            # the rest depends on R and z2 only
            ri = ((R ** 2) / torch.clamp(1 - R ** 2, epsilon, 1 - epsilon)).sum(dim=0)  # (nv,)

            # v_xi | z conditional mean
            outer_term = (1 / (1 + ri)).reshape((1, self.nv))
            inner_term_1 = R / torch.clamp(1 - R ** 2, epsilon, 1) / torch.sqrt(z2).reshape((self.m, 1))  # (m, nv)
            inner_term_2 = Z  # (ns, m)
            cond_mean = outer_term * torch.mm(inner_term_2, inner_term_1)  # (ns, nv)

            # calculate normed covariance matrix if needed
            need_sigma = (((indices is not None) and (t in indices)) or self.reg_type == 'Sigma')
            if need_sigma or return_factorization:
                inner_mat = 1.0 / (1 + ri).reshape((1, self.nv)) * R / torch.clamp(1 - R ** 2, epsilon, 1)
                factorization[t] = inner_mat
            if need_sigma:
                sigma[t] = torch.mm(inner_mat.t(), inner_mat)
                identity_matrix = torch.eye(self.nv, dtype=torch.float, device=self.device)
                sigma[t] = sigma[t] * (1 - identity_matrix) + identity_matrix

            # objective
            if self.weighted_obj:
                obj_part_1 = 0.5 * torch.log(
                    torch.clamp((((X - cond_mean) ** 2) * weights).sum(dim=0), epsilon, np.inf)).sum(dim=0)
            else:
                obj_part_1 = 0.5 * torch.log(torch.clamp(((X - cond_mean) ** 2).mean(dim=0), epsilon, np.inf)).sum(
                    dim=0)
            obj_part_2 = 0.5 * torch.log(z2).sum(dim=0)
            objs[t] = obj_part_1 + obj_part_2

        # experiments show that main_obj scales approximately linearly with nv
        # also it is a sum of over time steps, so we divide on (nt * nv)
        main_obj = 1.0 / (self.nt * self.nv) * sum(objs)

        # regularization
        reg_matrices = [None] * self.nt
        if self.reg_type == 'W':
            reg_matrices = self.ws
        if self.reg_type == 'Sigma':
            reg_matrices = sigma
        if self.reg_type == 'MI':
            reg_matrices = mi_xz

        reg_obj = torch.tensor(0.0, dtype=torch.float, device=self.device)

        # experiments show that L1 and L2 regularizations scale approximately linearly with nv
        if self.l1 > 0:
            l1_reg = sum([torch.abs(reg_matrices[t + 1] - reg_matrices[t]).sum() for t in range(self.nt - 1)])
            l1_reg = 1.0 / (self.nt * self.nv) * l1_reg
            reg_obj = reg_obj + self.l1 * l1_reg

        if self.l2 > 0:
            l2_reg = sum([((reg_matrices[t + 1] - reg_matrices[t]) ** 2).sum() for t in range(self.nt - 1)])
            l2_reg = 1.0 / (self.nt * self.nv) * l2_reg
            reg_obj = reg_obj + self.l2 * l2_reg

        total_obj = main_obj + reg_obj

        return {'total_obj': total_obj,
                'main_obj': main_obj,
                'reg_obj': reg_obj,
                'objs': objs,
                'sigma': sigma,
                'R': Rs,
                'factorization': factorization}

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
        self.x_input = x  # to have an access to input

        # initialize weights using the weighs of the linear CorEx trained on all data
        if self.init:
            if self.verbose > 0:
                print("Initializing with weights of a linear CorEx learned on whole data")
            init_start = time.time()
            lin_corex = Corex(nv=self.nv,
                              n_hidden=self.m,
                              max_iter=self.max_iter,
                              anneal=self.anneal,
                              verbose=self.verbose,
                              device=str(self.device),
                              gaussianize=self.gaussianize)
            # take maximum self.max_sample_cnt samples
            data_concat = np.concatenate(x, axis=0)
            if data_concat.shape[0] > self.max_sample_count:
                random.shuffle(data_concat)
                data_concat = data_concat[:self.max_sample_count]
            lin_corex.fit(data_concat)
            self.pretrained_weights = [lin_corex.get_weights()] * self.nt
            if self.verbose > 0:
                print("Initialization took {:.2f} seconds".format(time.time() - init_start))

        return self._train_loop()
