from __future__ import absolute_import
from __future__ import print_function
from scipy.stats import multivariate_normal
import numpy as np


def plot_cov_matrix(plt, cov, title=None, vmin=-1, vmax=+1):
    plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.imshow(cov, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.show()


def plot_for_next_timestep(plt, data, covs, title="Negative log-likelihood of estimate of time step $t$ under "
                                                  "the test data of timestep $t + 1$"):
    nt = len(data)
    nll = [-np.mean([multivariate_normal.logpdf(sx, cov=covs[t]) for sx in x])
           for x, t in zip(data[1:], range(nt - 1))]
    plt.bar(range(1, nt), nll, width=0.6)
    plt.title(title)
    plt.xlabel("Timestep")
    plt.ylim(0)
    plt.ylabel("Negative log-likelihood")
    plt.xticks(range(1, nt))
    plt.show()
    print("NLL for next time step = {}".format(np.mean(nll)))
    return np.mean(nll)
