from scipy.stats import multivariate_normal
import numpy as np


def calculate_nll_score(data, covs):
    nt = len(data)
    nll = [-np.mean([multivariate_normal.logpdf(sx, cov=covs[t]) for sx in x])
           for x, t in zip(data, range(nt))]
    return np.mean(nll)
