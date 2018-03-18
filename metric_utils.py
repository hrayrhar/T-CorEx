from __future__ import absolute_import

from scipy.stats import multivariate_normal
import numpy as np


def calculate_nll_score(data, covs):
    nt = len(data)
    try:
        nll = [-np.mean([multivariate_normal.logpdf(sx, cov=covs[t]) for sx in x])
            for x, t in zip(data, range(nt))]
    except Exception:
        nll = [np.inf]
    return np.mean(nll)
