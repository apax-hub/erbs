import numpy as np


def compute_cv_nl(Z, Z_ref):
    idx_is = []
    idx_js = []
    for ii, Zi in enumerate(Z):
        idx_j = (Z_ref == Zi).nonzero()[0]
        idx_i = np.full(idx_j.shape, ii)
        idx_is.append(idx_i)
        idx_js.append(idx_j)

    idx_i = np.concatenate(idx_is, axis=0)
    idx_j = np.concatenate(idx_js, axis=0)
    nl = np.vstack([idx_i, idx_j])
    return nl
