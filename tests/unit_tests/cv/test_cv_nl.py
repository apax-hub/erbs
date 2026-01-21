import numpy as np

from erbs.cv import compute_cv_nl


def test_compute_cv_nl():
    Z = np.array([1, 1, 2])
    Z_ref = np.array([2, 1, 2, 1])

    nl = compute_cv_nl(Z, Z_ref)

    assert nl.shape == (2, 6)
