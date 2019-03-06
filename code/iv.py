import numpy as np
from numba import njit
from scipy.linalg.blas import dgemm
import scipy.linalg as sl

# from generate_data import generate_data

"""Define different implementations of IV.

Each implementation returns the estimated parameter vector.

"""
# inputs
nobs = 5000
nvariables = 10
# x, y, z = generate_data(nobs=nobs, nexog=nvariables, nendog=2, ninstruments=5)


def weighting_matrix(z):
    nobs, k_prime = z.shape
    w = np.linalg.inv(np.dot(z.T, z) / nobs)
    return w


def matrix_inversion_np(x, y, z, w):
    xTz = x.T.dot(z)
    helper = xTz.dot(w)
    inverse_part = np.linalg.inv(np.dot(helper, xTz.T))
    y_part = helper.dot(z.T.dot(y))
    beta = inverse_part.dot(y_part)
    return beta


def lstsq_np(x, y, z, w):
    xTz = x.T.dot(z)
    helper = xTz.dot(w)
    inverse_part = np.dot(helper, xTz.T)
    y_part = helper.dot(z.T.dot(y))
    beta = np.linalg.lstsq(inverse_part, y_part, rcond=None)[0]
    return beta


def pseudo_inverse_np(x, y, z, w):
    xTz = x.T.dot(z)
    helper = xTz.dot(w)
    inverse_part = np.dot(helper, xTz.T)
    y_part = helper.dot(z.T.dot(y))
    beta = np.dot(np.linalg.pinv(inverse_part), y_part)
    return beta


def solve_np(x, y, z, w):
    xTz = x.T.dot(z)
    helper = xTz.dot(w)
    inverse_part = np.dot(helper, xTz.T)
    y_part = helper.dot(z.T.dot(y))
    beta = np.linalg.solve(inverse_part, y_part)
    return beta


def lls_with_blas(x, y, z, w, residuals=False):
    """
    https://gist.github.com/aldro61/5889795
    """
    xTz = x.T.dot(z)
    helper = xTz.dot(w)
    inverse_part = np.dot(helper, xTz.T)
    y_part = helper.dot(z.T.dot(y))
    i = dgemm(alpha=1.0, a=inverse_part, b=inverse_part, trans_b=True)
    beta = np.linalg.solve(i, dgemm(alpha=1.0, a=inverse_part, b=y_part)).flatten()
    return beta


def cholesky_np(x, y, z, w):
    xTz = x.T.dot(z)
    helper = xTz.dot(w)
    inverse_part = np.dot(helper, xTz.T)
    y_part = helper.dot(z.T.dot(y))
    l = np.linalg.cholesky(inverse_part)
    c = forward_substitution(l, y_part)
    beta = backward_substitution(l.T, c)
    return beta


def qr_np(x, y, z, w):
    xTz = x.T.dot(z)
    helper = xTz.dot(w)
    inverse_part = np.dot(helper, xTz.T)
    y_part = helper.dot(z.T.dot(y))
    q, r = np.linalg.qr(inverse_part)
    beta = np.linalg.inv(r).dot(q.T.dot(y_part))
    return beta


def matrix_inversion_scipy(x, y, z, w):
    xTz = x.T.dot(z)
    helper = xTz.dot(w)
    inverse_part = np.dot(helper, xTz.T)
    y_part = helper.dot(z.T.dot(y))
    beta = sl.inv(inverse_part).dot(y_part)
    return beta


def lstsq_scipy(x, y, z, w):
    xTz = x.T.dot(z)
    helper = xTz.dot(w)
    inverse_part = np.dot(helper, xTz.T)
    y_part = helper.dot(z.T.dot(y))
    beta = sl.lstsq(inverse_part, y_part)[0]
    return beta


# pseudo inverse implementation scipy was too slow and is not included in the plot


def solve_scipy(x, y, z, w):
    xTz = x.T.dot(z)
    helper = xTz.dot(w)
    inverse_part = np.dot(helper, xTz.T)
    y_part = helper.dot(z.T.dot(y))
    beta = sl.solve(inverse_part, y_part)
    return beta


def lu_solve_scipy(x, y, z, w):
    xTz = x.T.dot(z)
    helper = xTz.dot(w)
    inverse_part = np.dot(helper, xTz.T)
    y_part = helper.dot(z.T.dot(y))
    lu, piv = sl.lu_factor(inverse_part @ inverse_part)
    beta = sl.lu_solve((lu, piv), y_part @ inverse_part)
    return beta


# Helper functions for numpy cholesky decomposition
@njit
def forward_substitution(l, b):
    """Solves Ly=b.

    L has to be a lower triangular matrix. It is not required that the diagonal
    has only elements of 1.

    References
    ----------
    - https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/INT-APP/CURVE-linear-system.html

    """
    y = np.zeros(b.shape[0])
    y[0] = b[0] / l[0, 0]
    for i in range(1, b.shape[0]):
        _sum = np.sum(l[i, :i] * y[:i])
        y[i] = (b[i] - _sum) / l[i, i]
    return y


@njit
def backward_substitution(u, y):
    """Solves Ux=y.

    References
    ----------
    - https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/INT-APP/CURVE-linear-system.html

    """
    x = np.zeros(y.shape[0])
    x[-1] = y[-1] / u[-1, -1]
    for i in range(y.shape[0] - 2, -1, -1):
        _sum = np.sum(u[i, i + 1 :] * x[i + 1 :])
        x[i] = (y[i] - _sum) / u[i, i]

    return x
