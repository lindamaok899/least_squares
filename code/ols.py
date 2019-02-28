import numpy as np
from numba import njit
from scipy.linalg.blas import dgemm
import scipy.linalg as sl


"""Define different implementations of OLS using numpy and numpy and scipy.

Each implementation returns the estimated parameter vector.

"""
def matrix_inversion_np(x, y):
    beta = np.linalg.inv(x.T.dot(x)).dot(x.T.dot(y))
    return beta


def lstsq_np(x, y):
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    return beta


def pseudo_inverse_np(x, y):
    beta = np.dot(np.linalg.pinv(x), y)
    return beta


def solve_np(x,y):
    beta = np.linalg.solve(np.dot(x.T, x), np.dot(x.T, y))
    return beta

def lls_with_blas(x, y, residuals=False):
    """
    https://gist.github.com/aldro61/5889795
    """
    i = dgemm(alpha=1.0, a=x.T, b=x.T, trans_b=True)
    beta = np.linalg.solve(i, dgemm(alpha=1.0, a=x.T, b=y)).flatten()
    return beta

def cholesky_np(x, y):
    l = np.linalg.cholesky(x.T.dot(x))
    c = forward_substitution(l, x.T.dot(y))
    beta = backward_substitution(l.T, c)
    return beta

def qr_np(x, y):
    q, r = np.linalg.qr(x)
    beta = np.linalg.inv(r).dot(q.T.dot(y))
    return beta

def matrix_inversion_scipy(x, y):
    beta = sl.inv(x.T.dot(x)).dot(x.T.dot(y))
    return beta


def lstsq_scipy(x, y):
    beta = np.linalg.lstsq(x, y)[0]
    return beta

#pseudo inverse implementation scipy was too slow and is not included in the plot

def solve_scipy(x,y):
    beta = sl.solve(np.dot(x.T, x), np.dot(x.T, y))
    return beta

def lu_solve_scipy(x, y):
    lu, piv = sl.lu_factor(x.T @ x)
    beta = sl.lu_solve((lu, piv), x.T @ y)
    return beta

#Helper functions for numpy cholesky decomposition
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
        _sum = np.sum(u[i, i+1:] * x[i+1:])
        x[i] = (y[i] - _sum) / u[i, i]

    return x



