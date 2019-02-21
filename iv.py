"""Define different implementations of IV.

Each implementation returns the estimated parameter vector.

"""
import numpy as np


def naive_iv(y, x, z, w):
    xTz = x.T.dot(z)
    helper = xTz.dot(w)
    inverse_part = np.linalg.inv(np.dot(helper, xTz.T))
    y_part = helper.dot(z.T.dot(y))
    beta = inverse_part.dot(y_part)
    return beta


def naive_weights(z):
    nobs, k_prime = z.shape
    w = np.linalg.inv(np.dot(z.T, z) / nobs)
    return w

