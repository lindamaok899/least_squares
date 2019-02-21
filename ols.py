"""Define different implementations of OLS.

Each implementation returns the estimated parameter vector.

"""
import numpy as np


def naive_ols(x, y):
    xTx = x.T.dot(x)
    xTy = x.T.dot(y)
    beta = np.linalg.inv(xTx).dot(xTy)
    return beta
