import numpy as np
from numba import njit
from sklearn.datasets import make_regression
import pandas as pd
from scipy.linalg.blas import dgemm
import scipy.linalg as sl


#2
#generating data for ols
#np.random.seed(25)


#def generate_data_ols(nobs, variables):
    
#    means = np.ones(variables)
#    cov = np.eye(variables) + np.ones((variables, variables)) * 0.2
#    x = np.random.multivariate_normal(mean=means, cov=cov, size=nobs)
#    e = np.random.randn(nobs)
#    true_beta = np.random.uniform(-0.99, 0.99, variables)
#    y = x @ true_beta + e

#    return x, y


#function to generate a single dataset
variable = 50
nobs = 20000
def generate_data_ols(nobs, variables):
    x, y, coef = make_regression(n_samples=nobs, n_features=variables, n_informative=variables, 
                    effective_rank=None, noise=0.4, shuffle=True, coef=True, random_state=25)
    return x, y, coef
     
# generate multiple datasets with different nobs and constant variables
    x_obs_data = []
    y_obs_data = []
    coefs_obs_data = []
    
    for k in range(300,20000, 600):
        x, y, coef = generate_data_ols(k, variable)
        x_obs_data.append(x)
        y_obs_data.append(y)
        coefs_obs_data.append(coef)


# generate multiple datasets with constant nobs and different variables
    x_vars_data = []
    y_vars_data = []
    coefs_vars_data = []
    
    for k in range(5,50, 10):
        x, y, coef = generate_data_ols(nobs, k)
        x_vars_data.append(x)
        y_vars_data.append(y)
        coefs_vars_data.append(coef)

"""Define different implementations of OLS using numpy and numpy and spicy.

Each implementation returns the estimated parameter vector.

"""
#---------numpy implementations---------------
#mathematical implemantation numpy
@njit
def matrix_inversion_np(x, y):
    beta = np.linalg.inv(x.T.dot(x)).dot(x.T.dot(y))
    return beta

# least squares implementation numpy
@njit
def lstsq_np(x, y):
    beta = np.linalg.lstsq(x, y)[0]
    return beta

#pseudo inverse implementation numpy
@njit
def pseudo_inverse_np(x, y):
    beta = np.dot(np.linalg.pinv(x), y)
    return beta

# Solve implementation with numpy.
@njit
def solve_np(x,y):
    beta = np.linalg.solve(np.dot(x.T, x), np.dot(x.T, y))
    return beta

def lls_with_blas(x, y, residuals=False):
    """
    https://gist.github.com/aldro61/5889795
    """
    #a, b, coef = generate_data_ols(nobs, variables) - a, b are x, y
    i = dgemm(alpha=1.0, a=a.T, b=a.T, trans_b=True)
    beta = np.linalg.solve(i, dgemm(alpha=1.0, a=a.T, b=b)).flatten()
    return beta

#---------spicy implementations----------------
#matrix multiplication implementation spicy
#@njit
def matrix_inversion_spicy(x, y):
    beta = sl.inv(x.T.dot(x)).dot(x.T.dot(y))
    return beta

# least squares implementation spicy
@njit
def lstsq_spicy(x, y):
    beta = np.linalg.lstsq(x, y)[0]    
    return beta

#pseudo inverse implementation spicy
@njit
def pseudo_inverse_spicy(x, y):
    beta = np.dot(sl.pinv(x), y)
    return beta

# Solve implementation with scipy
#@njit
def solve_spicy(x,y):
    beta = sl.solve(np.dot(x.T, x), np.dot(x.T, y))
    return beta

