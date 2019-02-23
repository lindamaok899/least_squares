import numpy as np
from numba import njit
from sklearn.datasets import make_regression
import pandas as pd
from scipy.linalg.blas import dgemm
import scipy.linalg as sl
from timing import runtime, core_timer, find_good_unit

#Inputs
dataset_obs = np.hstack([np.arange(1, 6) * 300,
                           np.arange(3,7) * 800, np.arange(4,17) * 1600])
dataset_vars = np.arange(5,50,5)


#function to generate a single dataset, switch coef on for true beta
def generate_data_ols(nobs=20000, variable=50):
    x, y = make_regression(n_samples=nobs, n_features=variable, n_informative=variable, 
                    effective_rank=None, noise=0.4, shuffle=True, coef=False, random_state=25)
    return x, y

#benchmark algorithm for different dataset sizes
def benchmark_algorithm_datasets(dataset_obs, ols_function):
    """Benchmarking code at various datasets.
    Args:
        dataset_sizes(array): different observation sizes (nobs,).
        ols_function (function): the ols implementation to be benchmarked.
        function_args(numpy arrays): The arguments with which the function is 
            called.
        
    Returns: pd.DataFrame (len(dataset_sizes, 2)) with dataset sizes and timing
        for the different dataset sizes
    """
    
    output = []

    for index, size in enumerate(dataset_obs):
        # Use sklearns make_regression to generate a random dataset with specified
        x, y = generate_data_ols()
        # Start the functions with timer
        time_taken = core_timer(ols_function, args=(x, y))
        output.append(time_taken)
    time_data = np.asarray(output).reshape(len(dataset_obs))
    
    return pd.DataFrame(np.vstack([dataset_obs, time_data])).T

#benchmark algorithm for different number of variables
def benchmark_algorithm_variables(dataset_vars, ols_function):
    """Benchmarking code at various datasets.
    Args:
        dataset_vars(array): different observation sizes (variable,).
        ols_function (function): the ols implementation to be benchmarked.
        function_args(numpy arrays): The arguments with which the function is 
            called.
        
    Returns: pd.DataFrame (len(dataset_vars, 2)) with dataset sizes and timing
        for the different dataset sizes
    """
    
    output = []

    for index, size in enumerate(dataset_vars):
        # Use sklearns make_regression to generate a random dataset with specified
        x, y = generate_data_ols()
        # Start the functions with timer
        time_taken = core_timer(pseudo_inverse_np, args=(x, y))
        output.append(time_taken)
    time_data = np.asarray(output).reshape(len(dataset_vars))
    
    return pd.DataFrame(np.vstack([dataset_vars, time_data])).T


#Task 3
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

def lls_with_blas(x, y, nobs=20000, variable=50, residuals=False):
    """
    https://gist.github.com/aldro61/5889795
    """
    #a, b, coef = generate_data_ols(nobs, variables) - a, b are x, y
    
    a, b, coef = make_regression(n_samples=nobs, n_features=variable, n_informative=variable, 
                    effective_rank=None, noise=0.4, shuffle=True, coef=True, random_state=25)
    
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
#@njit
def lstsq_spicy(x, y):
    beta = np.linalg.lstsq(x, y)[0]    
    return beta

#pseudo inverse implementation spicy was too slow and is not included in the plot
@njit
def pseudo_inverse_spicy(x, y):
    beta = np.dot(sl.pinv(x), y)
    return beta

# Solve implementation with scipy
#@njit
def solve_spicy(x,y):
    beta = sl.solve(np.dot(x.T, x), np.dot(x.T, y))
    return beta     








