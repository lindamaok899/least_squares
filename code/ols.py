import numpy as np
from numba import njit
from scipy.linalg.blas import dgemm
import scipy.linalg as sl
from sklearn.datasets import make_regression
import pandas as pd
from timing import core_timer, runtime
import matplotlib.pyplot as plt
import seaborn as sns
import time
#from matplotlib import inline
sns.set_context('poster')
sns.set_palette('Paired', 10)
sns.set_color_codes()

start = time.time()

print(start)


data_dim_nobs = [(300, 30), (600, 30)]
data_dim_vars = [(600, 20), (600, 30)]


#ols implementations
"""Define different implementations of OLS using numpy and numpy and scipy.

Each implementation returns the estimated parameter vector.

"""
#---------numpy implementations---------------
#mathematical implemantation numpy
#@njit
def matrix_inversion_np(x, y):
    beta = np.linalg.inv(x.T.dot(x)).dot(x.T.dot(y))
    return beta

# least squares implementation numpy
#@njit
def lstsq_np(x, y):
    beta = np.linalg.lstsq(x, y)[0]
    return beta



#benchmark functions
#function to generate a single dataset, switch coef on for true beta
def generate_data_ols(nobs, variables):
    x, y = make_regression(n_samples=nobs, n_features=variables, n_informative=variables,
                    effective_rank=None, noise=0.4, shuffle=True, coef=False, random_state=25)
    return x, y


#benchmarking one function
def benchmark_one_function(data_dimensions, function):
    """Benchmarking code for various observation sizes.
    Args:
        data_dimension(array): different observation sizes (nobs or variables).
        function (function): the ols implementation to be benchmarked.


    Returns: pd.Series (len(dataset_sizes)) with dataset dinesions and timing
        for the different datasets
    """
    print('\n\n', function.__name__)
    
    output = []
    for nobs, nvariables in data_dimensions:
        print(nobs, nvariables)
        x, y = generate_data_ols(nobs=nobs, variables=nvariables)
        # Start the functions with timer
        time_taken = runtime(function, args=(x, y), duration=0.005)['median_runtime']
        output.append(time_taken)

    time_data = np.array(output).reshape(len(output), 1)

    all_data = np.hstack([data_dimensions, time_data])

    df = pd.DataFrame(
        data=all_data,
        columns=['nobs', 'nvariables', function.__name__]
    )

    df.set_index(['nobs', 'nvariables'], inplace=True)
    return df


#func_list = [matrix_inversion_np, lstsq_np, pseudo_inverse_np, solve_np,
#            lls_with_blas, matrix_inversion_scipy, lstsq_scipy,
#            solve_scipy, lu_solve_scipy, cholesky_np, qr_np]

func_list = [lstsq_np]

#benchmark all functions
def batch_benchmark(func_list, data_dimensions):
    """Run a batch benchark for the ols implementations.
    Args:
        funct_list (list): List of ols implementations from ols.py
        data_dimensions (list): ...
    Returns:
        runtime_data (pd.DataFrame)
    """
    batch_data = []
    for func in func_list:
        result = benchmark_one_function(data_dimensions, func)
        batch_data.append(result)

    runtime_data = pd.concat(batch_data, axis=1)
    return runtime_data


dim_list = [data_dim_nobs, data_dim_vars]
for dim in dim_list:
    plot_data = batch_benchmark(func_list=func_list, data_dimensions=dim)
    function_names = plot_data.columns
    plot_data.reset_index(inplace=True)

    print()

    for col in ['nobs', 'nvariables']:
        if len(plot_data[col].unique()) == 1:
            reduced_data = plot_data.drop(col, axis=1)
        else:
            x_name = col

    fig, ax = plt.subplots()

    for k in [1]:
        for funcname in function_names:
            sns.regplot(
                x=x_name, y=funcname, data=reduced_data, order=k,
                label=funcname, fit_reg=True, ax=ax)
        ax.set_xlabel(x_name, fontsize='xx-small')
        ax.set_ylabel('Time taken', fontsize='xx-small')
        plt.title('Ols Implementations', fontsize='x-small')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, fontsize = 'xx-small')
        plt.savefig("Perfomance_ols_{}_{}.png".format(x_name, k), bbox_inches='tight')
        plt.close()


stop = time.time()

print(stop - start)