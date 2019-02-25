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

startt = time.time()

#benchmark algorithm for different dataset sizes
def benchmark_algorithm_datasets(dataset_obs, functions):
    """Benchmarking code for various observation sizes.
    Args:
        dataset_sizes(array): different observation sizes (nobs,).
        functions (function): the ols implementation to be benchmarked.
        function_args(numpy arrays): The arguments with which the function is
            called.

    Returns: pd.DataFrame (len(dataset_sizes, 2)) with dataset sizes and timing
        for the different dataset sizes
    """

    output = []

    for size in dataset_obs:
        # Use sklearns make_regression to generate a random dataset with specified
        x, y = generate_data_ols(nobs=size)
        # Start the functions with timer
        time_taken = runtime(functions, args=(x, y), duration=0.5)['median_runtime']
        output.append(time_taken)
    time_data = np.asarray(output).reshape(len(dataset_obs))

    return pd.DataFrame(np.vstack(([dataset_obs, time_data])).T, columns=['x','y'])

#benchmark algorithm for different number of variables
def benchmark_algorithm_variables(dataset_vars, functions):
    """Benchmarking code for various variable numbers.
    Args:
        dataset_vars(array): different observation sizes (variable,).
        functions (function): the ols implementation to be benchmarked.
        function_args(numpy arrays): The arguments with which the function is
            called.

    Returns: pd.DataFrame (len(dataset_vars, 2)) with dataset sizes and timing
        for the different dataset sizes
    """

    output = []

    for size in dataset_vars:
        # Use sklearns make_regression to generate a random dataset with specified
        x, y = generate_data_ols(variables=size)
        # Start the functions with timer
        time_taken = runtime(functions, args=(x, y), duration=0.5)['median_runtime']
        output.append(time_taken)
    time_data = np.asarray(output).reshape(len(dataset_vars))

    return pd.DataFrame(np.vstack(([dataset_vars, time_data])).T, columns=['x','y'])

#getting datasets for plots
dataset_a = benchmark_algorithm_datasets(dataset_obs, matrix_inversion_np)
dataset_b = benchmark_algorithm_datasets(dataset_obs, lstsq_np)
dataset_c = benchmark_algorithm_datasets(dataset_obs, pseudo_inverse_np)
dataset_d = benchmark_algorithm_datasets(dataset_obs, solve_np)
dataset_e = benchmark_algorithm_datasets(dataset_obs, lls_with_blas)
dataset_f = benchmark_algorithm_datasets(dataset_obs, matrix_inversion_spicy)
dataset_g = benchmark_algorithm_datasets(dataset_obs, lstsq_spicy)
#dataset_h = benchmark_algorithm_datasets(dataset_obs, pseudo_inverse_spicy)
dataset_i = benchmark_algorithm_datasets(dataset_obs, solve_spicy)
dataset_j = benchmark_algorithm_datasets(dataset_obs, lu_solve_spicy)
dataset_k = benchmark_algorithm_datasets(dataset_obs, cholesky_np)
dataset_l = benchmark_algorithm_datasets(dataset_obs, qr_np)

#getting datasets for plots
dataset_aa = benchmark_algorithm_variables(dataset_vars, matrix_inversion_np)
dataset_bb = benchmark_algorithm_variables(dataset_vars, lstsq_np)
dataset_cc = benchmark_algorithm_variables(dataset_vars, pseudo_inverse_np)
dataset_dd = benchmark_algorithm_variables(dataset_vars, solve_np)
dataset_ee = benchmark_algorithm_variables(dataset_vars, lls_with_blas)
dataset_ff = benchmark_algorithm_variables(dataset_vars, matrix_inversion_spicy)
dataset_gg = benchmark_algorithm_variables(dataset_vars, lstsq_spicy)
#dataset_h = benchmark_algorithm_datasets(dataset_obs, pseudo_inverse_spicy)
dataset_ii = benchmark_algorithm_variables(dataset_vars, solve_spicy)
dataset_jj = benchmark_algorithm_variables(dataset_vars, lu_solve_spicy)
dataset_kk = benchmark_algorithm_variables(dataset_vars, cholesky_np)
dataset_ll = benchmark_algorithm_variables(dataset_vars, qr_np)

#plotting the perfomance comparison for different number of observations
for k in range (2):
    sns.regplot(x='x', y='y', data=dataset_a, order=k,
                label='matrix_inversion_np', dropna=False, fit_reg=True)
    sns.regplot(x='x', y='y', data=dataset_b, order=k,
                label='lstsq_np', dropna=False, fit_reg=True)
    sns.regplot(x='x', y='y', data=dataset_c, order=k,
                label='pseudo_inverse_np', dropna=False, fit_reg=True)
    sns.regplot(x='x', y='y', data=dataset_d, order=k,
                label='solve_np', dropna=False, fit_reg=True)
    sns.regplot(x='x', y='y', data=dataset_e, order=k,
                label='lls_with_blas', dropna=False, fit_reg=True)
    sns.regplot(x='x', y='y', data=dataset_f, order=k,
                label='matrix_inversion_spicy', dropna=False, fit_reg=True)
    sns.regplot(x='x', y='y', data=dataset_g, order=k,
                label='lstsq_spicy', dropna=False, fit_reg=True)
#    sns.regplot(x='x', y='y', data=dataset_h, order=k,
#                label='pseudo_inverse_spicy', dropna=False)
    sns.regplot(x='x', y='y', data=dataset_i, order=k,
                label='solve_spicy', dropna=False, fit_reg=True)
    sns.regplot(x='x', y='y', data=dataset_j, order=k,
                label='lu_solve_spicy', dropna=False, fit_reg=True)
    sns.regplot(x='x', y='y', data=dataset_k, order=k,
                label='cholesky_np', dropna=False, fit_reg=True)
    sns.regplot(x='x', y='y', data=dataset_l, order=k,
                label='qr_np', dropna=True, fit_reg=True)
    plt.gca().axis([300, 70000, -0.005, 0.15], fontsize='xx-small')
    plt.gca().set_xlabel('nobs', fontsize = 'xx-small')
    plt.gca().set_ylabel('Time taken', fontsize = 'xx-small')
    plt.title('Ols Implementations', fontsize = 'x-small')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, fontsize = 'xx-small')
    #plt.figure(figsize=(1,1))
    plt.savefig("Perfomance_ols_obs.png", bbox_inches='tight')
    plt.show()
    plt.clf()

#plotting the perfomance comparison for different number of observations
for k in range (2):
    sns.regplot(x='x', y='y', data=dataset_aa, order=k,
                label='matrix_inversion_np', dropna=False, fit_reg=True)
    sns.regplot(x='x', y='y', data=dataset_bb, order=k,
                label='lstsq_np', dropna=False, fit_reg=True)
    sns.regplot(x='x', y='y', data=dataset_cc, order=k,
                label='pseudo_inverse_np', dropna=False, fit_reg=True)
    sns.regplot(x='x', y='y', data=dataset_dd, order=k,
                label='solve_np', dropna=False, fit_reg=True)
    sns.regplot(x='x', y='y', data=dataset_ee, order=k,
                label='lls_with_blas', dropna=False, fit_reg=True)
    sns.regplot(x='x', y='y', data=dataset_ff, order=k,
                label='matrix_inversion_spicy', dropna=False, fit_reg=True)
    sns.regplot(x='x', y='y', data=dataset_gg, order=k,
                label='lstsq_spicy', dropna=False, fit_reg=True)
#    sns.regplot(x='x', y='y', data=dataset_h, order=k,
#                label='pseudo_inverse_spicy', dropna=False)
    sns.regplot(x='x', y='y', data=dataset_ii, order=k,
                label='solve_spicy', dropna=False, fit_reg=True)
    sns.regplot(x='x', y='y', data=dataset_jj, order=k,
                label='lu_solve_spicy', dropna=False, fit_reg=True)
    sns.regplot(x='x', y='y', data=dataset_kk, order=k,
                label='cholesky_np', dropna=False, fit_reg=True)
    sns.regplot(x='x', y='y', data=dataset_ll, order=k,
                label='qr_np', dropna=True, fit_reg=True)
    plt.gca().axis([300, 500, 0, 0.15], fontsize='xx-small')
    plt.gca().set_xlabel('number of variables', fontsize = 'xx-small')
    plt.gca().set_ylabel('Time taken', fontsize = 'xx-small')
    plt.title('Ols Implementations', fontsize = 'x-small')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, fontsize = 'xx-small')
    #plt.figure(figsize=(1,1))
    plt.savefig("Perfomance_ols_vars.png", bbox_inches='tight')
    plt.show()
    plt.clf()

