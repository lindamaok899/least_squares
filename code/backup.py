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

#inputs
dataset_obs = np.hstack([np.arange(1, 6) * 300,
                           np.arange(3,7) * 800, np.arange(4,17) * 1600])
dataset_vars = np.arange(5,50,10)
dataset_obs = np.arange(300,700,300)
dataset_vars = np.arange(30, 90,40)
dataset_obs = np.arange(300,700,150)
dataset_vars = np.arange(30, 90, 20)
data_dimensions = np.vstack((dataset_obs.T, dataset_vars.T))
nobs = 20000
variables = 50


#@njit
def matrix_inversion_np(x, y):
    beta = np.linalg.inv(x.T.dot(x)).dot(x.T.dot(y))
    return beta

# least squares implementation numpy
#@njit
def lstsq_np(x, y):
    beta = np.linalg.lstsq(x, y)[0]
    return beta

#pseudo inverse implementation numpy
#@njit
def pseudo_inverse_np(x, y):
    beta = np.dot(np.linalg.pinv(x), y)
    return beta

# Solve implementation with numpy.
#@njit
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

#Cholesky decomposition with numpy
#@njit
def cholesky_np(x, y):
    l = np.linalg.cholesky(x.T.dot(x))
    c = forward_substitution(l, x.T.dot(y))
    beta = backward_substitution(l.T, c)
    return beta

#@njit
def qr_np(x, y):
    q, r = np.linalg.qr(x)
    beta = np.linalg.inv(r).dot(q.T.dot(y))
    return beta

#---------scipy implementations----------------
#matrix multiplication implementation scipy
#@njit
def matrix_inversion_scipy(x, y):
    beta = sl.inv(x.T.dot(x)).dot(x.T.dot(y))
    return beta

# least squares implementation scipy
#@njit
def lstsq_scipy(x, y):
    beta = np.linalg.lstsq(x, y)[0]
    return beta

#pseudo inverse implementation scipy was too slow and is not included in the plot
#@njit
#def pseudo_inverse_scipy(x, y):
#    beta = np.dot(sl.pinv(x), y)
#    return beta

# Solve implementation with scipy
#@njit
def solve_scipy(x,y):
    beta = sl.solve(np.dot(x.T, x), np.dot(x.T, y))
    return beta

# LU decomposition with scipy
def lu_solve_scipy(x, y):
    lu, piv = sl.lu_factor(x.T @ x)
    beta = sl.lu_solve((lu, piv), x.T @ y)
    return beta

#Helper functions for numpy cholesky decomposition
#@njit
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


#@njit
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

    output = []
    #for i in range (0,1):
    for size in data_dimensions[0]:
        # Use sklearns make_regression to generate a random dataset with specified
        #x, y = generate_data_ols()
        x, y = make_regression(n_samples=size, n_features=variables, n_informative=variables,
                effective_rank=None, noise=0.4, shuffle=True, coef=False, random_state=25)
        # Start the functions with timer
        time_taken = runtime(function, args=(x, y), duration=0.016)['median_runtime']
        output.append(time_taken)
    time_data = np.asarray(output).reshape(len(data_dimensions.T))
    
    #return pd.DataFrame(np.vstack(([dataset_obs, time_data])).T, columns=['x','y'])
    return pd.Series(data=time_data, index=data_dimensions, name=function.__name__)


#func_list = [matrix_inversion_np, lstsq_np, pseudo_inverse_np, solve_np,
#            lls_with_blas, matrix_inversion_scipy, lstsq_scipy,
#            solve_scipy, lu_solve_scipy, cholesky_np, qr_np]

func_list = [matrix_inversion_np, lstsq_np]

#benchmark all functions
def batch_benchmark(func_list, data_dimensions):
    """Run a batch benchark for the ols implementations.
    Args:
        funct_list (list): List of ols implementations from ols.py
        benchmark_func (function): benchmark function to perform the batch
            benchmarknig

    Returns:
        batch_time_data(dict): dictionary with timings for the different
        observation sizes for all the ols implementations.
    """
    batch_data = []
    for func in func_list:
        #for i in range(0,1):
        result = benchmark_one_function(data_dimensions, func)
        batch_data.append(result)
        runtime_data = pd.concat(batch_data, axis=1)
    return runtime_data

#plotting the perfomance comparison for different number of observations
plot_data = batch_benchmark(func_list, data_dimensions)
function_names = plot_data.columns
plot_data.reset_index(inplace=True)

for funcname in function_names:
    for k in range (2):
        sns.regplot(x='x', y=funcname, data=plot_data, order=k,
                label=funcname.__name__, fit_reg=True)
    plt.gca().axis([300, 500, 0, 0.15], fontsize='xx-small')
    plt.gca().set_xlabel('number of variables', fontsize = 'xx-small')
    plt.gca().set_ylabel('Time taken', fontsize = 'xx-small')
    plt.title('Ols Implementations', fontsize = 'x-small')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, fontsize = 'xx-small')
    #plt.figure(figsize=(1,1))
    plt.savefig("Perfomance_ols_vars.png", bbox_inches='tight')
plt.show()
plt.clf()


    
    
    
stopp = time.time()

print(stopp - startt)    

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
dataset_f = benchmark_algorithm_datasets(dataset_obs, matrix_inversion_scipy)
dataset_g = benchmark_algorithm_datasets(dataset_obs, lstsq_scipy)
#dataset_h = benchmark_algorithm_datasets(dataset_obs, pseudo_inverse_scipy)
dataset_i = benchmark_algorithm_datasets(dataset_obs, solve_scipy)
dataset_j = benchmark_algorithm_datasets(dataset_obs, lu_solve_scipy)
dataset_k = benchmark_algorithm_datasets(dataset_obs, cholesky_np)
dataset_l = benchmark_algorithm_datasets(dataset_obs, qr_np)

#getting datasets for plots
dataset_aa = benchmark_algorithm_variables(dataset_vars, matrix_inversion_np)
dataset_bb = benchmark_algorithm_variables(dataset_vars, lstsq_np)
dataset_cc = benchmark_algorithm_variables(dataset_vars, pseudo_inverse_np)
dataset_dd = benchmark_algorithm_variables(dataset_vars, solve_np)
dataset_ee = benchmark_algorithm_variables(dataset_vars, lls_with_blas)
dataset_ff = benchmark_algorithm_variables(dataset_vars, matrix_inversion_scipy)
dataset_gg = benchmark_algorithm_variables(dataset_vars, lstsq_scipy)
#dataset_h = benchmark_algorithm_datasets(dataset_obs, pseudo_inverse_scipy)
dataset_ii = benchmark_algorithm_variables(dataset_vars, solve_scipy)
dataset_jj = benchmark_algorithm_variables(dataset_vars, lu_solve_scipy)
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
                label='matrix_inversion_scipy', dropna=False, fit_reg=True)
    sns.regplot(x='x', y='y', data=dataset_g, order=k,
                label='lstsq_scipy', dropna=False, fit_reg=True)
#    sns.regplot(x='x', y='y', data=dataset_h, order=k,
#                label='pseudo_inverse_scipy', dropna=False)
    sns.regplot(x='x', y='y', data=dataset_i, order=k,
                label='solve_scipy', dropna=False, fit_reg=True)
    sns.regplot(x='x', y='y', data=dataset_j, order=k,
                label='lu_solve_scipy', dropna=False, fit_reg=True)
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
                label='matrix_inversion_scipy', dropna=False, fit_reg=True)
    sns.regplot(x='x', y='y', data=dataset_gg, order=k,
                label='lstsq_scipy', dropna=False, fit_reg=True)
#    sns.regplot(x='x', y='y', data=dataset_h, order=k,
#                label='pseudo_inverse_scipy', dropna=False)
    sns.regplot(x='x', y='y', data=dataset_ii, order=k,
                label='solve_scipy', dropna=False, fit_reg=True)
    sns.regplot(x='x', y='y', data=dataset_jj, order=k,
                label='lu_solve_scipy', dropna=False, fit_reg=True)
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

