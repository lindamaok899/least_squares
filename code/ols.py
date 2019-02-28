import numpy as np
from numba import njit
from scipy.linalg.blas import dgemm
import scipy.linalg as sl
#from sklearn.datasets import make_regression
import pandas as pd
from generate_data import generate_data
from timing import runtime
import matplotlib.pyplot as plt
import seaborn as sns
from rmse import median_rmse

sns.set_context('poster')
sns.set_palette('Paired', 11)
sns.set_color_codes()

data_dim_nobs = [(300, 30), (600, 30)]
data_dim_vars = [(600, 20), (600, 30)]

nobs_list = (
   list(range(200, 2000, 200))
   + list(range(2000, 10000, 1000))
   + list(range(10000, 20000, 2000)))

data_dim_nobs = [(nobs, 30) for nobs in nobs_list]

nvariables_list = (
   list(range(5, 50, 10))
   + list(range(50, 500, 50)))

data_dim_vars = [(5000, nvariables) for nvariables in nvariables_list]

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



#def generate_data_ols(nobs, variables):
#    x, y, true_beta = make_regression(n_samples=nobs, n_features=variables, n_informative=variables,
#                    effective_rank=None, noise=0.4, shuffle=True, coef=True, random_state=25)
#    return x, y, true_beta

def benchmark_one_function(data_dimensions, function):
    """Benchmarking code for various observation sizes.
    Args:
        data_dimension(array): different observation sizes (nobs or variables).
        function (function): the ols implementation to be benchmarked.


    Returns: pd.Series (len(dataset_sizes)) with dataset dinesions and timing
        for the different datasets
    """

    output = []
    for nobs, nvariables in data_dimensions:
        x, y = generate_data(nobs=nobs, nexog=nvariables)
        # Start the functions with timer
        time_taken = runtime(function, args=(x, y), duration=1.0)['median_runtime']
        output.append(time_taken)

    time_data = np.array(output).reshape(len(output), 1)

    all_data = np.hstack([data_dimensions, time_data])

    df = pd.DataFrame(
        data=all_data,
        columns=['nobs', 'nvariables', function.__name__]
    )

    df.set_index(['nobs', 'nvariables'], inplace=True)
    return df


func_list = [matrix_inversion_np, lstsq_np, pseudo_inverse_np, solve_np,
           lls_with_blas, matrix_inversion_scipy, lstsq_scipy,
           solve_scipy, lu_solve_scipy, cholesky_np, qr_np]


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



nobs = 20000
nvariables = 50
true_beta = np.ones(nvariables)
collinearity_strength = np.arange(0.2, 0.9999999, 0.05999999999).reshape(14,1)

def rmse_one_function(collinearity_strength, function):
    """

    """
    output = []
    for strength in collinearity_strength:
        x, y = generate_data(nobs=nobs, nexog=nvariables, collinearity=strength)
        estimated_beta = function(x, y)
        rsme_output = median_rmse(true_beta, estimated_beta, function, args=(x,y), duration=1.0)
        output.append(rsme_output)

    rmse_data = np.array(output).reshape(len(output), 1)
    all_data = np.hstack([collinearity_strength, rmse_data])

    rmse_df = pd.DataFrame(
        data=all_data,
        columns=['collinearity_strength', function.__name__]
    )

    rmse_df.set_index(['collinearity_strength'], inplace=True)

    return rmse_df

def batch_rmse(collinearity_strength, func_list):
    """

    """

    batch_rmse = []
    for func in func_list:
        output = rmse_one_function(collinearity_strength, func)
        batch_rmse.append(output)

    batch_rmse_data = pd.concat(batch_rmse, axis=1)
    return batch_rmse_data


coll_strength = collinearity_strength.tolist()


def performance_plot(data, x_name, y_label):
    function_names = [col for col in data.columns if col != x_name]

    fig, ax = plt.subplots()

    for funcname in function_names:
        sns.lineplot(
            x=x_name, y=funcname, data=data,
            label=funcname, ax=ax)

    ax.set_xlabel(x_name, fontsize='xx-small')
    ax.set_ylabel(y_label, fontsize='xx-small')
    plt.title('Ols Implementations', fontsize='x-small')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, fontsize = 'xx-small')

    return fig


dim_list = [data_dim_nobs, data_dim_vars]
for dim in dim_list:
    plot_data = batch_benchmark(func_list=func_list, data_dimensions=dim)
    plot_data.reset_index(inplace=True)

    for col in ['nobs', 'nvariables']:
        if len(plot_data[col].unique()) == 1:
            reduced_data = plot_data.drop(col, axis=1)
        else:
            x_name = col

    fig = performance_plot(data=reduced_data, x_name=x_name, y_label='Runtime')

    plt.savefig("Perfomance_ols_{}.png".format(x_name), bbox_inches='tight')
    plt.close()


rmse_plot_data = batch_rmse(collinearity_strength=collinearity_strength, func_list=func_list)
function_names = rmse_plot_data.columns
rmse_plot_data.reset_index(inplace=True)

x_name = 'collinearity_strength'

fig = performance_plot(data=rmse_plot_data, x_name = x_name, y_label='Inaccuracy')
plt.savefig("Accuracy_ols.png".format(x_name), bbox_inches='tight')
plt.close()