import numpy as np
#from numba import njit
#from scipy.linalg.blas import dgemm
#import scipy.linalg as sl
import pandas as pd
from generate_data import generate_data
from timing import runtime
import matplotlib.pyplot as plt
import seaborn as sns
from rmse import median_rmse
import os
from ols import matrix_inversion_np, lstsq_np, pseudo_inverse_np, solve_np
from ols import lls_with_blas, matrix_inversion_scipy, lstsq_scipy, solve_scipy
from ols import lu_solve_scipy, cholesky_np, qr_np

sns.set_context('poster')
sns.set_palette('Paired', 11)
sns.set_color_codes()

if not os.path.exists('../bld'):
   os.mkdir('../bld')


def time_one_function(data_dimensions, function):
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


def batch_timing(func_list, data_dimensions):
    """Run a batch benchark for the ols implementations.
    Args:
        funct_list (list): List of ols implementations from ols.py
        data_dimensions (list): ...
    Returns:
        runtime_data (pd.DataFrame)
    """
    batch_data = []
    for func in func_list:
        result = time_one_function(data_dimensions, func)
        batch_data.append(result)

    runtime_data = pd.concat(batch_data, axis=1)
    return runtime_data



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

#inputs
nobs = 20000
nvariables = 50
true_beta = np.ones(nvariables)
collinearity_strength = np.arange(0.2, 0.9999999, 0.05999999999).reshape(14,1)
coll_strength = collinearity_strength.tolist()
nobs_list = (
   list(range(200, 2000, 200))
   + list(range(2000, 10000, 1000))
   + list(range(10000, 20000, 2000)))

data_dim_nobs = [(nobs, 30) for nobs in nobs_list]

nvariables_list = (
   list(range(5, 50, 10))
   + list(range(50, 500, 50)))

data_dim_vars = [(5000, nvariables) for nvariables in nvariables_list]

func_list = [matrix_inversion_np, lstsq_np, pseudo_inverse_np, solve_np,
           lls_with_blas, matrix_inversion_scipy, lstsq_scipy,
           solve_scipy, lu_solve_scipy, cholesky_np, qr_np]

#plots
dim_list = [data_dim_nobs, data_dim_vars]
for dim in dim_list:
    plot_data = batch_timing(func_list=func_list, data_dimensions=dim)
    plot_data.reset_index(inplace=True)

    for col in ['nobs', 'nvariables']:
        if len(plot_data[col].unique()) == 1:
            reduced_data = plot_data.drop(col, axis=1)
        else:
            x_name = col

    fig = performance_plot(data=reduced_data, x_name=x_name, y_label='Runtime')

    plt.savefig("../bld/Perfomance_ols_{}.png".format(x_name), bbox_inches='tight')
    plt.close()


rmse_plot_data = batch_rmse(collinearity_strength=collinearity_strength, func_list=func_list)
function_names = rmse_plot_data.columns
rmse_plot_data.reset_index(inplace=True)

x_name = 'collinearity_strength'

fig = performance_plot(data=rmse_plot_data, x_name = x_name, y_label='Inaccuracy')
plt.savefig("../bld/Accuracy_ols.png".format(x_name), bbox_inches='tight')
plt.close()

