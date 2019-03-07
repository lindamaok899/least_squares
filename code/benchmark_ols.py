import numpy as np
import pandas as pd
from generate_data import generate_data
from timing import runtime
import matplotlib.pyplot as plt
import seaborn as sns
import os
from ols import matrix_inversion_np, lstsq_np, pseudo_inverse_np, solve_np
from ols import lls_with_blas, matrix_inversion_scipy, lstsq_scipy, solve_scipy
from ols import lu_solve_scipy, cholesky_np, qr_np
from sklearn.metrics import mean_squared_error as mse

sns.set_context("poster")
sns.set_palette("Paired", 11)
sns.set_color_codes()

np.random.seed(20000)

if not os.path.exists("../bld"):
    os.mkdir("../bld")


def time_one_function(data_dimensions, function):
    """Benchmarking code for various observation and variable sizes for one 
        ols implementation.
    Args:
        data_dimension(array): different observation sizes (nobs or variables).
        function (function): the ols implementation to be benchmarked.


    Returns: pd.DataFrame (len(dataset_sizes)) with dataset dimesions and timing
        for the different datasets.
    """

    output = []
    for nobs, nvariables in data_dimensions:
        x, y = generate_data(nobs=nobs, nexog=nvariables)
        # Start the functions with timer
        time_taken = runtime(function, args=(x, y), duration=0.5)["median_runtime"]
        output.append(time_taken)

    time_data = np.array(output).reshape(len(output), 1)

    all_data = np.hstack([data_dimensions, time_data])

    df = pd.DataFrame(data=all_data, columns=["nobs", "nvariables", function.__name__])

    df.set_index(["nobs", "nvariables"], inplace=True)
    return df


def batch_timing(func_list, data_dimensions):
    """Run a batch benchark for the ols implementations.
    Args:
        funct_list (list): List of ols implementations from ols.py.
        data_dimensions (list): List of nobs, nvariables for which the functions
            will be timed.
    Returns:
        runtime_data (pd.DataFrame): runtime data for all ols implementations.
    """
    batch_data = []
    for func in func_list:
        result = time_one_function(data_dimensions, func)
        batch_data.append(result)

    runtime_data = pd.concat(batch_data, axis=1)
    return runtime_data


def rmse_one_function(collinearities, function, nobs, nexog, num_iter):
    """Benchmark the accuracy of one ols implementation for different
        observations and variable sizes.
    Args:
        collinearities (array): ndarray of correlation between exogenous x variables 
            scaled between  0.2 and 1.
        function (function): the iv function for which accuracy is benchmarked.
        nobs (int): number of observations.
        nexog (int): number of exogenous x variables.
        num_iter (int): number of iterations.
    
    Returns: pd.DataFrame (len(dataset_dimensions)) with accuracy data based on
        the root mean squared error (rmse) between the true beta and the 
        estimated beta.
        
    """

    true_beta = np.ones(nexog)
    median_mse_list = []
    for col in collinearities:
        mse_list = []
        for i in range(num_iter):
            x, y = generate_data(nobs=nobs, nexog=nvariables, collinearity=col)
            estimated_beta = function(x, y)
            mse_list.append(mse(true_beta, estimated_beta))
        median_mse_list.append(np.median(mse_list))

    data = np.vstack([collinearities, median_mse_list]).T

    rmse_df = pd.DataFrame(
        data=data, columns=["collinearity_strength", function.__name__]
    )

    rmse_df.set_index(["collinearity_strength"], inplace=True)

    return rmse_df


def batch_rmse(collinearities, func_list, nobs, nexog, num_iter=20):
    """Benchmark the accuracy of all ols implementations using datasets with varying
        collinearity levels.
        
    Args:
        collinearities (array): ndarray of correlation between exogenous x variables 
            scaled between  0.2 and 1.
        func_list (list): List of ols implementations from ols.py. 
        nobs (int): number of observations.
        nexog (int): number of exogenous x variables.
        num_iter (int): number of iterations.
    
    Returns: pd.DataFrame (len(dataset_dimensions)) with all accuracy data based 
        on the root mean squared error (rmse) between the true beta and the
        estimated beta.

    """

    batch_rmse = []
    for func in func_list:
        output = rmse_one_function(collinearities, func, nobs, nexog, num_iter)
        batch_rmse.append(output)

    batch_rmse_data = pd.concat(batch_rmse, axis=1)
    return batch_rmse_data


def generate_plots(data, x_name, y_label):
    """Generates perfomance and accuracy plots for benchmark visualisation.
    Args:
        data (pd.DataFrame): the perfomance/accuracy data to be plotted.
        x_name (string): the name to be displayed on the x-axis
        y_label (string): the name to be displayed on the y-axis.
        
    Returns:
        fig (figure): the desired output in figure form.
    
    """
    function_names = [col for col in data.columns if col != x_name]

    fig, ax = plt.subplots()

    for funcname in function_names:
        sns.lineplot(x=x_name, y=funcname, data=data, label=funcname, ax=ax)

    ax.set_xlabel(x_name, fontsize="xx-small")
    ax.set_ylabel(y_label, fontsize="xx-small")
    plt.title("Ols Implementations", fontsize="x-small")
    plt.legend(
        loc="center left", bbox_to_anchor=(1, 0.5), frameon=True, fontsize="xx-small"
    )

    return fig


# ======================================================================================
# inputs
# ======================================================================================
nobs = 5000

nvariables = 10

nobs_list = (
    list(range(200, 2000, 200))
    + list(range(2000, 10000, 1000))
    + list(range(10000, 20000, 2000))
)

collinearities = np.arange(0.2, 0.99, 0.02)

nvariables_list = list(range(5, 55, 5))

data_dim_nobs = [(n, 30) for n in nobs_list]

data_dim_vars = [(5000, n) for n in nvariables_list]

func_list = [
    matrix_inversion_np,
    lstsq_np,
    pseudo_inverse_np,
    solve_np,
    lls_with_blas,
    matrix_inversion_scipy,
    lstsq_scipy,
    solve_scipy,
    lu_solve_scipy,
    cholesky_np,
    qr_np,
]

# ======================================================================================
# timing plots
# ======================================================================================
dim_list = [data_dim_nobs, data_dim_vars]
for dim in dim_list:
    plot_data = batch_timing(func_list=func_list, data_dimensions=dim)
    plot_data.reset_index(inplace=True)

    for col in ["nobs", "nvariables"]:
        if len(plot_data[col].unique()) == 1:
            reduced_data = plot_data.drop(col, axis=1)
        else:
            x_name = col

    fig = generate_plots(data=reduced_data, x_name=x_name, y_label="Runtime")

    plt.savefig("../bld/Perfomance_ols_{}.png".format(x_name), bbox_inches="tight")
    plt.close()

# ======================================================================================
# Accurracy plots
# ======================================================================================
rmse_plot_data = batch_rmse(
    collinearities=collinearities,
    func_list=func_list,
    nobs=nobs,
    nexog=nvariables,
    num_iter=10,
)
function_names = rmse_plot_data.columns
rmse_plot_data.reset_index(inplace=True)

x_name = "collinearity_strength"

fig = generate_plots(data=rmse_plot_data, x_name=x_name, y_label="Inaccuracy")
plt.savefig("../bld/Accuracy_ols.png".format(x_name), bbox_inches="tight")
plt.close()
