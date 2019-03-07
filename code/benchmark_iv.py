import numpy as np
import pandas as pd
from timing import runtime
import matplotlib.pyplot as plt
import seaborn as sns
import os
from generate_data import generate_data
from iv import matrix_inversion_np, lstsq_np, pseudo_inverse_np, solve_np
from iv import lls_with_blas, matrix_inversion_scipy, lstsq_scipy, solve_scipy
from iv import lu_solve_scipy, cholesky_np, qr_np, weighting_matrix
from sklearn.metrics import mean_squared_error as mse

if not os.path.exists("../bld"):
    os.mkdir("../bld")

sns.set_context("poster")
sns.set_palette("Paired", 11)
sns.set_color_codes()

np.random.seed(9471)


def time_one_function(data_dimensions, function):
    """Benchmarking code for various observation and variable sizes as well as
    different number of instrumental variables for one iv implementation.
    Args:
        data_dimension(array): different observation sizes (nobs, variables, instruments).
        function (function): the iv implementation to be benchmarked.
    Returns: pd.DataFrame (len(dataset_sizes)) with dataset dimesions and timing
        for the different datasets.
    """

    output = []
    for nobs, nvariables, ninstruments in data_dimensions:
        x, y, z = generate_data(
            nobs=nobs,
            nexog=nvariables,
            nendog=ninstruments,
            instr_strength=instr_strength,
            ninstruments=ninstruments,
        )
        w = weighting_matrix(z)
        # Start the functions with timer
        time_taken = runtime(function, args=(x, y, z, w), duration=0.5)[
            "median_runtime"
        ]
        output.append(time_taken)

    time_data = np.array(output).reshape(len(output), 1)

    all_data = np.hstack([data_dimensions, time_data])

    df = pd.DataFrame(
        data=all_data, columns=["nobs", "nvariables", "ninstruments", function.__name__]
    )

    df.set_index(["nobs", "nvariables", "ninstruments"], inplace=True)

    return df


def batch_timing(func_list, data_dimensions):
    """Run a batch benchark for the iv implementations.
    Args:
        funct_list (list): List of iv implementations from iv.py.
        data_dimensions (list): List of nobs, nvariables, ninstruments for 
        which the functions will be timed.
    Returns:
        runtime_data (pd.DataFrame): runtime data for all iv implementations.
    """
    batch_data = []
    for func in func_list:
        result = time_one_function(data_dimensions, func)
        batch_data.append(result)

    runtime_data = pd.concat(batch_data, axis=1)
    return runtime_data


def rmse_one_function(accuracy_data, function, nobs, nexog, num_iter):
    """Benchmark the accuracy of one iv implementation for different
        observations, variable sizes as well as varied number of instruments.
    Args:
        accuracy_data (array): ndarray different accuracy measures
            (collinearity, ninstruments).
        function (function): the iv function for which accuracy is benchmarked.
        nobs (int): number of observations.
        nexog (int): number of exogenous x variables.
        num_iter (int): number of iterations.
    
    Returns: pd.DataFrame (len(dataset_dimensions)) with accuracy data based on
        the root mean squared error (rmse) between the true beta and the 
        estimated beta.
        
    """

    true_beta = np.ones(nexog + nendog)
    median_mse_list = []
    for collinearities, instruments in accuracy_data:
        mse_list = []
        for i in range(num_iter):
            x, y, z = generate_data(
                nobs=nobs,
                nexog=nvariables,
                nendog=ninstruments,
                instr_strength=instr_strength,
                ninstruments=instruments,
                collinearity=collinearities,
            )
            w = weighting_matrix(z)
            estimated_beta = function(x, y, z, w)
            mse_list.append(mse(true_beta, estimated_beta))
        median_mse_list.append(np.median(mse_list))
    
    median_data = np.array(median_mse_list).reshape(len(median_mse_list), 1)
    
    data = np.hstack([accuracy_data, median_data])

    rmse_df = pd.DataFrame(
        data=data, columns=["collinearity_strength", "instruments", function.__name__]
    )

    rmse_df.set_index(["collinearity_strength", "instruments"], inplace=True)

    return rmse_df


def batch_rmse(accuracy_data, func_list, nobs, nexog, num_iter=20):
    """Benchmark the accuracy of all iv implementations using datasets with varying
        collinearity levels.
        
    Args:
        accuracy_data (array): ndarray different accuracy measures
            (collinearity, ninstruments).
        func_list (list): List of iv implementations from iv.py. 
        nobs (int): number of observations.
        nexog (int): number of exogenous x variables.
        num_iter (int): number of iterations.
    
    Returns: pd.DataFrame (len(dataset_dimensions)) with all accuracy data based 
        on the root mean squared error (rmse) between the true beta and the
        estimated beta.

    """

    batch_rmse = []
    for func in func_list:
        output = rmse_one_function(accuracy_data, func, nobs, nexog, num_iter)
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
    plt.title("IV Implementations", fontsize="x-small")
    plt.legend(
        loc="center left", bbox_to_anchor=(1, 0.5), frameon=True, fontsize="xx-small"
    )

    return fig


# ======================================================================================
# Inputs
# ======================================================================================
nobs = 5000

nvariables = 10

nendog = 5

instr_strength = 0.8

# just identified case
ninstruments = 5

nexog = nvariables


nobs_list = (
    list(range(200, 2000, 200))
    + list(range(2000, 10000, 1000))
    + list(range(10000, 20000, 2000))
)

nvariables_list = list(range(5, 55, 5))

nistruments_list = list(range(5, 50, 5))

collinearities = list(np.arange(0.2, 0.8, 0.02))

data_dim_vars = [(5000, n, 10) for n in nvariables_list]

data_dim_nobs = [(n, 30, 10) for n in nobs_list]

data_dim_instr = [(3000, 30, n) for n in nistruments_list]

accuracy_collinearities = [(n, 5) for n in collinearities]

accuracy_nistruments = [(0.2, n) for n in nistruments_list]



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
dim_list = [data_dim_nobs, data_dim_vars, data_dim_instr]

for i, dim in enumerate(dim_list):
    plot_data = batch_timing(func_list=func_list, data_dimensions=dim)
    plot_data.reset_index(inplace=True)
    reduced_data = pd.concat([plot_data.iloc[:, i], plot_data.iloc[:, 3:]], axis=1)
    x_name = reduced_data.columns[0]
    fig = generate_plots(data=reduced_data, x_name=x_name, y_label="Runtime")

    plt.savefig("../bld/Perfomance_iv_{}.png".format(x_name), bbox_inches="tight")
    plt.close()


# ======================================================================================
# Accurracy plots
# ======================================================================================
accuracy_list = [accuracy_collinearities, accuracy_nistruments]

for accuracy in accuracy_list:
    plot_data = batch_rmse(
    accuracy_data=accuracy,
    func_list=func_list,
    nobs=nobs,
    nexog=nvariables,
    num_iter=10,
)
    plot_data.reset_index(inplace=True)

    for col in ["collinearity_strength", "instruments"]:
        if len(plot_data[col].unique()) == 1:
            reduced_data = plot_data.drop(col, axis=1)
        else:
            x_name = col
    
    fig = generate_plots(data=reduced_data, x_name=x_name, y_label="Inaccuracy")
    
    plt.savefig("../bld/Accuracy_iv_{}.png".format(x_name), bbox_inches="tight")
    plt.close()
