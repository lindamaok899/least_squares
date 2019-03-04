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

sns.set_context("poster")
sns.set_palette("Paired", 11)
sns.set_color_codes()

#inputs
nobs = 5000
nendog=2
instr_strength = 0.99999
ninstruments = 10
nvariables = 10
collinearities = np.arange(0.2, 0.99, 0.02)


def time_one_function(data_dimensions, function):
    """Benchmarking code for various observation and variable sizes for one 
        ols implementation.
    Args:
        data_dimension(array): different observation sizes (nobs, variables, instruments).
        function (function): the ols implementation to be benchmarked.


    Returns: pd.DataFrame (len(dataset_sizes)) with dataset dimesions and timing
        for the different datasets.
    """

    output = []
    for nobs, nvariables, ninstruments in data_dimensions:
        x, y, z = generate_data(nobs=nobs, nexog=nvariables, instr_strength=instr_strength,
                                ninstruments=ninstruments)
        w = weighting_matrix(z)
        # Start the functions with timer
        time_taken = runtime(function, args=(x, y, z, w), duration=0.5)["median_runtime"]
        output.append(time_taken)

    time_data = np.array(output).reshape(len(output), 1)

    all_data = np.hstack([data_dimensions, time_data])

    df = pd.DataFrame(data=all_data, columns=["nobs", "nvariables", "ninstruments", function.__name__])

    df.set_index(["nobs", "nvariables", "ninstruments"], inplace=True)
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
# timing plots
# ======================================================================================
np.random.seed(5471)

nobs_list = (
    list(range(200, 2000, 200))
    + list(range(2000, 10000, 1000))
    + list(range(10000, 20000, 2000))
)

nvariables_list = list(range(5, 55, 5))

nistruments_list = list(range(10, 35, 3))

data_dim_vars = [(5000, n, 10) for n in nvariables_list]

data_dim_nobs = [(n, 30, 10) for n in nobs_list]

data_dim_instr = [(3000, 30, n) for n in nistruments_list]


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

 
dim_list = [data_dim_nobs, data_dim_vars, data_dim_instr]
for dim in dim_list:
    plot_data = batch_timing(func_list=func_list, data_dimensions=dim)
    plot_data.reset_index(inplace=True)
    #fix this part
    for col in ["nobs", "nvariables", "ninstruments"]:
        if len(plot_data[col].unique()) == 1:
            reduced_data = plot_data.drop(col, axis=1)
        else:
            x_name = col

    fig = generate_plots(data=reduced_data, x_name=x_name, y_label="Runtime")

    plt.savefig("../bld/Perfomance_iv_{}.png".format(x_name), bbox_inches="tight")
    plt.close()
    
test = len(plot_data[col].unique())