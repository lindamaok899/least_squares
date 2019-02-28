import numpy as np
from sklearn.metrics import mean_squared_error as mse

def median_rmse(true_beta, estimated_beta, function, args=(), duration=1.0):
    """
    
    """
    rmse = core_rmse(true_beta, estimated_beta, function, args)
    avg_rmse = np.median(rmse)
    return avg_rmse

def core_rmse(true_beta, estimated_beta, function, args=()):
    rmse = mse(true_beta, estimated_beta)
    return rmse

