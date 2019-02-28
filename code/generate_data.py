import numpy as np
import pandas as pd

def generate_data(nobs, nexog, nendog=0, ninstruments=0, collinearity=0.2, endogeneity=0.1, instr_strength=0.3, beta=None):
    """Generate data for ols or iv estimation.
    
    
    Args:
        nobs (int): number of observations
        nexog (int): number of exogenous x variables
        nendog (int, optional): number of endogenous x variables. Default 0.
        ninstruments (int, optional): number of pure instruments, i.e. instruments that are 
            not x variables. Default 0. Must be >= nendog.
        collinearity (float, optional): Highest correlation between exogenous x variables.
        endogeneity (float, optional): correlation between endegenous x variables and error term
        instr_strength (float, optional): correlation of instruments with endogenous x variables
        
    Returns:
        x (np.ndarray): array of shape (nobs, nexog + nendog) with explanatory variables
        y (np.ndarray): array of length (nobs) with dependent variable
        z (np.ndarray): array of shape (nobs, nexog + ninstruments). Only returned if
            ninstruments > 0
    
    """
    assert ninstruments >= nendog, (
        'You need at least as many instruments as endogenous x variables.')
    
    assert nexog >= 2, (
        'You need at least two exogenous variables')
    
    assert collinearity >= 0.2, (
        'The minimum level of collinearity is 0.2.')
    
    cov = _generate_cov_matrix(nexog, nendog, ninstruments, collinearity, endogeneity, instr_strength)
    means = _generate_means(nexog, nendog, ninstruments)
    
    if beta is None:
        beta = np.ones(nexog + nendog)
    
    
    raw_data = np.random.multivariate_normal(mean=means, cov=cov, size=nobs)
    
    exog_names, endog_names, instr_names = _variable_names(nexog, nendog, ninstruments)
    
    cols = exog_names + endog_names + instr_names + ['epsilon']
    
    df = pd.DataFrame(data=raw_data, columns=cols)
    x = df[exog_names + endog_names].values
    z = df[exog_names + instr_names].values
    epsilon = df['epsilon'].values
    
    y = np.dot(x, beta) + epsilon
    
    if ninstruments >= 1:
        return x, y, z
    else:
        return x, y
    
    
def _variable_names(nexog, nendog, ninstruments):
    exog_names = ['exog_{}'.format(i) for i in range(nexog)]
    endog_names = ['endog_{}'.format(i) for i in range(nendog)]
    instr_names = ['instr_{}'.format(i) for i in range(ninstruments)]
    return exog_names, endog_names, instr_names
    
    
def _generate_cov_matrix(nexog, nendog, ninstruments, collinearity, endogeneity, instr_strength):
    exog_names, endog_names, instr_names = _variable_names(nexog, nendog, ninstruments)
    cols = exog_names + endog_names + instr_names + ['epsilon']
    
    
    cov = np.zeros((len(cols), len(cols)))
    upper_indices = np.triu_indices(len(cols), k=1)
    nupper = len(upper_indices[0])
    cov[upper_indices] = np.random.uniform(low=0.0, high=0.1, size=nupper)
    #cov[upper_indices] = np.random.uniform(low=-0.1, high=0.1, size=nupper)
    cov_df = pd.DataFrame(data=cov, columns=cols, index=cols)
    cov_df.loc[exog_names + instr_names, 'epsilon'] = 0
    cov_df.loc[endog_names, instr_names] = instr_strength
    cov_df.loc[endog_names, 'epsilon'] = endogeneity
    cov_df.loc['exog_0', 'exog_1'] = collinearity
    
    cov = cov_df.values
    
    cov += cov.T
    cov[np.diag_indices(len(cols))] = 1
    return cov


def _generate_means(nexog, nendog, ninstruments):
    return np.zeros(nexog + nendog + ninstruments + 1)