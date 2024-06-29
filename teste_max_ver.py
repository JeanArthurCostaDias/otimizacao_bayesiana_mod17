# Import necessary libraries
from datetime import datetime
from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import warnings
import arviz as az
from sklearn.metrics import mean_squared_error

# Ignore warnings
warnings.filterwarnings('ignore')

# Define a linear constraint function for PyMC3
def linear_constraint(xmin, xmax, form: str = None):
    '''
    Returns a linear ramp function, for deriving a value on [0, 1] from
    an input value `x`:

        if x >= xmax:
            return 1
        if x <= xmin:
            return 0
        return (x - xmin) / (xmax - xmin)

    Parameters
    ----------
    xmin : int or float
        Lower bound of the linear ramp function
    xmax : int or float
        Upper bound of the linear ramp function
    form : str
        Type of ramp function: "reversed" decreases as x increases;
        "binary" returns xmax when x == 1; default (None) is increasing
        as x increases.

    Returns
    -------
    function
    '''
    assert form is None or form in ('reversed', 'binary'),\
        'Argument "form" must be None or one of: "reversed", "binary"'

    if form == 'reversed':
        return lambda x: pm.math.switch(x >= xmax, 0,
            pm.math.switch(x < xmin, 1, 1 - (x - xmin) / (xmax - xmin)))
    if form == 'binary':
        return lambda x: pm.math.switch(x == 1, xmax, xmin)
    return lambda x: pm.math.switch(x >= xmax, 1,
        pm.math.switch(x < xmin, 0, (x - xmin) / (xmax - xmin)))

# Define a linear constraint function for NumPy
def linear_constraint_normal(xmin, xmax, form: str = None):
    '''
    Returns a linear ramp function, for deriving a value on [0, 1] from
    an input value x:

        if x >= xmax:
            return 1
        if x <= xmin:
            return 0
        return (x - xmin) / (xmax - xmin)

    Parameters
    ----------
    xmin : int or float
        Lower bound of the linear ramp function
    xmax : int or float
        Upper bound of the linear ramp function
    form : str
        Type of ramp function: "reversed" decreases as x increases;
        "binary" returns xmax when x == 1; default (None) is increasing
        as x increases.

    Returns
    -------
    function
    '''
    assert form is None or form in ('reversed', 'binary'),\
        'Argument "form" must be None or one of: "reversed", "binary"'
    assert form == 'binary' or np.any(xmax >= xmin),\
        'xmax must be greater than/ equal to xmin'
    if form == 'reversed':
        return lambda x: np.where(x >= xmax, 0,
            np.where(x < xmin, 1, 1 - np.divide(
                np.subtract(x, xmin), xmax - xmin)))
    if form == 'binary':
        return lambda x: np.where(x == 1, xmax, xmin)
    return lambda x: np.where(x >= xmax, 1,
        np.where(x < xmin, 0,
            np.divide(np.subtract(x, xmin), xmax - xmin)))



def gpp_calc(params, fpar, tmin, vpd, par):
    'Daily GPP as static method, avoids overhead of class instantiation'
    # "params" argument should be a Sequence of atomic parameter values
    #   in the order prescribed by "required_parameters"
    tmin_scalar = linear_constraint(params[1], params[2])(tmin)
    vpd_scalar = linear_constraint(
        params[3], params[4], form = 'reversed')(vpd)
    lue = params[0] * tmin_scalar * vpd_scalar
    return 1e3 * lue * fpar * par

# Function to calculate GPP using normal linear constraints
def gpp_calc_normal(params, fpar, tmin, vpd, par):
    '''
    Daily GPP as static method, avoids overhead of class instantiation

    Parameters
    ----------
    params : list
        List of parameter values for the GPP calculation.
    fpar : numpy array
        Fraction of absorbed photosynthetically active radiation (fPAR).
    tmin : numpy array
        Minimum daily temperature.
    vpd : numpy array
        Vapor pressure deficit.
    par : numpy array
        Photosynthetically active radiation (PAR).

    Returns
    -------
    numpy array
        Calculated GPP values.
    '''
    tmin_scalar = linear_constraint_normal(params[1], params[2])(tmin)
    vpd_scalar = linear_constraint_normal(params[3], params[4], form='reversed')(vpd)
    lue = params[0] * tmin_scalar * vpd_scalar
    return 1e3 * lue * fpar * par

# Load drivers data from file
drivers_santarem = np.load('drivers_santarem.npy')

# Parameters for GPP calculation
parametros = [0.001405, -8.0, 9.09, 1000.0, 4000.0, 26.9, 2.0, 2.0, 1.1, 0.162, 0.00604, 0.00519, 0.00397]

# Calculate GPP using normal linear constraints
gpp = gpp_calc_normal(parametros, drivers_santarem[0], drivers_santarem[1], drivers_santarem[2], drivers_santarem[3])

# Create date index for Santarem
index_santarem = pd.date_range(start='2002-01-01', end='2006-12-31', freq='D')
index_santarem = index_santarem.union(pd.date_range(start='2008-01-01', end='2011-12-31', freq='D'))

# Create DataFrame for GPP
gpp_santarem = pd.DataFrame(gpp, index=index_santarem)
gpp_santarem = gpp_santarem.rename_axis('index')
gpp_santarem = gpp_santarem[0].rename('GPP')

# Load observed data
santarem = pd.read_csv('observacoes_santarem_diario.csv', index_col=0, parse_dates=True, date_format='%Y-%m-%d')

# Filter data for the years 2002-2006
gpp_santarem = gpp_santarem['2002':'2006']
santarem = santarem['2002':'2006']

# Observed data (fictitious example)
observed_data = santarem.values

# Define the PyMC3 model
with pm.Model() as model:
    # Priors for the parameters
    tminmin = pm.Uniform('tminmin', lower=None, upper=8)
    tminmax = pm.Uniform('tminmax', lower=9, upper=None)
    vpdmin = pm.Uniform('vpdmin', lower=None, upper=30000)
    vpdmax = pm.Uniform('vpdmax', lower=1000, upper=None)
    epsilonj = pm.Uniform('epsilonj', lower=None, upper=1)

    pm.Potential('tminmin_constraint', pm.math.switch(tminmin >= tminmax, -np.inf, 0))
    pm.Potential('vpdmin_constraint', pm.math.switch(vpdmin >= vpdmax, -np.inf, 0))
    
    params = [epsilonj, tminmin, tminmax, vpdmin, vpdmax]

    # Predictions
    predictions = gpp_calc(params, drivers_santarem[0][:len(gpp_santarem.index)],
                           drivers_santarem[1][:len(gpp_santarem.index)],
                           drivers_santarem[2][:len(gpp_santarem.index)],
                           drivers_santarem[3][:len(gpp_santarem.index)])

    # Calculate RMSE
    rmse = pm.math.sqrt(pm.math.sum((predictions - observed_data) ** 2) / predictions.shape[0])

    # Likelihood (assuming normal distribution for the error)
    likelihood = pm.Normal('likelihood', mu=predictions, sigma=rmse, observed=observed_data)

    # Sampling using NUTS (No-U-Turn Sampler)
    trace = pm.sample(5000, tune=1000, return_inferencedata=True)

    # Find the Maximum A Posteriori (MAP) estimate
    map_estimate = pm.find_MAP()

# Analysis of results
print("MAP estimate for tminmin:", map_estimate['tminmin'])
print("MAP estimate for tminmax:", map_estimate['tminmax'])
print("MAP estimate for vpdmin:", map_estimate['vpdmin'])
print("MAP estimate for vpdmax:", map_estimate['vpdmax'])
print("MAP estimate for epsilonj:", map_estimate['epsilonj'])

# Save trace to NetCDF file
trace.to_netcdf('trace.nc')

# Calculate and print RMSE for the initial parameters
before = [0.001405, -8.0, 9.09, 1000.0, 4000.0, 26.9, 2.0, 2.0, 1.1, 0.162, 0.00604, 0.00519, 0.00397]
gpp_before = gpp_calc_normal(before, drivers_santarem[0][:len(gpp_santarem.index)],
                             drivers_santarem[1][:len(gpp_santarem.index)],
                             drivers_santarem[2][:len(gpp_santarem.index)],
                             drivers_santarem[3][:len(gpp_santarem.index)])
print("RMSE for initial parameters:", mean_squared_error(observed_data, gpp_before, squared=False))

# Calculate and print RMSE for the updated parameters
new_params = [map_estimate['epsilonj'], map_estimate['tminmin'], map_estimate['tminmax'],
              map_estimate['vpdmin'], map_estimate['vpdmax']]
new_gpp = gpp_calc_normal(new_params, drivers_santarem[0][:len(gpp_santarem.index)],
                          drivers_santarem[1][:len(gpp_santarem.index)],
                          drivers_santarem[2][:len(gpp_santarem.index)],
                          drivers_santarem[3][:len(gpp_santarem.index)])
print("RMSE for updated parameters:", mean_squared_error(observed_data, new_gpp, squared=False))
