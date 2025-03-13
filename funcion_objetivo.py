import numpy as np 
import pandas as pd

def nash_sqrt(array_obs, array_sim):

    numerador = np.nansum([np.square(np.sqrt(i) - np.sqrt(j)) for i,j in zip(array_obs,array_sim)])
    denominador = np.nansum([np.square(np.sqrt(i) - np.nanmean(np.sqrt(array_obs))) for i in array_obs])
    nash = np.around(1 - numerador/denominador,5)
    
    return nash
    
def nash_log(array_obs, array_sim):

    numerador = np.nansum([np.square(np.log(i) - np.log(j)) for i,j in zip(array_obs,array_sim)])
    denominador = np.nansum([np.square(np.log(i) - np.nanmean(np.log(array_obs))) for i in array_obs])
    nash = np.around(1 - numerador/denominador,5)
    return nash

def nash(array_obs, array_sim):

    numerador = np.nansum([np.square(i - j) for i,j in zip(array_obs,array_sim)])
    denominador = np.nansum([np.square(i - np.nanmean(array_obs)) for i in array_obs])
    nash = np.around( 1- numerador/denominador,5)

    return nash
