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

def balance(array_obs:np.array, array_sim:np.array) -> float:
    mean_sim = np.nanmean(array_sim)
    mean_obs = np.nanmean(array_obs)

    balance = 100*np.abs(mean_obs-mean_sim)/mean_obs
    
    return balance

def minimos(array_obs:np.array, array_sim:np.array) -> float:
    q05_sim = np.nanquantile(array_sim,0.05)
    q05_obs = np.nanquantile(array_obs,0.05)

    q_05 = 100*np.abs(q05_obs-q05_sim)/q05_obs    
    return q_05
    
def rmse(array_obs:np.array, array_sim:np.array) -> float:
    
    pass
    