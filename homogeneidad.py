#%%
from cProfile import label
import pandas as pd
import numpy as np
import pyhomogeneity as ph
import pymannkendall as pm
import os
import matplotlib.pyplot as plt

# %%

def mann_kendall(df,var):

    # se agregan los datos para realizar el test
    if var == 'pptn':
        df_res = df.resample('YE').sum()

    else:
        df_res = df.resample('YE').mean()

    df_res_limpio =  df_res[df_res['valor']>0]
    data = np.array(df_res_limpio.iloc[:,0])

    results = pm.original_test(data)

    return results


def media_movil(datos, ventana):
    media_movil = []
    for i in range(len(datos) - ventana + 1):
        media = np.nanmean(datos[i:i+ventana])
        media_movil.append(media)
    return media_movil

def plot_media_movil(datos:np.array,ventana):
    
    media = media_movil(datos,ventana)
    
    plt.figure(figsize=(12,6))
    plt.plot(datos,color='blue',label='Serie')
    plt.plot(media,color='red',label='media movil')
    plt.legend()
    plt.grid()

def pettit(df,var):

    if var == 'pptn':
        df_res = df.resample('YE').sum()

    else:
        df_res = df.resample('YE').mean()

    col = df.columns[0]
    # Se aliminan los valores anuales que son 0
    df_res_limpio =  df_res[df_res[col]>0]

    data = np.array(df_res_limpio.iloc[:,0])

    result = ph.pettitt_test(data)

    return result

def outliers_desvest(df:pd.DataFrame,k=3.0):

    #Frecuencia de los datos
    idx = df.index
    freq = idx.inferred_freq
    # Vector de caudales
    col = df.columns[0]
    media = np.nanmean(df[col])
    std = np.nanstd(df[col])

    # Estimacion umbrales
    umbral_superior = media + k*std
    umbral_inferior = media - k*std
    print(umbral_superior,umbral_inferior)
    # Copia del dataframe para no sobreescribir el original

    # Datos acotados
    df_acotado = df[(df[col] < umbral_superior)]
    outliers = df[(df[col] > umbral_superior)]
    # Re arreglo de los datos
    inicio = df_acotado.index[0]
    fin = df_acotado.index[-1]
    new_idx = pd.date_range(inicio,fin,freq=freq)
    df_acotado_ = df_acotado.reindex(new_idx)
    
    
    return df_acotado_, outliers

def percentiles(df:pd.DataFrame):
    
    #Frecuencia de los datos
    idx = df.index
    freq = idx.inferred_freq
    
    #Vector de caudales
    col = df.columns[0]
    
    lim_superior = np.nanpercentile(df,99)
    lim_inferior = np.nanpercentile(df,1)
    
    # Datos acotados
    df_acotado = df[(df[col] < lim_superior)]
    outliers = df[(df[col] > lim_superior)]
    # Re arreglo de los datos
    inicio = df_acotado.index[0]
    fin = df_acotado.index[-1]
    new_idx = pd.date_range(inicio,fin,freq=freq)
    df_acotado_ = df_acotado.reindex(new_idx)
    
    return df_acotado_, outliers
    
    

def plot_pettit(df,var):
    
    # Columna con datos
    col = df.columns[0]
    # Arreglo de los datos
    if var == 'pptn':
        df_res = df.resample('YE').sum()
    else:
        df_res = df.resample('YE').mean()
    # Se eliminan anualidades con valores 0
    df_res = df_res[df_res[col]>0]
    min_valor = df_res[col].min()
    
    mn = df_res.index[0]
    mx = df_res.index[-1]
    
    resultados_pettit = pettit(df,var)
    
    change_point = resultados_pettit.cp
    loc = df_res.index[change_point]
    mu1 = resultados_pettit.avg.mu1
    mu2 = resultados_pettit.avg.mu2
    p_value = resultados_pettit.p
    
    plt.figure(figsize=(14, 7))
    plt.plot(df_res,color='blue')
    plt.hlines(mu1, xmin=mn, xmax=loc, 
               linestyles='--', colors='orange',
               lw=1.5, label='mu1 : ' + str(round(mu1,2)))
    plt.hlines(mu2, xmin=loc, xmax=mx, 
               linestyles='--', colors='g', lw=1.5, 
               label='mu2 : ' + str(round(mu2,2)))

    plt.axvline(x=loc, linestyle='-.' , color='red', 
                lw=1.5, label='Change point : '+ loc.strftime('%Y-%m-%d') + '\n p-value : ' + str(p_value))
    plt.ylim(min_valor)
    plt.grid()
 

    plt.xlabel('Años',fontsize=14)
    plt.ylabel('Precipitación[mm]',fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='upper right',fontsize=13)
    plt.tight_layout()
    # plt.savefig(path_guardado)
    # plt.close()







