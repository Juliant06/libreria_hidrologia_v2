from tkinter import font
from turtle import color
from networkx import density
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as opt
import scipy.stats as stats
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker
import matplotlib.cm as cm
class tormenta:

    def __init__(self,df:pd.DataFrame,estacion:str,path:str):
        self.df = df
        self.estacion = estacion
        self.path = path

    # Funciones de extraccion de datos
    def ajuste_potencial(self,x_data:np.array,y_data:np.array):

        def curva_potencial(x,a,b):
            return a*x**b
        
        # Ajuste de curva 
        param_opt, param_cov = opt.curve_fit(curva_potencial, 
                                             x_data, y_data)
        # Extraccion de parametros
        a_opt,b_opt = param_opt

        x_fit = np.linspace(min(x_data), max(x_data), 1000)
        y_fit = curva_potencial(x_fit, a_opt, b_opt)

        return (a_opt,b_opt, x_fit, y_fit)

    def ajuste_fdp(self,data,fn,):
        
        # Estimacion parameros
        params = fn.dist.fit(data)
        x = np.linspace(np.min(data), np.max(data), 1000)
        pdf = fn.pdf(x,*params)

        return (x,pdf)

    
    def filtro_mjo(self,) -> tuple:

        # Fase inactiva MJO cuando amplitud menor a 1
        fase_inactiva = self.df[self.df['amplitude'] < 1.0]
        
        fases_activas = list()
        
        for fase in range(1,9):

            df_fase = self.df[(self.df['phase'] == fase) & \
                              (self.df['amplitude'] > 1.0)]
            
            fases_activas.append(df_fase)
        
        return (fase_inactiva, fases_activas)
    
    def filtro_enso(self,) -> tuple:

        df_nino = self.df[self.df['enso_phase'] == 'niño']
        df_nina = self.df[self.df['enso_phase'] == 'niña']
        df_neutro = self.df[self.df['enso_phase'] == 'neutro']

        return (df_nino, df_nina, df_neutro)
    
    def analisis_frecuencias(self,df:pd.DataFrame,col:str):
        
        # Crea array con los datos de interes
        var = np.array(df[col])
        # Crea dataframe
        df_analisis = pd.DataFrame({'variable': var})
        # Estimacion excedencia
        df_sort = df_analisis.sort_values(by='variable',
                                 ascending=False).reset_index(drop=True)
        
        df_sort['Excedencia'] = (df_sort.index + 1)/(len(df_sort) + 1)*100

        q_10 = np.interp(1, df_sort['Excedencia'], df_sort['variable'])
        q_90 = np.interp(90, df_sort['Excedencia'], df_sort['variable'])

        return (q_10,q_90)
    
    def intensidad(self,):

        # Delimitacion dataframes
        df_nino,df_nina,df_neutro = self.filtro_enso()
        list_dfs = [df_nino,df_nina,df_neutro]
        
        list_resultados = list()
        for df in list_dfs:

            duraciones =  df['duracion'].unique()

            dic_resultados = {'duracion':duraciones,
                              'q_10':[],
                              'q_90':[]}
            
            for duracion in duraciones:

                df_duracion = df[df['duracion'] == duracion]
                q_10, q_90 = self.analisis_frecuencias(df_duracion,
                                                       'intensidad_media')
                dic_resultados['q_10'].append(q_10)
                dic_resultados['q_90'].append(q_90)
            
            list_resultados.append(dic_resultados)
        
        return list_resultados
    
    def intensidad_completo(self,):

        duraciones =  self.df['duracion'].unique()

        dic_resultados = {'duracion':duraciones,
                            'q_10':[],'q_90':[]}

        for duracion in duraciones:

                df_duracion = self.df[self.df['duracion'] == duracion]

                q_10, q_90 = self.analisis_frecuencias(df_duracion,
                                           'intensidad_media')

                dic_resultados['q_10'].append(q_10)
                dic_resultados['q_90'].append(q_90)
        
        return dic_resultados
    
    def intensidad_mjo(self,):
        
        fases = self.filtro_mjo()
        fase_inactiva = fases[0]
        fases_activas = fases[1]

        # Duracion del estado inactivo

        duracion_inactiva = fase_inactiva['duracion'].unique()

        lista_fases = list()
        dic_inactiva = {'duracion':duracion_inactiva,
                        'q_10':[],
                        'q_90':[]}

        # Extraccion datos fase inactiva
        for duracion in duracion_inactiva:

            df_duracion = fase_inactiva[fase_inactiva['duracion']==duracion]
            q_10,q_90 = self.analisis_frecuencias(df_duracion,'intensidad_media')
            dic_inactiva['q_10'].append(q_10)
            dic_inactiva['q_90'].append(q_90)
        
        lista_fases.append(dic_inactiva)

        # Extraccion datos fases activa
        # Itera sobre las fases de la MJO activas

        
        for fase in fases_activas:
            
            duraciones = fase['duracion'].unique()
            # Crea diccionario para almacenar las fases
            # Se crea un diccionario para cada fase

            dic_fase = {'duracion':duraciones,
                        'q_10':[],
                        'q_90':[]}

            for duracion in duraciones:

                df_duracion = fase[fase['duracion']==duracion]
                q_10,q_90 = self.analisis_frecuencias(df_duracion,'intensidad_media')

                # Se almacenan los resultados
                dic_fase['q_10'].append(q_10)
                dic_fase['q_90'].append(q_90)

            lista_fases.append(dic_fase)
        
        return lista_fases
        
        # Extraccion datos fase activa

    ### Funciones para graficar ###
    ###############################

    def plots(self,var:str,ylabel:str,
              path_guardado:str,
              xlabel='Duracion [min]'):

        fig, axd = plt.subplot_mosaic(
        [['top'],
         ['middle'],
         ['bottom']],
        figsize=(9, 9), layout="constrained" )
        
        for k, ax in axd.items():
            # annotate_axes(ax, f'axd[{k!r}]', fontsize=14)
            # Plot grande duracion vs variables
            if k == 'top':
                # Arrays de datos
                x_data = self.df['duracion'].values
                y_data = self.df[var].values
                # Ajuste
                a_opt,b_opt,x_fit,y_fit = self.ajuste_potencial(x_data,
                                                                y_data)
                sns.scatterplot(x=x_data, y=y_data, ax=axd['top'], 
                                color='white',edgecolor='black')
                # ax.plot(x_fit,y_fit,
                        # label=f"$y = {a_opt:.2f}x^{{{b_opt:.2f}}}$")

                ax.set_xlabel(xlabel,fontsize=10)
                # ax.set_ylabel(ylabel,fontsize=10)
                ax.grid()
                ax.legend()
            
            # Este plot para el enso
            elif k == 'middle':
                
                dfs_enso = self.filtro_enso()
                labels = ['Niño','Niña','Neutro']
                dict_ajuste = dict()

                for idx,df in enumerate(dfs_enso):

                    label = labels[idx]

                    x_data_enso = df['duracion']
                    y_data_enso = df[var]
                    dict_ajuste[label] = self.ajuste_potencial(x_data_enso,
                                                                    y_data_enso)
    

                palette = {"niña": "blue", 
                           "neutro": "gray", 
                           "niño": "red"}
                

                
                sns.scatterplot(data=self.df,x='duracion',y=var, 
                                hue='enso_phase',palette=palette,
                                edgecolor="black",ax = axd['middle'])
                

                # for fase_enso in labels:
                    
                    # # Valores de ajuste
                    # a_fase = dict_ajuste[fase_enso][0]
                    # b_fase = dict_ajuste[fase_enso][1]
                    # # variables
                    # x_fase = dict_ajuste[fase_enso][2]
                    # y_fase = dict_ajuste[fase_enso][3]
                    
                    # label_nino = f"{fase_enso}:$y = {a_fase:.2f}x^{{{b_fase:.2f}}}$"
                    # ax.plot(x_fase,y_fase,label=label_nino,
                            # color=palette[fase_enso.lower()],
                            # linestyle='--')
                
                ax.set_xlabel(xlabel,fontsize=10)
                ax.set_ylabel(ylabel,fontsize=10)
         
                ax.grid()
                ax.legend(loc='upper right')
                ax.set_xlim(0,20000)

            # Plots MJO
            else:
                fase_inactiva, fases_activa = self.filtro_mjo()

                x_data_mjo = fase_inactiva['duracion'].values
                y_data_mjo = fase_inactiva[var].values
                a_inac,b_inac, x_fit, y_fit = self.ajuste_potencial(x_data_mjo,
                                                                     y_data_mjo)

                sns.scatterplot(data=fase_inactiva, x='duracion', y=var,
                                color='black',ax=axd['bottom'],label='Fase 0')
                
                label = f"fase inactiva:$y = {a_inac:.2f}x^{{{b_inac:.2f}}}$"
                # ax.plot(x_fit,y_fit, label=label, color='black' )
                
                for i in range(1,9):
                    df_fase = fases_activa[i-1]
                    fase_mjo = df_fase['phase'].unique()[0]
                    x_data_mjo = df_fase['duracion'].values
                    y_data_mjo = df_fase[var].values
                    a_opt,b_opt, x_fit, y_fit = self.ajuste_potencial(x_data_mjo,                                               
                                                                    y_data_mjo)

                    sns.scatterplot(data=df_fase, x='duracion',y=var,s=30,
                                    edgecolor="black",ax = axd['bottom'],label=f'Fase {i}')
                    
                    # label = f"fase {fase_mjo}:$y = {a_opt:.2f}x^{{{b_opt:.2f}}}$"
                    # ax.plot(x_fit,y_fit,label=label)

                ax.legend(ncol=3,framealpha=0)

                ax.set_xlabel(xlabel,fontsize=10)
                ax.set_ylabel(ylabel,fontsize=10)
                ax.grid()
                ax.set_xlim(0,20000)
                # ax.plot(x,y2)
                # ax.grid()

        # fig.suptitle(f'Estación {self.estacion}',fontsize=17)
        plt.savefig(path_guardado)
        plt.close()

    def plot_intensidad_10(self,):

        # Datos full
        full = self.intensidad_completo()
        duracion_full = full['duracion']

        mean_duracion = np.mean(duracion_full)
        # std_duracion = np.std(duracion_full)
        # max_eje = std_duracion + mean_duracion
        duracion = duracion_full/mean_duracion

        q_10_full = full['q_10']
        a,b, x, y = self.ajuste_potencial(duracion,q_10_full)

        # Delimitacion datos ENSO
        nino, nina, neutro = self.intensidad()
        fases = [nino, nina, neutro]
        labels = ['Niño','Niña','Neutro']
        colors = ['red','blue','grey']

        # ciclo for
        label_ajuste = f'${a:.2f}x^{{{b:.2f}}}$'

        plt.figure(figsize=(12,4))
        plt.plot(x,y,label=label_ajuste,
                    color='black',linestyle='--')

        for idx,fase in enumerate(fases):

            duracion = fase['duracion']
            mean_duracion = np.mean(duracion)

            duracion = duracion/mean_duracion
     
            q_10 = fase['q_10']
            a_opt,b_opt, x_fit, y_fit = self.ajuste_potencial(duracion,q_10)     

        # Graficos

            temporalidad = labels[idx]
            color = colors[idx]
            label_ajuste = f'{temporalidad} ${a_opt:.2f}x^{{{b_opt:.2f}}}$'
            # print(color)
            plt.scatter(x=duracion,y=q_10,
                        label = temporalidad,color=color,alpha=0.5)
            plt.plot(x_fit,y_fit,color,
                     label=label_ajuste,linestyle='--')
    
        plt.xlim(0,0.25)
        plt.title(f'$I_{{99}}$ distintas duraciones',fontsize=15,weight='bold')
        plt.ylabel('Intensidad [mm/min]',fontsize=13)
        plt.xlabel('Duracion[min]/$\mu$[min]',fontsize=13)

        plt.grid()
        plt.legend()
        


    def plot_intensidad_90(self,):

        # Datos full
        full = self.intensidad_completo()

        duracion_full = full['duracion']
        mean_duracion = np.mean(duracion_full)
        std_duracion = np.std(duracion_full)
        max_eje = std_duracion + mean_duracion

        q_90_full = full['q_10']
        a,b, x, y = self.ajuste_potencial(duracion_full,
                                            q_90_full)

        # Delimitacion datos
        nino, nina, neutro = self.intensidad()
        fases = [nino, nina, neutro]
        labels = ['Niño','Niña','Neutro']
        colors = ['red','blue','grey']

        
        label_ajuste = f'${a:.2f}x^{{{b:.2f}}}$'

        plt.figure(figsize=(12,4))
        plt.plot(x,y,label=label_ajuste,
                    color='black',linestyle='--')
    
        # ciclo for fases
        for idx,fase in enumerate(fases):
            duracion = fase['duracion']
            q_90 = fase['q_90']
            a_opt,b_opt, x_fit, y_fit = self.ajuste_potencial(duracion,q_90)
    
        # Graficos

            temporalidad = labels[idx]
            color = colors[idx]
            label_ajuste = f'{temporalidad} ${a_opt:.2f}x^{{{b_opt:.2f}}}$'

            plt.scatter(x=duracion,y=q_90,
                        label = temporalidad,color=color,alpha=0.5)
            plt.plot(x_fit,y_fit,color,
                     label=label_ajuste,linestyle='--')
        plt.ylim(0,0.4)    
        plt.xlim(0,max_eje)
        plt.grid()
        plt.legend()

    def plot_intensidad_10_mjo(self,):

        # Datos full
        full = self.intensidad_completo()
        duracion_full = full['duracion']

        mean_duracion = np.mean(duracion_full)
        duracion = duracion_full/mean_duracion
        # max_eje = std_duracion + mean_duracion
        q_10_full = full['q_10']
        a, b, x, y = self.ajuste_potencial(duracion,q_10_full)

        # Delimitacion datos MJO
        # Regresa una lista de diccionarios con las fases
        fases = self.intensidad_mjo()

        # ciclo for
        label_ajuste = f'${a:.2f}x^{{{b:.2f}}}$'

        plt.figure(figsize=(12,4))
        plt.plot(x,y,label=label_ajuste,
                    color='black',linestyle='--')

        for idx,fase in enumerate(fases):

            duracion = fase['duracion']
            mean_duracion = np.mean(duracion)
            duracion_escalada = duracion/mean_duracion
     
            q_10 = fase['q_10']
            a_opt,b_opt, x_fit, y_fit = self.ajuste_potencial(duracion_escalada,q_10)
        # Graficos

            label_ajuste = f'${a_opt:.2f}x^{{{b_opt:.2f}}}$'

            plt.scatter(x=duracion_escalada ,y=q_10,
                        label = f'fase {idx}',alpha=0.5)
            plt.plot(x_fit,y_fit,
                     label=label_ajuste,linestyle='--',alpha=0.0)
        
        # plt.xlim()

        plt.title(f'$I_{{99}}$ distintas duraciones',fontsize=15,weight='bold')
        plt.ylabel('Intensidad [mm/min]',fontsize=13)
        plt.xlabel('Duracion[min]/$\mu$[min]',fontsize=13)
        plt.grid()
        plt.legend(loc='best',ncol=3,framealpha=0)



    
    def hist_plot_enso(self,):
        
        df_nino, df_nina, df_neutro = self.filtro_enso()


        # Datos de ejemplo
        data_full = np.log(self.df['duracion'])
        data_nino = np.log(df_nino['duracion'])
        data_nina = np.log(df_nina['duracion'])
        data_neutro = np.log(df_neutro['duracion'])

        # Crear figura y subplots
        fig, axes = plt.subplots(1, 4, figsize=(15, 5))  # 1 fila, 3 columnas

        # Configurar cada histograma

        axes[0].hist(data_full, bins=30,density=True ,
            color='white', alpha=0.7, edgecolor='black')
        axes[0].set_title('Datos')


        axes[1].hist(data_nino, bins=30,density=True ,
                    color='red', alpha=0.7, edgecolor='black')
        axes[1].set_title('Fase Niño')

        axes[2].hist(data_neutro, bins=30,density=True ,
            color='grey', alpha=0.7, edgecolor='black')
        axes[2].set_title('Fase Neutra')

        axes[3].hist(data_nina, bins=30,density=True ,
                    color='blue', alpha=0.7, edgecolor='black')
        axes[3].set_title('Fase Niña')

        # Etiquetas comunes
        for ax in axes:
            ax.set_xlabel('log(Duración)',fontsize=12)
            ax.set_ylabel('Densidad',fontsize=12)
            ax.grid()

        plt.tight_layout()

    def hist_plot_mjo(self,):

        data_full = np.log(self.df['duracion'])

        # Fases MJO:
        fases = self.filtro_mjo() #fase inactiva, fases activas
        
        fases_activas = fases[1]

        
        # Crear figura y subplots
        fig, axes = plt.subplots(2, 5, figsize=(20, 10))
        colors = cm.get_cmap('tab20', 10-1)
        # Configurar histogramas en cada subplot
        for i, ax in enumerate(axes.flatten()):

            if i-1 == -1:
                ax.hist(data_full, bins=30,density=True,
                        alpha=0.7, color='white',
                        edgecolor='black',)
                ax.set_title(f'Datos',fontsize =14,weight='bold')
            
            elif i-1 == 0:
                data = np.log(fases[i-1]['duracion'])
                ax.hist(data, bins=30,density=True,
                        alpha=0.7, color=colors(i),
                        edgecolor='black')
                ax.set_title(f'Fase {i-1}',fontsize =14,weight='bold')
            
            else:
                data = np.log(fases_activas[i-2]['duracion'])
                ax.hist(data, bins=30,density=True,
                        alpha=0.7, color=colors(i),
                        edgecolor='black')
                ax.set_title(f'Fase {i-1}',fontsize =14,weight='bold')

            # Configurar el eje x en escala logarítmic

            ax.set_xlabel('log(Duración)',fontsize=12)
            ax.set_ylabel('Densidad',fontsize=12)
            ax.tick_params(axis='x', labelsize=11)
            ax.grid(linestyle='--',color='black')

        plt.tight_layout()
        plt.show()

            
        
