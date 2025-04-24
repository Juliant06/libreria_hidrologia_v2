import pandas as pd
import numpy as np
import re
import geopandas as gpd
import xarray as xr
import scipy.stats as stats
import matplotlib.pyplot as plt

# llenado de estaciones
def llenar_na(df):
    # Seleccion de fechas
    fecha_inicio = df.index[0]
    fecha_final = df.index[-1]
    # Generacion lista con fecha
    rango_fechas = pd.date_range(fecha_inicio,fecha_final,freq='d')
    # re-index de datos 
    df = df.reindex(rango_fechas)

    return df

# Leer archivos data
# Funcion para leer archivos de IDEAM en data

def read_data(archivo):

    # Lectura de los datos
    df = pd.read_csv(archivo,
                     sep='|',
                     index_col='Fecha')
    
    # Formato timestamp para el indice
    df.index = pd.to_datetime(df.index)
    # Cambio de nombre en la columna
    # Se nombre la columna con el codigo de la estacion
    pattern = r'@(\d+)' # Patron de regex para la extraccion del codigo

    match = re.search(pattern,archivo)[1]
    # Se aplica el cambio de nombre
    df.rename(columns = {'Valor':match}, 
                       inplace = True)
    
    #Se rellena el dataframe
    df_lleno = llenar_na(df) 
    
    return df_lleno

def pptn_media_anual(df,
                     umbral=0.1):
    
    # Se rellena el dataframe con datos NA
    df_lleno = llenar_na(df)

    #Se agregan los datos faltantes de manera anual
    df_faltantes = df_lleno.isna().resample('YE').sum()/365
    # Se agregan los datos de manera anual 
    df_anual = df_lleno.resample('YE').sum()
    # Filtro de datos
    datos_anuales = df_anual[df_faltantes < umbral].dropna()
    # Estimacion de la media anual
    media_anual = np.mean(datos_anuales)
    
    return media_anual

def razon_normal(df):

    cols = list(df.columns)

    Ns = [pptn_media_anual(df[col]) for col in cols]
    dic_media_pptn = {k:v for k,v in zip(cols,Ns)}

    for col in cols:
        # Crea una copia de la lista de columnas 
        # se actualiza cada vez que se corre el codigo
        col2 = list(df.columns)
        col2.remove(col)
        for date in df.index:
            Nx = dic_media_pptn[col]
            #Lista de columnas a revisar
            # Lista de valores medios mulianuales
            Ns = [dic_media_pptn[col] for col in col2]
            # Verifica si el punto esta vacio
            if pd.isna(df.at[date,col]):
                # Esta linea extrae los valores de precipitacion de las otras columnas
                Pi = [df.at[date,col] for col in col2]
                # Se realiza un calculo parcial del metodo de razon normal
                parte_1 = [(Nx/Ni)*Pi for Ni,Pi in zip(Ns,Pi)]
                # Se realiza el relleno de la precipitacion, se ignoran los Na en otras columnas
                Px = np.nanmean(parte_1)
                # Se actualiza el valor de precipitacion
                df.at[date,col] = Px
    
    return df

def info_estacion(df:pd.DataFrame,codigo:int) -> dict:

    # Este codigo es temporal 
    # Se busca una menera mas eficiente
    # De acceder a la información almacenada 
    # En el catálogo de estaciones de IDEAM

    # Se convierte el codigo de la variable a entero
    # Es la forma en la cual el CNE lo lee
    codigo = int(codigo)

    cne = pd.read_csv('Cat_logo_Nacional_de_Estaciones_del_IDEAM_20240627.csv',
                      index_col='Codigo')
    # Extracción de coordenadas en texto plano
    cne_index = cne.index
    # Chequeo si el código está en el CNE
    if codigo not in cne_index:
        raise Exception("Código no encontrado en el registro del catálogo")
    coord_cne = cne.at[codigo, 'Ubicación']

    # Patron de regex para la extracción de las coordenadas
    patron = r'-?\d+\.\d+'
    coordenadas = re.findall(patron,coord_cne) 

    #Almacenamiento de las coordenadas y altitud de la estacion
    #Se convierten las coordenadas a flotante
    estacion = cne.at[codigo, 'Nombre']
    coord_y = float(coordenadas[0])
    coord_x = float(coordenadas[1])
    altitud = cne.at[codigo,'Altitud']
    pptn_media = pptn_media_anual(df)
    registro = len(df)
    faltantes = df.isna().sum().iloc[0]
    fecha_inicial = df.index[0]
    fecha_final = df.index[-1]
    

    # Creación de diccionario con los datos extraidos
    dic_informacion = {
        'Nombre': estacion,
        'Y': coord_y,
        'X': coord_x,
        'altitud': altitud,
        'pptn_media_anual': pptn_media,
        'registro': registro,
        'Faltantes':faltantes,
        'Fecha inicial': fecha_inicial,
        'Fecha final': fecha_final,
    }

    return dic_informacion

def shape_estaciones(df):

    # creacion de un geodataframe 
    # El cual contiene la informacion de las estaciones
    gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.X, df.Y), 
    crs="EPSG:4326"
    )

    # Se convierte el sistema coordenado del geodataframe
    gdf.to_crs('EPSG:9377',inplace=True)
    
    gdf['X'] = gdf.get_coordinates()['x']
    gdf['Y'] = gdf.get_coordinates()['y']

    # Se crea shape con geodataframe


    return gdf

def ciclo_anual(df:pd.DataFrame, umbral:float) -> pd.DataFrame:

    # Se rellena el dataframe con datos NA
    df_lleno = llenar_na(df)
    #Se agregan los datos faltantes de manera mensual
    df_faltantes = df_lleno.isna().resample('ME').sum()/31
    # Se agregan los datos de manera mensual
    df_mensual = df_lleno.resample('ME').sum()
    # Filtro de datos
    datos_mensual = df_mensual[df_faltantes < umbral]
    #Generacion del ciclo anual
    ciclo_anual = datos_mensual.groupby(datos_mensual.index.month).mean()
    # Estimacion de la media anual
    
    return ciclo_anual

def analisis_frecuencias(df,var:np.array):
        
        # Crea array con los datos de interes
        var = np.array(df[var])
        # Crea dataframe
        df_analisis = pd.DataFrame({'variable': var})
        # Estimacion excedencia
        df_sort = df_analisis.sort_values(by='variable',
                                 ascending=False).reset_index(drop=True)
        
        df_sort['Excedencia'] = (df_sort.index + 1)/(len(df_sort) + 1)*100

        q_10 = np.interp(1, df_sort['Excedencia'], df_sort['variable'])
        q_90 = np.interp(90, df_sort['Excedencia'], df_sort['variable'])

        return (q_10,q_90)
    
def tormentas(df:pd.DataFrame, mit:int)->pd.DataFrame:

    # Importante: El dataframe debe ser solo el indice en formato datetime
    # y el valor de la precipitacion.
    # El codigo te regresa el dataframe inicial 
    # Con una columna nueva con un identificador numerico, que identifica el evento de precipitacion.

    # Identifica la columna que contiene laprecipitacion = df.columns[0]
    precipitacion = df.columns[0]
    # Definir el umbral de tiempo (número de ceros consecutivos que delimitan tormentas)
    # se divide por 15 dado que es el diferencial de tiempo de epm
    # Se debe programar mejor para que tome cualquier evento
    umbral_ceros = mit/15

    # Inicializar variables
    evento_id = 0
    en_evento = False
    eventos = []

    # Iterar sobre la serie de tiempo
    for i, lluvia in enumerate(df[precipitacion]):
        if lluvia > 0:
            if not en_evento:  # Iniciar un nuevo evento
                evento_id += 1
                en_evento = True
            eventos.append(evento_id)  # Asignar el evento actual
        else:
            # Verificar si hay suficientes ceros consecutivos después de un valor > 0
            conteo_ceros = 1
            for j in range(i + 1, len(df[precipitacion])):
                if df[precipitacion].iloc[j] == 0:
                    conteo_ceros += 1
                else:
                    break
            # Si los ceros consecutivos superan el umbral, terminar el evento
            if conteo_ceros >= umbral_ceros:
                en_evento = False
            eventos.append(evento_id if en_evento else 0)

    df['Evento'] = eventos

    # Mostrar el DataFrame con eventos identificados
    return df

def curva_duracion(caudal:np.array):
        # Arreglos de caudales
        caudal_sorted = np.sort(caudal)[::-1]
        caudal_sorted = caudal_sorted[~(np.isnan(caudal_sorted))]
        
        
        # 📌 Calcular la frecuencia de no excedencia (percentiles)
        n_obs = len(caudal_sorted)
        prob_exce = np.arange(1, n_obs + 1) / n_obs * 100
        # 📌 Graficar la curva de duración de caudales
        
        plt.figure(figsize=(10, 5))
        plt.plot(prob_exce, caudal_sorted,
                 linestyle="-", 
                 color="blue", label="Observados")
        
        plt.xlabel("Porcentaje de tiempo excedido (%)")
        plt.ylabel("Caudal (m³/s)")
        plt.title("Curva de Duración de Caudales")
        plt.grid(True, linestyle="--")
        plt.legend()
        plt.show()
        
        return prob_exce, caudal_sorted


class caudales_extremos:
    
    def __init__(self,df:pd.DataFrame):
        
        self.df = df
    
    def q_min_max(self,fn) -> np.array:

        # Extrae la columna que almacena los caudales
        col = self.df.columns[0]

        # Se resamplea de manera anual para extraer máximos y minimos
        if fn == 'max':
            q_extremo = self.df.resample('YE').max()
        else: 
            q_extremo = self.df.resample('YE').min()
        
        # Se eliman valores NA
        q_extremo = q_extremo.dropna()
        array_extremos = q_extremo[col].values

        # regresa un array con los datos
        return array_extremos

    def frecuencias(self,fn) -> dict:

        data = self.q_min_max(fn)
        dic_dist = dict()
        distributions = [stats.gumbel_r, 
                         stats.lognorm, 
                         stats.pearson3]

        for distribution in distributions:
            # Extrae el nombre de la distribucion
            distribucion = distribution.name
            #Ajusta la fdp
            params = distribution.fit(data)
            # Aplica prueba de bondad de ajuste    
            ks_statistic, ks_pvalue = stats.kstest(data, distribution.name, args=params)
            # Almacena resultados
            dic_dist[distribucion] = (ks_statistic,ks_pvalue)

        return dic_dist

    def fdp(self,fn):
        
        dic_dist = self.frecuencias(fn)
        
        #Delimitacion datos en dataframe
        df_fdps = pd.DataFrame(dic_dist,index=['estadistico','p_value']).T
        # filtro por p_values validos
        mask = df_fdps['p_value']>0.05
        df_fdps_filtro = df_fdps[mask]
        
        # Si el dataframe está vacio 
        # Notificar que ninguna fdp se ajsuta
        if len(df_fdps_filtro) == 0:
            raise Exception('Revisar datos, ninguna función se ajusta')
        # Se regresa como ganadora la función que los mejores ajustes
        else:
            # Estadistico con menor valor
            min = df_fdps_filtro['estadistico'].min()
            # Determinacion funcion mejor ajuste
            fdp_ajuste = df_fdps_filtro[df_fdps_filtro['estadistico']==min]
        
        return fdp_ajuste
    
    def caudales_diseño(self,fn:str) ->list:
        
        # Serie de caudales
        data = self.q_min_max(fn)
        # Seleccion funciones de distribución
        fn_ajuste = self.fdp(fn)
        fdp = fn_ajuste.index[0]
        # print(fdp)
        distribution = getattr(stats, fdp)
        # Periodos de retorno
        tr = [2.33,5,10,25,50,100]
        q_tr = list()
        for periodo in tr:
            # Estimacion periodo de retorno caudal
            if fn == 'max':
                p_periodo = 1 - 1/periodo
            else:
                p_periodo = 1/periodo
            # Ajuste de la fdp
            params = distribution.fit(data)
            q_retorno = distribution.ppf(p_periodo,*params)
            # Almacenamiento de los datos
            q_tr.append(q_retorno)
        
        return q_tr

# Agregar funcion para estimacion de multiples series
# Se puede alterar la funcion actual
    
class caudales_ambientales:
    
    def __init__(self,df:pd.DataFrame):
        
        # Dataframe con los cuadales
        self.df = df
        
        
    def metodologia_1(self,):
        
        # Minimo historico
        # Arreglos de caudales
        caudal = self.df
        caudal_array = self.df.iloc[:,0].values
        caudal_sorted = np.sort(caudal_array)[::-1]
        q_975 = np.quantile(caudal_sorted,0.025)
        #Porcentaje de descuento
        caudal_resample = caudal.resample('ME').mean()
        ciclo_anual = caudal_resample.groupby(caudal_resample.index.month).mean()
        pctg_descuento = np.min(ciclo_anual)*0.25
        #Reducción por caudal ambiental
        reduccion = 0.25*np.mean(caudal.resample('YE').mean())
        
        return q_975,pctg_descuento,reduccion
    
    def metodologia_irh(self):

        # Arreglos de caudales
        caudal = self.df
        caudal_array = caudal.iloc[:,0].values
        caudal_sorted = np.sort(caudal_array)[::-1]
        caudal_sorted = caudal_sorted[~(np.isnan(caudal_sorted))]
        
        # Calcular la frecuencia de no excedencia (percentiles)
        n_obs = len(caudal_sorted)
        prob_exce = np.arange(1, n_obs + 1) / n_obs * 100
        
        # Estimacion IRH
        q50 = np.quantile(caudal_sorted,0.5)
        distancias = np.abs(caudal_sorted - q50)
        minimo = np.nanmin(distancias)
        idx = np.where(distancias==minimo)[0][0]
        
        #Area full
        area_full = np.trapz(y=caudal_sorted,x=prob_exce)
        
        ## Slice de arrays
        caudal_sorted_slice = caudal_sorted[idx:]
        prob_exce_slice = prob_exce[idx:]
        area_q50 = np.trapz(y=caudal_sorted_slice,
                                x=prob_exce_slice)
        area_rect = prob_exce[idx]*q50
        numerador = area_q50 + area_rect
        irh = np.round(numerador/area_full,2)
        
        # Determinacion regulacion hídrica
        
        if irh > 0.85:
            calificacion = 'Muy alta retención y regulación de humedad'
        elif irh > 0.75 and irh <= 0.85:
            calificacion = 'Alta retención y regulación de humedad'
        elif irh > 0.65 and irh <= 0.75:
            calificacion = 'Media retención y regulación de humedad media'
        elif irh > 0.5 and irh <= 0.65:
            calificacion = 'Baja retención y regulación de humedad'
        else:
            calificacion = 'Muy baja retención y regulación de humedad'        
        
        return irh, calificacion
    
    
    def clasificar_enso(self,):
        
        # Lectura del dataframe con el oni
        url = 'https://raw.githubusercontent.com/Juliant06/libreria_hidrologia_v2/refs/heads/test_branch/data/oni_final.csv'
        df_oni = pd.read_csv(url,index_col=0,
                             parse_dates=[0])
        df_clasificar = self.df.copy()
        #Columna que contiene los caudales
        columna = df_clasificar.columns[0]
        
        # Inicializamos con "Neutral"
        df_oni['enso_phase'] = 'neutro'  
        for i in range(len(df_oni) - 4):
        # Tomamos una ventana de 5 meses consecutivos
            window = df_oni['ONI'].iloc[i:i+5]
            if all(window >= 0.5):
                df_oni.iloc[i:i+4, 1] = 'niño'
            elif all(window <= -0.5):
                df_oni.iloc[i:i+4, 1] = 'niña'
                
        # Clasificacion datos con el oni
        df_clasificar['month'] = df_clasificar.index.to_period('M')
        df_oni['month'] = df_oni.index.to_period('M')
        df_merged = pd.merge(df_clasificar, 
                             df_oni, 
                             on='month', 
                             how='left')
        df_merged.index = df_clasificar.index
        
        df_clasificado = df_merged[[columna,'enso_phase']]
        return df_clasificado
    
        
    def media_movil(self,array:np.array,ventana=7):
        media_movil = []
        for i in range(len(array) - ventana + 1):
            media = np.nanmean(array[i:i+ventana])
            media_movil.append(media)
        return media_movil
    
    
    def frecuencias_7q10(self,data:np.array) -> tuple:

        distribution = stats.gumbel_r
        tr = 10
        # Extrae el nombre de la distribucion
        distribucion = distribution.name
        #Ajusta la fdp
        params = distribution.fit(data)
        # Aplica prueba de bondad de ajuste    
        ks_statistic, ks_pvalue = stats.kstest(data, distribution.name, args=params)
        # Almacena resultados
        resultados_ks = (ks_statistic,ks_pvalue)
        p_periodo = 1/tr
        tr_10 = distribution.ppf(p_periodo,*params)
        
        # A partir del ajuste estimar el Tr 10 años
        

        return tr_10, resultados_ks
        
    def metodologia_3(self,):
        
        # Extraccion columna de interes
        col = self.df.columns[0]
        #Lectura de datos clasificados
        df_clasificado = self.clasificar_enso()
        dic_dfs_fases = dict()
        # Fases del enso para discretizar datos
        fases = df_clasificado['enso_phase'].unique()
        
        # For loop para generar las series
        for fase in fases:
            
            mask = df_clasificado['enso_phase'] == fase
            df_fase = df_clasificado[mask]
            # almacenamiento de dataframes
            dic_dfs_fases[fase] = df_fase
        
        # Extraccion del Q95
        dic_q95 = dict()
        for fase,df in dic_dfs_fases.items():
            #lista con los valores
            val_q95 = list()
            for mes in range(1,13):
                
                mask = df.index.month == mes
                df_mes = df[mask]
                q_95 = np.nanquantile(df_mes[col],0.05)
                val_q95.append(q_95)
            
            dic_q95[fase] = val_q95
        
        # Conversion resultados a dataframe
        df_q95 = pd.DataFrame(dic_q95,columns=fases)
        
        # Estimacion 7Q10
        
        # media movil
        dic_7q10 = dict()
        for fase,df in dic_dfs_fases.items():
            
            #Copia para no afectar el dataframe original
            df = df.copy()
            # Extrae la columna de interes 
            # Se puede generalizar mejor
            col = df.columns[0]
            #Media movil por fase
            df['media_movil'] = df[col].copy().rolling(window=7).mean().shift(1)
            # Resampleo de los datos
            df_res_min = pd.DataFrame(df['media_movil'].resample('ME').min())
            # Valor q10
            valores_q10 = list()
            # Se debe delimitar por mes
            for mes in range(1,13):
                # Delimitacion mes de analisis
                mask = df_res_min.index.month == mes
                df_mes = df_res_min[mask]
                #Dropeo de NA
                datos_limpios = df_mes.dropna()
                #Estimacion Q10 
                datos_analisis = datos_limpios.iloc[:,0].values
                q_10 = self.frecuencias_7q10(datos_analisis)[0]
                #Guardado de datos
                valores_q10.append(q_10)
                
            # Almacenamiento de datos
            dic_7q10[fase] = valores_q10
        
        df_7q10 = pd.DataFrame(dic_7q10,columns=fases)
        
        return df_q95, df_7q10
                

                
                
            
            
            
        
        
        
        
        
        
        
    


        
        
        
        
        
    
        
        
        
        
        
        
        
        