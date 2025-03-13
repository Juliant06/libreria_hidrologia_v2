from matplotlib.pylab import rand
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

class modelo_tanques:
    """ Modelo agregado de simulación hidrológica de caudales basado en el artículo de Vélez et al.(2010)
        Obtenido de: https://repositorio.unal.edu.co/bitstream/handle/unal/8020/DA_242.pdf?sequence=1&isAllowed=y
    """    
    def __init__(self, pptn: np.array,
                almac_capilar:float,
                cond_capa_sup:float,
                cond_capa_inf:float,
                perdidas_subt:float,
                tr_2:float,
                tr_3:float,
                tr_4:float,
                h:float,
                area:float,
                param_lluvia:float):
        """_summary_

        Args:
            pptn (array): Array de datos unidimensional que contiene los valores de precipitación
            almac_capilar (float): Tamaño del tanque de almacenamiento superficial en mm
            cond_capa_sup (float): Conductividad hidráulica de la capa superior del suelo asociada a la cobertura en condiciones de saturación
            cond_capa_inf (float): Conductividad hidráulica en la capa inferior del suelo
            perdidas_subt (float): Pérdidad subterránea
            tr_2 (float): Tiempo de residencia
            tr_3 (float): Tiempo de residencia
            tr_4 (float): Tiempo de residencia
            h (float): Cota promedio de la cuenca de interés
            area (float): Área de la cuenca
            param_lluvia (float): Párametro de ajuste de la lluvia
        """        
    
        # parametros 
        self.pptn = pptn*param_lluvia #precipitacion
        self.almac_capilar = almac_capilar #Almacenamiento capilar (Tamaño del tanque)
        self.ks = cond_capa_sup # Conductividad de la capa superior
        self.kp = cond_capa_inf # Conducitividad capa inferior
        self.perdidas_subt = perdidas_subt # Pérdidas subterráneas
        self.tr_2 = tr_2 # Tiempo de residencia
        self.tr_3 = tr_3 # Tiempo de residencia
        self.tr_4 = tr_4 # Tiempo de residencia
        self.alpha = 2.0 #Exponente infiltracion
        self.beta = 0.7 #Exponente evaporación
        self.h = h #Altitud msnm
        self.area = area
    
        # Condiciones iniciales 
        # Cantidad de agua en el tanque al momento de la modelacion
        self.almac_capilar_0 = random.uniform(0,120)  #60.0
        self.almac_agua_superficial = random.uniform(0,10) #5.0
        self.almac_z_sup = random.uniform(0,30)  #15
        self.almac_z_inf = random.uniform(0,2000) #1000
    
    
    ### Ecuaciones del modelo ###
    
    #Metodo budyko cenicafe
    def etp(self):
        etp = 4.658*np.exp(-0.0002*self.h)

        return etp
    
    
    def tanque_1(self,p,Hi):
        """ Almacenamiento capilar

        Args:
            p (float): precipitación
            Hi (float): Volumen del tanque en cada paso temporal 

        Returns:
            tuple: flujo excedente (x_2), escorrentía (y_1) y D1 cantidad de agua que entra al nivel estático
        """        
        
        phi = 1 - (Hi/self.almac_capilar)**self.alpha
    
        D1 = min(p*phi, self.almac_capilar - Hi)
    
        # Salidas por evapotranspiracion
        y_1 = min(self.etp()*(Hi/self.almac_capilar)**self.beta,Hi)
        #Infiltracion
        x_2 = p - D1
        #actualizacion del almacenamiento capilar 
    
        return x_2, y_1, D1
    
    
    def tanque_2(self, x_2, H2):
        """ Almacenamiento de flujo superficial

        Args:
            x_2 (float): Flujo excedente del almacenamiento capilar
            H2 (float): Contenido de agua del tanque 2 en cada paso temporal

        Returns:
            tuple: flujo superficial (y_2), flujo excedente (x_3) y D2 cantidad de agua que entra al tanque
        """        
    
        # Entradas de agua al tanque
        D2 = max(0,x_2 - self.ks)
        a_2 = 1/self.tr_2
        # Salida por escorrentia 
        y_2 = a_2*H2
    
        # Infiltracion
        x_3 = x_2 - D2
    
        return y_2, x_3, D2
    
    def tanque_3(self, x_3, H3):
        """ Almacenamiento de flujo subsuperficial

        Args:
            x_3 (float): Flujo gravitacional
            H3 (float): Contenido de agua del tanque 3 en cada paso temporal

        Returns:
            tuple: flujo subsuperficial (y_3), percolación (x_4), D3 cantidad de agua que entra al tanque
        """        
    
        # Entradas de agua al tanque
        D3 = max(0,x_3 - self.kp)
    
        a_3 = 1/self.tr_3
        #Salidas de agua subsuperficial
        y_3 = a_3*H3
        # Percolacion
        x_4 = x_3 - D3
    
        return y_3,x_4,D3
    
    def tanque_4(self, x_4, H4):
        """Almacenamiento de flujo subterráneo

        Args:
            x_4 (float): percolación
            H4 (float): Contenido de agua del tanque 4 en cada paso temporal

        Returns:
            tuple: flujo base (y_4) y cantidad de agua que entra al tanque (D4)
        """        
        
        D4 = max(0,x_4 - self.perdidas_subt)
        a_4 = 1/self.tr_4
        #Flujo base
        y_4 = a_4*H4
    
    
        return y_4, D4
    
    def flujo_base(self, y_4):
        """_summary_

        Args:
            y_4 (float): lámina de agua correspondiente al flujo base en mm

        Returns:
            float: Regresa el caudal (m3/dia) asociado al flujo base
        """        
        flujo_base = y_4*self.area*1000/86400

        return flujo_base
    
    def caudal(self,y_2,y_3,y_4):
        """ Cálculo del caudal simulado

        Args:
            y_2 (float): Escorrentía directa
            y_3 (float): Flujo subsuperficial
            y_4 (float): Flujo base

        Returns:
            float: Caudal simulado (m3/dia)
        """        
    
        # Calculo del caudal
        Q = (y_2 + y_3 + y_4)*self.area*1000/86400
    
        return Q 
    
    ### Funciones para correr el modelo ###
    
    def correr(self):
        """ Función que se encarga de correr todo el modelo

        Returns:
            tuple: Regresa un array con el flujo base y un array con el caudal simulado.
        """        
    
        # Lista que contiene los caudales simulados
        caudal_simulado = []
        # Lista del flujo base
        flujo_base = []
    
        for i in range(len(self.pptn)):
        
            #Condiciones iniciales
            if i == 0:  
                Hi = self.almac_capilar_0    
                p = self.pptn[i]
                x_2, y1,D1 = self.tanque_1(p,Hi)
                # Actualizacion del almacenamiento capilar 
                Hi = Hi - y1 + D1
                # Tanque 2
                H2 = self.almac_agua_superficial
                y_2, x_3, D2 = self.tanque_2(x_2,H2)
                # actualizacion H2
                H2 = H2 - y_2 + D2
                #Tanque 3
                H3 = self.almac_z_sup
                y_3, x_4, D3 = self.tanque_3(x_3,H3)
                #Actualizacion H3
                H3 = H3 - y_3 + D3
                # Tanque 4
                H4 = self.almac_z_inf
                y_4, D4 = self.tanque_4(x_4,H4)
                H4 = H4 - y_4 + D4
                # Calculo del caudal simulado
                caudal = self.caudal(y_2,y_3,y_4)
                # Calculo del flujo base 
                fb= self.flujo_base(y_4)
                # Almacenamiento de los datos simulados
                caudal_simulado.append(caudal)
                flujo_base.append(fb)
            else:
                p = self.pptn[i]
                x_2, y1, D1 = self.tanque_1(p,Hi)
                # Actualizacion del almacenamiento capilar 
                Hi = Hi - y1 + D1 
                #Tanque_2
                y_2, x_3, D2 = self.tanque_2(x_2,H2)
                # actualizacion H2
                H2 = H2 - y_2 + D2
                #Tanque 3
                y_3, x_4, D3 = self.tanque_3(x_3,H3)
                #Actualizacion H3
                H3 = H3 - y_3 + D3
                # Tanque 4
                y_4, D4 = self.tanque_4(x_4,H4)
                H4 = H4 - y_4 + D4
                # Calculo del caudal simulado
                caudal = self.caudal(y_2,y_3,y_4)
                #calculo flujo base
                fb= self.flujo_base(y_4)
                # Almacenamiento de los datos simulados
                caudal_simulado.append(caudal)
                flujo_base.append(fb)
        return caudal_simulado, flujo_base

# 
