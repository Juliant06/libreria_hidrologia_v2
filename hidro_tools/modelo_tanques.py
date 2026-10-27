import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class modelo_tanques:
    """ Modelo agregado de simulación hidrológica de caudales basado en el artículo de Vélez et al.(2010)
        Obtenido de: https://repositorio.unal.edu.co/bitstream/handle/unal/8020/DA_242.pdf?sequence=1&isAllowed=y

        Uso manual (parámetros ingresados por el usuario):
            modelo = modelo_tanques(pptn, almac, ks, kp, x5, tr2, tr3, tr4, alpha, beta, f_lluvia, cota, area)
            q_sim, q_base = modelo.correr()

        Uso Monte Carlo:
            df, sims = modelo_tanques.montecarlo(pptn, q_obs, cota, area, fobj=funcion_objetivo, n_iter=1000)
    """

    # Parámetros calibrables en el orden en que los recibe __init__ (después de pptn)
    NOMBRES_PARAMS = ['almac_capilar', 'ks', 'kp', 'perdidas_subt',
                      'tr_2', 'tr_3', 'tr_4', 'alpha', 'beta', 'Lluvia']

    # Rangos por defecto del Monte Carlo (min, max).
    # Hu, Ks, Kp, X5, Tr2, Tr3, Tr4: rangos de la Tabla 2 de Vélez et al. (2010)
    # alpha: 1-3 (HBV; GR4J usa 2) | beta: 0.5-1 (GR-2/GR-3 usa 0.5, HBV usa 1)
    RANGOS = {
        'almac_capilar': (10, 420),   # mm
        'ks': (2, 90),                # mm/día
        'kp': (0, 10),                # mm/día
        'perdidas_subt': (0, 1.0),    # mm/día (en el artículo casi siempre = 0)
        'tr_2': (1.0, 9.0),           # días (>= 1, porque a = 1/Tr no puede superar 1)
        'tr_3': (2.0, 10.0),          # días
        'tr_4': (30, 200),            # días
        'alpha': (1, 3),
        'beta': (0.5, 1),
        'Lluvia': (1.0, 1.15),        # factor de ajuste de la lluvia
    }

    def __init__(
            self, 
            pptn: np.array,
            almac_capilar: float,
            cond_capa_sup: float,
            cond_capa_inf: float,
            perdidas_subt: float,
            tr_2: float,
            tr_3: float,
            tr_4: float,
            alpha: float,
            beta: float,
            param_lluvia: float,
            h: float,
            area: float,
            cond_ini: list = None
        ):
        """
        Args:
            pptn (array): Array unidimensional con la precipitación diaria (mm/día)
            almac_capilar (float): Capacidad del tanque de almacenamiento capilar, Hu, en mm
            cond_capa_sup (float): Conductividad hidráulica de la capa superior del suelo (Ks), mm/día
            cond_capa_inf (float): Conductividad hidráulica de la capa inferior del suelo (Kp), mm/día
            perdidas_subt (float): Pérdidas subterráneas (X5), mm/día
            tr_2 (float): Tiempo de residencia del tanque 2 (días)
            tr_3 (float): Tiempo de residencia del tanque 3 (días)
            tr_4 (float): Tiempo de residencia del tanque 4 (días)
            alpha (float): Exponente de la función de infiltración (phi)
            beta (float): Exponente de la evapotranspiración
            param_lluvia (float): Parámetro de ajuste de la lluvia
            h (float): Cota promedio de la cuenca de interés (msnm)
            area (float): Área de la cuenca (km2)
            cond_ini (list, optional): [H1, H2, H3, H4] en mm. Si es None se usan los valores de arranque por defecto.
        """

        if min(tr_2, tr_3, tr_4) < 1:
            raise ValueError('Los tiempos de residencia deben ser >= 1 día: con Tr < 1 el coeficiente '
                             'a = 1/Tr > 1 y el tanque se vacía por debajo de cero.')

        # parametros
        self.pptn = np.asarray(pptn, dtype=float) * param_lluvia  # precipitacion
        self.almac_capilar = almac_capilar  # Almacenamiento capilar (Tamaño del tanque)
        self.ks = cond_capa_sup  # Conductividad de la capa superior
        self.kp = cond_capa_inf  # Conducitividad capa inferior
        self.perdidas_subt = perdidas_subt  # Pérdidas subterráneas
        self.tr_2 = tr_2  # Tiempo de residencia
        self.tr_3 = tr_3  # Tiempo de residencia
        self.tr_4 = tr_4  # Tiempo de residencia
        self.alpha = alpha  # Exponente infiltracion
        self.beta = beta  # Exponente evaporación
        self.h = h  # Altitud msnm
        self.area = area

        # Condiciones iniciales (mm)
        if cond_ini is None:
            cond_ini = [0.5 * almac_capilar, tr_2 * 1.0, tr_3 * 1.0, tr_4 * 1.0]
        (self.almac_capilar_0,
         self.almac_agua_superficial,
         self.almac_z_sup,
         self.almac_z_inf) = cond_ini

    ### Ecuaciones del modelo ###

    def cond_iniciales(self,) -> list:

        # Retorna las condiciones iniciales
        # utilizadas para correr la simulacion

        condiciones = [
            self.almac_capilar_0,
            self.almac_agua_superficial,
            self.almac_z_sup,
            self.almac_z_inf,
        ]

        return condiciones

    # Metodo budyko cenicafe
    def etp(self):
        etp = 4.658 * np.exp(-0.0002 * self.h)

        return etp

    def tanque_1(self, p, Hi):
        """ Almacenamiento capilar

        Args:
            p (float): precipitación
            Hi (float): Volumen del tanque en cada paso temporal

        Returns:
            tuple: flujo excedente (x_2), evapotranspiración real (y_1) y D1 cantidad de agua que entra al nivel estático
        """
        phi = 1 - (Hi / self.almac_capilar) ** self.alpha
        D1 = min(p * phi, self.almac_capilar - Hi)
        # Salidas por evapotranspiracion
        y_1 = min(self.etp() * (Hi / self.almac_capilar) ** self.beta, Hi)
        # Excedente
        x_2 = p - D1
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
        D2 = max(0, x_2 - self.ks)
        a_2 = 1 / self.tr_2
        # Salida por escorrentia
        y_2 = a_2 * H2
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
        D3 = max(0, x_3 - self.kp)
        a_3 = 1 / self.tr_3
        # Salidas de agua subsuperficial
        y_3 = a_3 * H3
        # Percolacion
        x_4 = x_3 - D3
        return y_3, x_4, D3

    def tanque_4(self, x_4, H4):
        """Almacenamiento de flujo subterránea

        Args:
            x_4 (float): percolación
            H4 (float): Contenido de agua del tanque 4 en cada paso temporal

        Returns:
            tuple: flujo base (y_4) y cantidad de agua que entra al tanque (D4)
        """
        D4 = max(0, x_4 - self.perdidas_subt)
        a_4 = 1 / self.tr_4
        # Flujo base
        y_4 = a_4 * H4
        return y_4, D4

    def flujo_base(self, y_4):
        """
        Args:
            y_4 (float): lámina de agua correspondiente al flujo base en mm/día

        Returns:
            float: Caudal (m3/s) asociado al flujo base (área en km2)
        """
        flujo_base = y_4 * self.area * 1000 / 86400
        return flujo_base

    def caudal(self, y_2, y_3, y_4):
        """ Cálculo del caudal simulado

        Args:
            y_2 (float): Escorrentía directa
            y_3 (float): Flujo subsuperficial
            y_4 (float): Flujo base

        Returns:
            float: Caudal simulado (m3/s). mm/día * km2 * 1000 / 86400 = m3/s
        """
        Q = (y_2 + y_3 + y_4) * self.area * 1000 / 86400
        return Q

    ### Funciones para correr el modelo ###

    def curva_duracion(self, q_obs, q_sim):
        q_obs = np.asarray(q_obs, dtype=float)
        q_sim = np.asarray(q_sim, dtype=float)

        # Solo se comparan los días con dato observado (los NaN de np.sort quedarían al inicio al invertir)
        mask = ~np.isnan(q_obs)
        q_obs_sorted = np.sort(q_obs[mask])[::-1]
        q_sim_sorted = np.sort(q_sim[mask])[::-1]

        # Probabilidad de excedencia (Weibull)
        n = len(q_obs_sorted)
        prob_exce = np.arange(1, n + 1) / (n + 1) * 100

        plt.figure(figsize=(10, 5))

        plt.plot(prob_exce, q_obs_sorted,
                 linestyle="-",
                 color="blue", label="Observados")

        plt.plot(prob_exce, q_sim_sorted,
                 linestyle="-", alpha=0.8,
                 color="red", label="Simulados")

        plt.xlabel("Porcentaje de tiempo excedido (%)")
        plt.ylabel("Caudal (m³/s)")
        plt.title("Curva de Duración de Caudales")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.show()

    def plot_resultados(self, q_obs, q_sim, pptn):

        fig, ax = plt.subplots(figsize=(12, 6))

        # Limites de los graficos
        max_q = np.nanmax(q_obs)
        max_pptn = np.nanmax(pptn)

        ax.plot(np.array(q_obs), color="r", label='Observado')
        ax.plot(q_sim, color='blue', label='Simulado', alpha=0.8)
        ax.set_ylabel('Caudal [$m^3/s$]')
        ax.set_ylim(0, max_q + 50)

        # Create second axes, in order to get the bars from the top you can multiply
        # by -1
        ax2 = ax.twinx()
        ax2.plot(-np.array(pptn), color='black', label='precipitación')
        ax2.set_ylabel('Precipitación [mm]')
        ax2.set_ylim(-max_pptn - 60, 0)
        # Leyendas
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()

        ax.legend(handles1 + handles2, labels1 + labels2, loc="upper center",
                  bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False)

        plt.show()

    def correr(self):
        """ Función que se encarga de correr todo el modelo

        Returns:
            tuple: Regresa una lista con el caudal simulado y una lista con el flujo base (m3/s).
        """

        caudal_simulado = []
        caudal_base = []

        # Condiciones iniciales de los 4 tanques
        Hi, H2, H3, H4 = self.cond_iniciales()

        for p in self.pptn:
            # Tanque 1
            x_2, y_1, D1 = self.tanque_1(p, Hi)
            Hi = Hi - y_1 + D1
            # Tanque 2
            y_2, x_3, D2 = self.tanque_2(x_2, H2)
            H2 = H2 - y_2 + D2
            # Tanque 3
            y_3, x_4, D3 = self.tanque_3(x_3, H3)
            H3 = H3 - y_3 + D3
            # Tanque 4
            y_4, D4 = self.tanque_4(x_4, H4)
            H4 = H4 - y_4 + D4

            caudal_simulado.append(self.caudal(y_2, y_3, y_4))
            caudal_base.append(self.flujo_base(y_4))

        return caudal_simulado, caudal_base

    ### Monte Carlo ###

    @classmethod
    def almacenamiento(cls):
        """ Diccionario donde se guardan los parámetros y el desempeño de cada corrida """
        claves = cls.NOMBRES_PARAMS + ['cond_iniciales', 'Nash', 'Nash_mod',
                                       'rmse', 'rmse_mod', 'Balance', 'q_05']
        return {k: [] for k in claves}

    @classmethod
    def muestrear_params(cls, rng, rangos=None, fijos=None, coherencia=True, max_intentos=1000):
        """ Genera un set de parámetros aleatorio (uniforme).

        Args:
            rng: np.random.Generator
            rangos (dict, optional): Rangos {'nombre': (min, max)} que reemplazan a los de RANGOS
            fijos (dict, optional): Parámetros que NO se muestrean, {'nombre': valor}
            coherencia (bool): Si es True exige coherencia física del artículo: Ks > Kp y Tr2 <= Tr3 <= Tr4
        Returns:
            dict con los parámetros en el orden de NOMBRES_PARAMS
        """
        r = {**cls.RANGOS, **(rangos or {})}
        fijos = fijos or {}

        for _ in range(max_intentos):
            p = {k: fijos[k] if k in fijos else rng.uniform(*r[k]) for k in cls.NOMBRES_PARAMS}
            if not coherencia or (p['ks'] > p['kp'] and p['tr_2'] <= p['tr_3'] <= p['tr_4']):
                return p
        raise RuntimeError('No se logró un set de parámetros coherente; revisa rangos/fijos.')

    @classmethod
    def montecarlo(cls, pptn, q_obs, cota, area, fobj,
                   n_iter=1000, rangos=None, fijos=None,
                   coherencia=True, calentamiento=0, seed=None):
        """ Corre n_iter simulaciones con parámetros aleatorios y calcula su desempeño.

        Args:
            pptn: precipitación diaria de la cuenca
            q_obs: caudal observado (misma longitud que pptn)
            cota (float): cota media de la cuenca (msnm)
            area (float): área de la cuenca (km2)
            fobj: módulo/objeto con nash, nash_sqrt, rmse, rmse_modificado, balance y minimos
            n_iter (int): número de corridas
            rangos (dict, optional): rangos propios por parámetro
            fijos (dict, optional): parámetros que se dejan fijos (ej. {'perdidas_subt': 0})
            coherencia (bool): impone Ks > Kp y Tr2 <= Tr3 <= Tr4
            calentamiento (int): días iniciales que se excluyen de las métricas (artículo: 3 a 6 meses)
            seed (int, optional): semilla para reproducir las corridas

        Returns:
            tuple: (DataFrame con parámetros y métricas, lista con los caudales simulados)
        """
        pptn = np.asarray(pptn, dtype=float)
        q_obs = np.asarray(q_obs, dtype=float)

        if len(pptn) != len(q_obs):
            raise ValueError('No coinciden las longitudes de pptn y q_obs (revisar fechas)')

        rng = np.random.default_rng(seed)
        registro = cls.almacenamiento()
        simulaciones = []
        obs = q_obs[calentamiento:]

        for _ in range(n_iter):
            params = cls.muestrear_params(rng, rangos, fijos, coherencia)

            # Montaje de parametros al modelo y simulacion
            modelo = cls(pptn, *params.values(), cota, area)
            caudal_sim, _ = modelo.correr()
            caudal_sim = np.asarray(caudal_sim)
            simulaciones.append(caudal_sim)
            sim = caudal_sim[calentamiento:]

            # Parametros y condiciones iniciales utilizadas
            for k, v in params.items():
                registro[k].append(v)
            registro['cond_iniciales'].append(modelo.cond_iniciales())

            # Desempeño del modelo
            registro['Nash'].append(fobj.nash(obs, sim))
            registro['Nash_mod'].append(fobj.nash_sqrt(obs, sim))
            registro['rmse'].append(fobj.rmse(obs, sim))
            registro['rmse_mod'].append(fobj.rmse_modificado(obs, sim))
            registro['Balance'].append(fobj.balance(obs, sim))
            registro['q_05'].append(fobj.minimos(obs, sim))

        return pd.DataFrame(registro), simulaciones