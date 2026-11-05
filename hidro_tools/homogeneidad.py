import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymannkendall as pm
import pyhomogeneity as ph


# ------------------------------------------------------------------
# Agregación anual (una sola vez, reutilizada por MK y Pettitt)
# ------------------------------------------------------------------
def serie_anual(df: pd.DataFrame, var: str,
                min_completitud: float = 0.9) -> pd.Series:
    """Agrega datos DIARIOS a escala anual descartando años incompletos."""
    if var not in ('pptn', 'caudal'):
        raise ValueError("var debe ser 'pptn' o 'caudal'")

    s = df.iloc[:, 0]
    n_validos = s.resample('YE').count()
    n_esperados = np.where(n_validos.index.is_leap_year, 366, 365)
    completo = (n_validos / n_esperados) >= min_completitud

    if var == 'pptn':
        agg = s.resample('YE').sum(min_count=1)   # año todo NaN -> NaN, no 0
    else:
        agg = s.resample('YE').mean()

    return agg[completo].dropna()


# ------------------------------------------------------------------
# Mann-Kendall (elige versión según autocorrelación)
# ------------------------------------------------------------------
def mann_kendall(df, var, min_completitud=0.9, alpha=0.05):
    serie = serie_anual(df, var, min_completitud)
    if len(serie) < 10:
        raise ValueError(f'Solo {len(serie)} años válidos (mínimo 10)')

    data = serie.values
    r1 = pd.Series(data).autocorr(lag=1)
    hay_autocorr = abs(r1) > 1.96 / np.sqrt(len(data))

    if hay_autocorr:
        res = pm.hamed_rao_modification_test(data, alpha=alpha)
    else:
        res = pm.original_test(data, alpha=alpha)

    return res, {'autocorr_lag1': r1, 'version_modificada': hay_autocorr}


# ------------------------------------------------------------------
# Pettitt
# ------------------------------------------------------------------
def pettitt(df, var, min_completitud=0.9, alpha=0.05):
    serie = serie_anual(df, var, min_completitud)
    res = ph.pettitt_test(serie.values, alpha=alpha)
    return serie, res


def plot_pettitt(df, var, estacion, path_guardado,
                 min_completitud=0.9, alpha=0.05):
    serie, res = pettitt(df, var, min_completitud, alpha)
    ylabel = ('Precipitación anual [mm]' if var == 'pptn'
              else 'Caudal medio anual [m³/s]')

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(serie.index, serie.values, color='blue', marker='o', ms=4)

    # Solo se dibuja el cambio si es significativo
    if res.h:
        loc = serie.index[res.cp]
        ax.hlines(res.avg.mu1, serie.index[0], loc, linestyles='--',
                  colors='orange', lw=1.5, label=f'mu1: {res.avg.mu1:.2f}')
        ax.hlines(res.avg.mu2, loc, serie.index[-1], linestyles='--',
                  colors='g', lw=1.5, label=f'mu2: {res.avg.mu2:.2f}')
        ax.axvline(loc, ls='-.', color='red', lw=1.5,
                   label=f'Cambio: {loc.year} (p = {res.p:.4f})')
    else:
        ax.plot([], [], ' ', label=f'Sin cambio significativo (p = {res.p:.3f})')

    ax.set_title(f'Test de Pettitt estación {estacion}', fontsize=15)
    ax.set_xlabel('Años', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid()
    ax.legend(loc='upper right', fontsize=13)
    fig.tight_layout()
    fig.savefig(path_guardado)
    plt.close(fig)


# ------------------------------------------------------------------
# Media móvil
# ------------------------------------------------------------------
def plot_media_movil(serie: pd.Series, ventana: int):
    media = serie.rolling(ventana, center=True).mean()   # alineada con la serie
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(serie, color='blue', label='Serie')
    ax.plot(media, color='red', label=f'Media móvil ({ventana})')
    ax.legend()
    ax.grid()
    return fig


# ------------------------------------------------------------------
# Outliers: se MARCAN, no se borran
# ------------------------------------------------------------------
def detectar_outliers(df: pd.DataFrame, k: float = 3.0,
                      metodo: str = 'sigma', log: bool = False,
                      solo_superior: bool = True):
    """
    metodo: 'sigma' (media ± k·std) o 'mad' (mediana ± k·1.4826·MAD).
    log=True aplica el criterio sobre log(x), recomendado para caudales.
    Devuelve (df con outliers -> NaN, outliers, (lim_inf, lim_sup)).
    """
    col = df.columns[0]
    s = df[col].astype(float)
    x = np.log(s.where(s > 0)) if log else s

    if metodo == 'sigma':
        centro, escala = x.mean(), x.std()
    elif metodo == 'mad':
        centro = x.median()
        escala = 1.4826 * (x - centro).abs().median()
    else:
        raise ValueError("metodo debe ser 'sigma' o 'mad'")

    lim_sup = centro + k * escala
    lim_inf = centro - k * escala

    mask = x > lim_sup
    if not solo_superior:
        mask = mask | (x < lim_inf)

    df_marcado = df.copy()
    df_marcado.loc[mask, col] = np.nan          # conserva el índice original
    outliers = df.loc[mask]

    lims = (np.exp(lim_inf), np.exp(lim_sup)) if log else (lim_inf, lim_sup)
    return df_marcado, outliers, lims


# ------------------------------------------------------------------
# Grubbs-Beck (Bulletin 17B) sobre MÁXIMOS ANUALES
# ------------------------------------------------------------------
def grubbs_beck(serie: pd.Series):
    x = serie.dropna()
    if (x <= 0).any():
        raise ValueError('Se requieren valores > 0 (se usa logaritmo)')
    n = len(x)
    if n < 10:
        raise ValueError('Se requieren al menos 10 datos')

    log_x = np.log10(x)
    media, desv = log_x.mean(), log_x.std(ddof=1)

    if n <= 150:
        k_n = (-0.9043 + 3.345 * np.sqrt(np.log10(n))
               - 0.4046 * np.log10(n))              # log10(N), no ln
    else:
        k_n = (-3.62201 + 6.2844 * n**0.25 - 2.49835 * n**0.5
               + 0.491436 * n**0.75 - 0.037911 * n)

    x_h = 10 ** (media + k_n * desv)
    x_l = 10 ** (media - k_n * desv)

    mask = (x < x_l) | (x > x_h)
    return x[~mask], x[mask], (x_l, x_h)


def plot_grubbs_beck(serie: pd.Series, estacion=None, path_guardado=None):
    limpio, outliers, (x_l, x_h) = grubbs_beck(serie)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(serie.index, serie.values, color='blue', marker='o', ms=4,
            label='Máximos anuales')
    if len(outliers) > 0:
        ax.scatter(outliers.index, outliers.values, color='red', s=70,
                   zorder=3, label='Outlier')
    else:
        print('No se presentan outliers')
    ax.axhline(x_h, ls='--', color='orange', label=f'Umbral sup: {x_h:.1f}')
    ax.axhline(x_l, ls='--', color='green', label=f'Umbral inf: {x_l:.1f}')

    titulo = 'Detección de outliers (Grubbs-Beck)'
    ax.set_title(f'{titulo} {estacion}' if estacion else titulo, fontsize=15)
    ax.grid()
    ax.legend(fontsize=12)

    if path_guardado:
        fig.savefig(path_guardado, bbox_inches='tight')
        plt.close(fig)
    return fig