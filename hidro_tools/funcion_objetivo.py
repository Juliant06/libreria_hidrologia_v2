import numpy as np
import pandas as pd


def _pares(array_obs, array_sim):
    """ Devuelve obs y sim solo en los días donde AMBOS son válidos (no NaN).
    Así todas las métricas se calculan sobre exactamente los mismos días. """
    obs = np.asarray(array_obs, dtype=float)
    sim = np.asarray(array_sim, dtype=float)
    mask = ~(np.isnan(obs) | np.isnan(sim))
    return obs[mask], sim[mask]


def _nse(obs, sim):
    numerador = np.sum((obs - sim) ** 2)
    denominador = np.sum((obs - obs.mean()) ** 2)
    return np.around(1 - numerador / denominador, 5)


def nash_sqrt(array_obs, array_sim):
    with np.errstate(invalid='ignore'):
        obs, sim = _pares(np.sqrt(np.asarray(array_obs, dtype=float)),
                          np.sqrt(np.asarray(array_sim, dtype=float)))
    return _nse(obs, sim)


def nash_log(array_obs, array_sim, eps=1e-3):
    # eps evita log(0) = -inf cuando hay caudales nulos
    with np.errstate(invalid='ignore'):
        obs, sim = _pares(np.log(np.asarray(array_obs, dtype=float) + eps),
                          np.log(np.asarray(array_sim, dtype=float) + eps))
    return _nse(obs, sim)


def nash(array_obs, array_sim):
    obs, sim = _pares(array_obs, array_sim)
    return _nse(obs, sim)


def balance(array_obs: np.array, array_sim: np.array, absoluto: bool = True) -> float:
    # Error porcentual en el balance (BE, ec. 17 del artículo). Óptimo = 0 %.
    # absoluto=False conserva el signo: positivo = el modelo subestima el volumen.
    obs, sim = _pares(array_obs, array_sim)
    be = 100 * (obs.mean() - sim.mean()) / obs.mean()
    return np.abs(be) if absoluto else be


def minimos(array_obs: np.array, array_sim: np.array) -> float:
    obs, sim = _pares(array_obs, array_sim)
    q05_sim = np.quantile(sim, 0.05)
    q05_obs = np.quantile(obs, 0.05)

    q_05 = 100 * np.abs(q05_obs - q05_sim) / q05_obs
    return q_05


def rmse(array_obs: np.array, array_sim: np.array) -> float:
    obs, sim = _pares(array_obs, array_sim)
    return np.round(np.sqrt(np.mean((obs - sim) ** 2)), 3)


def rmse_modificado(array_obs: np.array, array_sim: np.array) -> float:
    # RMSE sobre la raíz cuadrada de los caudales observados y simulados
    with np.errstate(invalid='ignore'):
        obs, sim = _pares(np.sqrt(np.asarray(array_obs, dtype=float)),
                          np.sqrt(np.asarray(array_sim, dtype=float)))
    return np.round(np.sqrt(np.mean((obs - sim) ** 2)), 3)