"""Funciones compartidas por los scripts de entrenamiento, evaluación y prueba."""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np


def controlador_pd(info: dict, kp: float = 12.0, kd: float = 1.0, kv: float = 0.6) -> float:
    """Controlador clásico de referencia (sin aprendizaje).

    Acelera hacia el lado al que cae el péndulo. Sirve para comprobar que el
    sistema es equilibrable y como línea base para comparar con la política RL.
    """
    u = (kp * info["incl_cabeceo"] + kd * info["dincl_cabeceo"]
         + kv * (info["vel_adelante"] - info["vel_cmd"]))
    return float(np.clip(u, -1.0, 1.0))


def kwargs_entorno_para(n_obs: int, n_acciones: int) -> dict:
    """Configuración del entorno compatible con un modelo entrenado.

    Los modelos anteriores a la orden de giro tienen una observación de 20 + n_acciones
    valores; los nuevos, 21 + n_acciones.
    """
    return dict(modo_accion="independiente" if n_acciones == 4 else "diferencial",
                comando_giro=n_obs == 21 + n_acciones)


def kwargs_para_modelo(modelo) -> dict:
    return kwargs_entorno_para(int(modelo.observation_space.shape[0]), int(modelo.action_space.shape[0]))


def buscar_vecnormalize(ruta_modelo: str | Path) -> Path | None:
    """Encuentra las estadísticas de normalización guardadas junto a un modelo."""
    ruta_modelo = Path(ruta_modelo)
    carpeta = ruta_modelo.parent
    candidatos = []
    m = re.match(r"(.*)_(\d+)_steps$", ruta_modelo.stem)
    if m:  # checkpoint: modelo_1000000_steps.zip -> modelo_vecnormalize_1000000_steps.pkl
        candidatos.append(carpeta / f"{m.group(1)}_vecnormalize_{m.group(2)}_steps.pkl")
    candidatos += [carpeta / "vecnormalize.pkl", carpeta.parent / "vecnormalize.pkl"]
    return next((c for c in candidatos if c.exists()), None)


def cargar_normalizador(ruta_modelo: str | Path):
    """Carga el VecNormalize (solo para normalizar observaciones, sin entorno)."""
    ruta = buscar_vecnormalize(ruta_modelo)
    if ruta is None:
        print("Aviso: no se encontró vecnormalize.pkl; se usarán observaciones sin normalizar.")
        return None
    with open(ruta, "rb") as f:
        vn = pickle.load(f)
    vn.training = False
    vn.norm_reward = False
    return vn


def cargar_modelo(ruta_modelo: str | Path):
    """Carga un modelo PPO o SAC de Stable-Baselines3 sin saber de antemano cuál es."""
    from stable_baselines3 import PPO, SAC

    for clase in (PPO, SAC):
        try:
            return clase.load(str(ruta_modelo), device="cpu")
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError(f"No se pudo cargar el modelo {ruta_modelo}")
