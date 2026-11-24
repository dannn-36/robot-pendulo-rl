"""Funciones compartidas por los scripts de entrenamiento, evaluación y prueba."""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np

# Calidad del dibujo 3D (argumentos de RobotPenduloEnv). Solo cambia lo que se ve:
# la física usa siempre el terreno de 5 cm. Medido en una RTX 2080 a 960x540:
# "alta" tarda ~35 ms por imagen (más que los 20 ms de un paso de control) y
# "media" ~3 ms; en un portátil sin gráfica dedicada, "alta" va a saltos.
CALIDADES = {
    "alta": dict(resolucion_visual=None, sombras=True, tamano_sombras=4096),  # terreno de 5 cm
    "media": dict(resolucion_visual=0.1, sombras=True, tamano_sombras=2048),  # terreno de 10 cm
    "baja": dict(resolucion_visual=0.2, sombras=False, tamano_sombras=1024),  # terreno de 20 cm
}


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
    try:
        with open(ruta, "rb") as f:
            vn = pickle.load(f)
    except ModuleNotFoundError as e:
        if "numpy._core" in str(e):  # guardado con numpy 2: numpy 1.x no lo puede leer
            raise RuntimeError(f"{ruta} se guardó con numpy 2 y aquí está instalado numpy "
                               f"{np.__version__}. Actualízalo: pip install -U \"numpy>=2\"") from e
        raise
    vn.training = False
    vn.norm_reward = False
    return vn


def cargar_modelo(ruta_modelo: str | Path):
    """Carga un modelo PPO o SAC de Stable-Baselines3 sin saber de antemano cuál es.

    No carga el learning rate ni el clip range: se guardan como funciones de Python
    (cloudpickle) que pueden no leerse con otra versión de Python, y para usar la
    política no hacen falta.
    """
    from stable_baselines3 import PPO, SAC

    ruta = Path(ruta_modelo)
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el modelo {ruta}")
    sin_funciones = dict(learning_rate=0.0, lr_schedule=lambda _: 0.0, clip_range=lambda _: 0.0)
    errores = []
    for clase in (PPO, SAC):
        try:
            return clase.load(str(ruta), device="cpu", custom_objects=sin_funciones)
        except Exception as e:  # noqa: BLE001
            errores.append(f"{clase.__name__}: {type(e).__name__}: {e}")
    raise RuntimeError(f"No se pudo cargar el modelo {ruta}:\n  " + "\n  ".join(errores))
