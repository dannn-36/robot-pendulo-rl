"""Comprobaciones rápidas del simulador (sin entrenar nada).

    python probar_entorno.py              # chequeo de API, velocidad y galería de terrenos
    python probar_entorno.py --sin-imagen # sin generar la galería (no necesita OpenGL)
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from robot_pendulo import TIPOS_TERRENO, RobotPenduloEnv


if hasattr(sys.stdout, "reconfigure"):  # consolas Windows sin UTF-8
    sys.stdout.reconfigure(errors="replace")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sin-imagen", action="store_true")
    p.add_argument("--salida", default="terrenos.png")
    args = p.parse_args()

    from stable_baselines3.common.env_checker import check_env

    env = RobotPenduloEnv()
    check_env(env, warn=True)
    print(f"OK API Gymnasium correcta · observación {env.observation_space.shape} · acción {env.action_space.shape}")

    env.reset(seed=0)
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < 5.0:
        _, _, term, trunc, _ = env.step(env.action_space.sample() * 0.3)
        n += 1
        if term or trunc:
            env.reset()
    vel = n / (time.perf_counter() - t0)
    print(f"OK {vel:,.0f} pasos/s por proceso ({vel * env.dt:,.0f}x tiempo real)")
    env.close()

    if args.sin_imagen:
        return
    import imageio.v2 as imageio

    env = RobotPenduloEnv(render_mode="rgb_array", ancho_video=480, alto_video=300, camara="lateral",
                          empujones=False)
    imagenes = []
    for tipo in TIPOS_TERRENO:
        env.reset(seed=1, options=dict(tipo_terreno=tipo, dificultad=1.0, posicion=(0.0, 0.0), rumbo=0.0))
        for _ in range(10):
            env.step(np.zeros(4))
        imagenes.append(env.render())
    env.close()
    filas = [np.concatenate(imagenes[i:i + 3], axis=1) for i in (0, 3)]
    imageio.imwrite(args.salida, np.concatenate(filas, axis=0))
    print(f"OK Galería de terrenos (dificultad 1) guardada en {args.salida}: " + ", ".join(TIPOS_TERRENO))


if __name__ == "__main__":
    main()
