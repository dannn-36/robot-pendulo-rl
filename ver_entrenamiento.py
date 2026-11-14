"""Muestra en 3D cómo va el entrenamiento, mientras entrena.

Carga el checkpoint más reciente de un entrenamiento y lo ejecuta en la ventana 3D.
Al terminar cada episodio comprueba si hay uno más nuevo y lo carga, así ves al
robot mejorar. Usa el nivel de terreno que el currículo haya alcanzado.

    python ver_entrenamiento.py                          # el entrenamiento más reciente en runs/
    python ver_entrenamiento.py runs/ppo_20261113_1230
    python ver_entrenamiento.py --terreno colinas --dificultad 0.5

Cierra la ventana para salir (el entrenamiento sigue).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from robot_pendulo import TIPOS_TERRENO, RobotPenduloEnv
from robot_pendulo.utilidades import cargar_modelo, cargar_normalizador, kwargs_para_modelo

if hasattr(sys.stdout, "reconfigure"):  # consolas Windows sin UTF-8
    sys.stdout.reconfigure(errors="replace")


def ultimo_checkpoint(carpeta: Path) -> tuple[int, Path] | None:
    candidatos = []
    for f in (carpeta / "checkpoints").glob("modelo_*_steps.zip"):
        m = re.fullmatch(r"modelo_(\d+)_steps", f.stem)
        if m and (f.parent / f"modelo_vecnormalize_{m.group(1)}_steps.pkl").exists():
            candidatos.append((int(m.group(1)), f))
    final = carpeta / "modelo_final.zip"
    if final.exists() and (carpeta / "vecnormalize.pkl").exists():
        info = carpeta / "info_entrenamiento.json"
        pasos = json.loads(info.read_text()).get("pasos", 0) if info.exists() else 0
        candidatos.append((max(pasos, 1), final))
    return max(candidatos) if candidatos else None


def nivel_actual(carpeta: Path) -> float | None:
    for nombre in ("estado_curriculo.json", "info_entrenamiento.json"):
        f = carpeta / nombre
        try:
            datos = json.loads(f.read_text())
            return float(datos.get("nivel", datos.get("nivel_final")))
        except (OSError, ValueError, TypeError):
            continue
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("carpeta", nargs="?", help="carpeta del entrenamiento (por defecto, la más reciente en runs/)")
    p.add_argument("--terreno", choices=TIPOS_TERRENO, default=None, help="por defecto, aleatorio")
    p.add_argument("--dificultad", type=float, default=None,
                   help="por defecto, el nivel actual del currículo")
    args = p.parse_args()

    if args.carpeta:
        carpeta = Path(args.carpeta)
    else:
        runs = [d for d in Path("runs").glob("*") if d.is_dir()]
        if not runs:
            sys.exit("No hay entrenamientos en runs/. Lanza antes: python entrenar.py")
        carpeta = max(runs, key=lambda d: d.stat().st_mtime)
    print(f"Siguiendo el entrenamiento en {carpeta}")

    env, actual, modelo, normalizador, avisado = None, None, None, None, False
    try:
        while True:
            ck = ultimo_checkpoint(carpeta)
            if ck is None:
                if not avisado:
                    print("Esperando el primer checkpoint del entrenamiento...")
                    avisado = True
                time.sleep(5)
                continue
            if ck != actual:
                try:
                    modelo, normalizador = cargar_modelo(ck[1]), cargar_normalizador(ck[1])
                except Exception:  # noqa: BLE001  (archivo a medio escribir)
                    time.sleep(2)
                    continue
                actual = ck
                print(f"Cargado {ck[1].name}: {ck[0] / 1e6:.1f} M pasos de entrenamiento")
                if env is None:
                    # Terreno dibujado a 10 cm y sin sombras: el de 5 cm satura la GPU y
                    # frena el entrenamiento si este también la usa. La física no cambia.
                    env = RobotPenduloEnv(render_mode="human", resolucion_visual=0.1, sombras=False,
                                          **kwargs_para_modelo(modelo))
                env.texto_extra = f"{ck[0] / 1e6:.1f} M pasos"

            dificultad = args.dificultad
            if dificultad is None:
                dificultad = nivel_actual(carpeta)
            if dificultad is None:
                dificultad = 0.3
            opciones = dict(dificultad=dificultad)
            if args.terreno:
                opciones["tipo_terreno"] = args.terreno
            obs, info = env.reset(options=opciones)
            while True:
                o = normalizador.normalize_obs(obs) if normalizador is not None else obs
                accion, _ = modelo.predict(o, deterministic=True)
                obs, _, term, trunc, info = env.step(accion)
                if not env.visor_abierto():
                    return
                if term or trunc:
                    break
            estado = "se cayó" if info["cayo"] else "aguantó"
            print(f"  {info['tipo_terreno']:9s} dificultad {info['dificultad']:.2f}: {estado} "
                  f"({env.pasos * env.dt:.1f} s)")
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
