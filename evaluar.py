"""Visualiza y mide una política entrenada.

Ejemplos:
    python evaluar.py runs/ppo_xxx/modelo_final.zip                         # ventana 3D
    python evaluar.py runs/ppo_xxx/modelo_final.zip --terreno rugoso --dificultad 1
    python evaluar.py runs/ppo_xxx/modelo_final.zip --video demo.mp4 --episodios 3
    python evaluar.py runs/ppo_xxx/modelo_final.zip --sin-ventana --episodios 60   # solo estadísticas
    python evaluar.py --pd --sin-ventana --episodios 60     # controlador PD clásico, para comparar

Terreno propio (mapa de alturas visto desde arriba: arriba = norte, derecha = este):
    python evaluar.py runs/ppo_xxx/modelo_final.zip --mapa terrenos/pista_ejemplo.png --tamano 24 --altura-blanco 0.6 --inicio -10 0 --vel-cmd 0.4
    python evaluar.py runs/ppo_xxx/modelo_final.zip --mapa mi_terreno.npy     # alturas en metros

En la ventana: flechas ↑/↓ cambian la velocidad ordenada y ←/→ giran el rumbo 15°
(modelos con orden de giro); doble clic sobre una pieza + Ctrl + arrastrar con el
botón derecho para empujar el robot.

Si la ventana va a saltos (portátil sin gráfica dedicada), usa --calidad baja. Solo
cambia el dibujo: la física y los resultados son los mismos.

Con --vel-cmd o --mapa el robot mantiene el rumbo inicial (va recto); sin ellos recibe
órdenes aleatorias que cambian cada 5-10 s, como en el entrenamiento.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np

from robot_pendulo import TIPOS_TERRENO, RobotPenduloEnv
from robot_pendulo.terreno import cargar_mapa_alturas
from robot_pendulo.utilidades import (CALIDADES, cargar_modelo, cargar_normalizador, controlador_pd,
                                      kwargs_para_modelo)


if hasattr(sys.stdout, "reconfigure"):  # consolas Windows sin UTF-8
    sys.stdout.reconfigure(errors="replace")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("modelo", nargs="?", help="ruta al modelo .zip entrenado")
    p.add_argument("--pd", action="store_true", help="usar el controlador PD clásico en vez de un modelo")
    p.add_argument("--episodios", type=int, default=5)
    p.add_argument("--terreno", choices=TIPOS_TERRENO, default=None, help="por defecto, aleatorio")
    p.add_argument("--dificultad", type=float, default=1.0)
    p.add_argument("--vel-cmd", type=float, default=None, help="velocidad ordenada fija (m/s, máx. 0,5)")
    p.add_argument("--duracion", type=float, default=20.0, help="segundos por episodio")
    p.add_argument("--video", type=str, default=None, help="guardar vídeo .mp4 en vez de abrir ventana")
    p.add_argument("--sin-ventana", action="store_true", help="solo calcular estadísticas")
    p.add_argument("--calidad", choices=tuple(CALIDADES), default=None,
                   help="detalle del dibujo 3D (por defecto: media en la ventana, alta en vídeo). "
                        "baja = para portátiles sin gráfica dedicada; no cambia la física")
    p.add_argument("--sin-perturbaciones", action="store_true",
                   help="sin empujones, ruido ni aleatorización de dinámica")
    p.add_argument("--estocastico", action="store_true", help="muestrear acciones en vez de usar la media")
    p.add_argument("--semilla", type=int, default=12345)
    g = p.add_argument_group("terreno propio")
    g.add_argument("--mapa", type=str, default=None,
                   help="mapa de alturas: imagen en grises (.png/.jpg) o matriz en metros (.npy/.csv)")
    g.add_argument("--altura-blanco", type=float, default=0.5,
                   help="metros que representa el blanco de la imagen (negro = 0 m)")
    g.add_argument("--tamano", type=float, default=None,
                   help="metros que mide el lado mayor del mapa (por defecto, 30 m: todo el terreno)")
    g.add_argument("--inicio", type=float, nargs=2, metavar=("X", "Y"), default=None,
                   help="posición inicial del robot en metros (por defecto, aleatoria cerca del centro)")
    g.add_argument("--rumbo", type=float, default=None,
                   help="hacia dónde mira el robot, en grados (0 = este/derecha del mapa, 90 = norte/arriba)")
    args = p.parse_args()
    if not args.pd and not args.modelo:
        p.error("indica la ruta del modelo o usa --pd")

    modelo = normalizador = None
    if not args.pd:
        import torch

        # La red es pequeña: con un solo hilo va igual de rápido y no le quita CPU
        # a la física ni a la ventana 3D (importante en portátiles con pocos núcleos).
        torch.set_num_threads(1)
        modelo = cargar_modelo(args.modelo)
        normalizador = cargar_normalizador(args.modelo)
    kw_modelo = {} if args.pd else kwargs_para_modelo(modelo)

    modo_render = "rgb_array" if args.video else (None if args.sin_ventana else "human")
    extra = dict(ruido_obs=0.0, aleatorizar_dinamica=False, empujones=False) if args.sin_perturbaciones else {}
    if modo_render is not None:
        extra.update(CALIDADES[args.calidad or ("alta" if args.video else "media")])
    env = RobotPenduloEnv(render_mode=modo_render, nivel=args.dificultad, duracion_episodio=args.duracion,
                          **kw_modelo, **extra)

    alturas = None
    if args.mapa:
        alturas = cargar_mapa_alturas(args.mapa, args.altura_blanco)
        lado = args.tamano or 2 * env.cfg.semitamano_terreno
        desnivel = float(alturas.max() - alturas.min())
        print(f"Terreno propio: {args.mapa} ({alturas.shape[1]}x{alturas.shape[0]} puntos, "
              f"{lado / max(alturas.shape[0] - 1, alturas.shape[1] - 1) * 100:.1f} cm por punto, "
              f"lado {lado:.1f} m, desnivel {desnivel:.2f} m)")

    escritor = None
    if args.video:
        import imageio.v2 as imageio
        escritor = imageio.get_writer(args.video, fps=25, quality=7, macro_block_size=8)

    stats = defaultdict(lambda: defaultdict(list))
    try:
        for ep in range(args.episodios):
            opciones = dict(dificultad=args.dificultad)
            if args.terreno:
                opciones["tipo_terreno"] = args.terreno
            if args.vel_cmd is not None:
                opciones["vel_cmd"] = args.vel_cmd
            if alturas is not None:
                opciones["alturas"] = alturas
                opciones["tamano_alturas"] = args.tamano
            if args.inicio is not None:
                opciones["posicion"] = tuple(args.inicio)
            if args.rumbo is not None:
                opciones["rumbo"] = np.radians(args.rumbo)
            if args.vel_cmd is not None or alturas is not None:
                opciones["mantener_rumbo"] = True
            obs, info = env.reset(seed=args.semilla + ep, options=opciones)
            tipo, retorno, incl, err_v, pasos = info["tipo_terreno"], 0.0, [], [], 0
            x0, y0, psi0 = info["x"], info["y"], info["rumbo"]
            while True:
                if args.pd:
                    accion = np.full(4, controlador_pd(info))
                else:
                    o = normalizador.normalize_obs(obs) if normalizador is not None else obs
                    accion, _ = modelo.predict(o, deterministic=not args.estocastico)
                obs, r, term, trunc, info = env.step(accion)
                retorno += r
                pasos += 1
                incl.append(np.degrees(info["inclinacion"]))
                err_v.append(abs(info["vel_adelante"] - info["vel_cmd"]))
                if escritor is not None and pasos % 2 == 0:
                    escritor.append_data(env.render())
                if modo_render == "human" and not env.visor_abierto():
                    raise KeyboardInterrupt
                if term or trunc:
                    break
            s = stats[tipo]
            s["exito"].append(0.0 if info["cayo"] else 1.0)
            s["segundos"].append(pasos * env.dt)
            s["incl_media"].append(float(np.mean(incl)))
            s["incl_max"].append(float(np.max(incl)))
            s["err_vel"].append(float(np.mean(err_v)))
            # Avance en la dirección inicial y desvío lateral
            dx, dy = info["x"] - x0, info["y"] - y0
            avance = dx * np.cos(psi0) + dy * np.sin(psi0)
            desvio = -dx * np.sin(psi0) + dy * np.cos(psi0)
            s["avance"].append(float(avance))
            estado = "CAYÓ" if info["cayo"] else "ok"
            print(f"Episodio {ep + 1:3d} | {tipo:13s} dif {info['dificultad']:.2f} | {estado:4s} "
                  f"| {pasos * env.dt:5.1f} s | inclinación media {np.mean(incl):4.1f}° máx {np.max(incl):4.1f}° "
                  f"| error vel {np.mean(err_v):.2f} m/s | avance {avance:+5.1f} m desvío {desvio:+5.1f} m")
    except KeyboardInterrupt:
        print("Detenido.")
    finally:
        if escritor is not None:
            escritor.close()
            print(f"Vídeo guardado en {args.video}")
        env.close()

    if stats:
        print("\nResumen por terreno")
        print(f"{'terreno':13s} {'episodios':>9s} {'éxito':>6s} {'incl. media':>11s} {'incl. máx':>9s} "
              f"{'error vel':>9s} {'avance':>8s}")
        todos = []
        for tipo in (*TIPOS_TERRENO, "personalizado"):
            if tipo not in stats:
                continue
            s = stats[tipo]
            todos += s["exito"]
            print(f"{tipo:13s} {len(s['exito']):9d} {np.mean(s['exito']):6.0%} {np.mean(s['incl_media']):10.1f}° "
                  f"{np.mean(s['incl_max']):8.1f}° {np.mean(s['err_vel']):7.2f} m/s {np.mean(s['avance']):+6.1f} m")
        print(f"{'TOTAL':13s} {len(todos):9d} {np.mean(todos):6.0%}")


if __name__ == "__main__":
    main()
