"""Entrena la política de equilibrio con aprendizaje por refuerzo (PPO por defecto).

El currículo empieza en terreno plano y sube la dificultad máxima del terreno
cada vez que el robot sobrevive la mayoría de los episodios.

Ejemplos:
    python entrenar.py                                  # PPO, 30 M pasos, GPU si hay
    python entrenar.py --pasos 5e6                      # prueba más corta
    python entrenar.py --dispositivo cpu                # forzar CPU
    python entrenar.py --algo sac --n-envs 8
    python entrenar.py --continuar runs/ppo_xxx/modelo_final.zip --pasos 10e6

Seguimiento en vivo:
    tensorboard --logdir runs
"""
from __future__ import annotations

import argparse
import json
import sys
import os
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from robot_pendulo import TIPOS_TERRENO, RobotPenduloEnv
from robot_pendulo.utilidades import buscar_vecnormalize, kwargs_entorno_para
from robot_pendulo.vec_lotes import VecEnvPorLotes


class Curriculo(BaseCallback):
    """Sube el nivel de terreno cuando la tasa de éxito supera `subir`
    y lo baja si cae por debajo de `bajar`.

    Éxito = no caer y seguir la velocidad ordenada (error medio < `umbral_error_vel`).
    Si solo se exige no caer, la política llega a la dificultad máxima sacrificando
    obedecer: aprende a quedarse casi quieta, que es lo más seguro.
    """

    def __init__(self, nivel_inicial: float, ventana: int = 200, subir: float = 0.8,
                 bajar: float = 0.4, paso: float = 0.05, ruta_estado: Path | None = None,
                 umbral_error_vel: float = 0.20, verbose: int = 1):
        super().__init__(verbose)
        self.nivel = nivel_inicial
        self.umbral_error_vel = umbral_error_vel
        self.errores: list[float] = []
        self.ruta_estado = ruta_estado  # lo lee ver_entrenamiento.py
        self.ventana, self.subir, self.bajar, self.paso = ventana, subir, bajar, paso
        self.resultados: list[float] = []

    def _on_training_start(self) -> None:
        self.training_env.env_method("fijar_nivel", self.nivel)
        self._guardar_estado()

    def _guardar_estado(self) -> None:
        if self.ruta_estado is not None:
            self.ruta_estado.write_text(json.dumps(dict(nivel=self.nivel, pasos=int(self.num_timesteps))))

    def _on_step(self) -> bool:
        for info, hecho in zip(self.locals["infos"], self.locals["dones"]):
            if hecho:
                err = info.get("error_vel_episodio", float("nan"))
                obedece = not np.isfinite(err) or err < self.umbral_error_vel  # sin datos = episodio corto
                self.resultados.append(1.0 if (not info.get("cayo", False) and obedece) else 0.0)
                if np.isfinite(err):
                    self.errores.append(err)
        if len(self.resultados) >= self.ventana:
            tasa = float(np.mean(self.resultados))
            self.resultados.clear()
            if self.errores:
                self.logger.record("curriculo/error_vel", float(np.mean(self.errores)))
                self.errores.clear()
            anterior = self.nivel
            if tasa >= self.subir:
                self.nivel = min(1.0, round(self.nivel + self.paso, 3))
            elif tasa < self.bajar:
                self.nivel = max(0.0, round(self.nivel - self.paso, 3))
            if self.nivel != anterior:
                self.training_env.env_method("fijar_nivel", self.nivel)
                self._guardar_estado()
                if self.verbose:
                    print(f"[currículo] {self.num_timesteps:>10,} pasos | éxito {tasa:.0%} "
                          f"-> nivel {anterior:.2f} -> {self.nivel:.2f}")
            self.logger.record("curriculo/tasa_exito", tasa)
        self.logger.record("curriculo/nivel", self.nivel)
        return True


class EvaluacionPorTerreno(BaseCallback):
    """Cada `cada` pasos evalúa la política (determinista) en cada tipo de terreno
    con dificultad 1 y guarda la tasa de éxito en TensorBoard."""

    def __init__(self, cada: int, episodios: int = 3, verbose: int = 1, **kwargs_entorno):
        super().__init__(verbose)
        self.cada, self.episodios = cada, episodios
        self.env = RobotPenduloEnv(**kwargs_entorno)
        self._proxima = cada

    def _on_step(self) -> bool:
        if self.num_timesteps < self._proxima:
            return True
        self._proxima += self.cada
        normalizar = getattr(self.training_env, "normalize_obs", lambda o: o)
        resumen = []
        for i, tipo in enumerate(TIPOS_TERRENO):
            exitos, inclinaciones, errores_vel = 0, [], []
            for ep in range(self.episodios):
                obs, info = self.env.reset(seed=10_000 + 100 * i + ep,
                                           options=dict(tipo_terreno=tipo, dificultad=1.0))
                while True:
                    accion, _ = self.model.predict(normalizar(obs), deterministic=True)
                    obs, _, term, trunc, info = self.env.step(accion)
                    inclinaciones.append(info["inclinacion"])
                    errores_vel.append(abs(info["vel_adelante"] - info["vel_cmd"]))
                    if term or trunc:
                        exitos += int(not info["cayo"])
                        break
            tasa = exitos / self.episodios
            self.logger.record(f"eval/exito_{tipo}", tasa)
            self.logger.record(f"eval/inclinacion_media_{tipo}", float(np.degrees(np.mean(inclinaciones))))
            self.logger.record(f"eval/error_vel_{tipo}", float(np.mean(errores_vel)))
            resumen.append(f"{tipo} {tasa:.0%} ({np.mean(errores_vel):.2f} m/s)")
        if self.verbose:
            print(f"[evaluación dificultad 1] {self.num_timesteps:>10,} pasos | " + " · ".join(resumen))
        return True


if hasattr(sys.stdout, "reconfigure"):  # consolas Windows sin UTF-8
    sys.stdout.reconfigure(errors="replace")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    p.add_argument("--pasos", type=float, default=30e6, help="pasos totales de entrenamiento")
    p.add_argument("--n-envs", type=int, default=48, help="robots simulados en paralelo")
    p.add_argument("--n-procesos", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                   help="procesos de simulación (por defecto, núcleos físicos)")
    p.add_argument("--dispositivo", choices=["auto", "cuda", "cpu"], default="auto",
                   help="dónde se entrena la red: auto = GPU NVIDIA si PyTorch la detecta")
    p.add_argument("--nivel-inicial", type=float, default=None, help="nivel de terreno inicial [0-1]")
    p.add_argument("--sin-curriculo", action="store_true", help="entrenar siempre con dificultad máxima")
    p.add_argument("--modo-accion", choices=["independiente", "diferencial"], default="independiente")
    p.add_argument("--eval-cada", type=float, default=2e6, help="pasos entre evaluaciones (0 = nunca)")
    p.add_argument("--checkpoint-cada", type=float, default=1e6,
                   help="pasos entre checkpoints (ver_entrenamiento.py carga el último)")
    p.add_argument("--continuar", type=str, default=None, help="modelo .zip desde el que seguir")
    p.add_argument("--salida", type=str, default=None, help="carpeta de resultados")
    p.add_argument("--gamma", type=float, default=0.99, help="factor de descuento")
    p.add_argument("--umbral-error-vel", type=float, default=0.20,
                   help="m/s: un episodio solo cuenta como éxito para el currículo si, además de "
                        "no caer, su error medio de velocidad está por debajo de esto")
    p.add_argument("--semilla", type=int, default=0)
    args = p.parse_args()

    salida = Path(args.salida or f"runs/{args.algo}_{time.strftime('%Y%m%d_%H%M%S')}")
    salida.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    dispositivo = args.dispositivo
    if dispositivo == "auto":
        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    elif dispositivo == "cuda" and not torch.cuda.is_available():
        sys.exit("PyTorch no detecta CUDA. Instala la versión con CUDA (ver README) o usa --dispositivo cpu.")
    nombre_disp = torch.cuda.get_device_name(0) if dispositivo == "cuda" else "CPU"

    # Nivel inicial del currículo
    nivel = args.nivel_inicial
    if nivel is None and args.continuar:
        info_prev = Path(args.continuar).parent / "info_entrenamiento.json"
        if info_prev.exists():
            nivel = json.loads(info_prev.read_text()).get("nivel_final")
    if nivel is None:
        nivel = 1.0 if args.sin_curriculo else 0.0

    # Entornos en paralelo
    if args.n_procesos > 1:
        clase_vec, kw_vec = VecEnvPorLotes, dict(n_procesos=args.n_procesos)
    else:
        clase_vec, kw_vec = DummyVecEnv, None
    kw_entorno = dict(modo_accion=args.modo_accion)
    if args.continuar:  # el entorno debe coincidir con el del modelo (los antiguos no tienen orden de giro)
        from stable_baselines3.common.save_util import load_from_zip_file
        datos, _, _ = load_from_zip_file(args.continuar, device="cpu")
        kw_entorno = kwargs_entorno_para(int(datos["observation_space"].shape[0]),
                                         int(datos["action_space"].shape[0]))
    venv = make_vec_env(RobotPenduloEnv, n_envs=args.n_envs, seed=args.semilla, vec_env_cls=clase_vec,
                        vec_env_kwargs=kw_vec, env_kwargs=dict(kw_entorno, nivel=nivel))
    if args.continuar:
        ruta_vn = buscar_vecnormalize(args.continuar)
        venv = VecNormalize.load(str(ruta_vn), venv) if ruta_vn else VecNormalize(venv)
        venv.training = True
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=(args.algo == "ppo"), clip_obs=10.0, gamma=args.gamma)

    # Algoritmo
    if args.continuar:
        Clase = PPO if args.algo == "ppo" else SAC
        modelo = Clase.load(args.continuar, env=venv, device=dispositivo, tensorboard_log=str(salida / "tb"))
    elif args.algo == "ppo":
        n_steps = 256  # por robot: 48 robots x 256 = 12.288 muestras por actualización
        modelo = PPO(
            "MlpPolicy", venv, n_steps=n_steps, batch_size=max(64, n_steps * args.n_envs // 4),
            n_epochs=5, learning_rate=3e-4, gamma=args.gamma, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.0, max_grad_norm=1.0, seed=args.semilla, device=dispositivo, verbose=1,
            tensorboard_log=str(salida / "tb"),
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                               activation_fn=torch.nn.ELU, log_std_init=-1.0),
        )
    else:
        modelo = SAC(
            "MlpPolicy", venv, learning_rate=3e-4, buffer_size=1_000_000, batch_size=512,
            learning_starts=20_000, train_freq=1, gradient_steps=max(1, args.n_envs // 2),
            gamma=args.gamma, tau=0.005, seed=args.semilla, device=dispositivo, verbose=1,
            tensorboard_log=str(salida / "tb"),
            policy_kwargs=dict(net_arch=[256, 256]),
        )

    callbacks = [CheckpointCallback(save_freq=max(1, int(args.checkpoint_cada) // args.n_envs),
                                    save_path=str(salida / "checkpoints"), name_prefix="modelo",
                                    save_vecnormalize=True)]
    curriculo = None
    if not args.sin_curriculo:
        curriculo = Curriculo(nivel_inicial=nivel, ruta_estado=salida / "estado_curriculo.json",
                              umbral_error_vel=args.umbral_error_vel)
        callbacks.append(curriculo)
    if args.eval_cada > 0:
        callbacks.append(EvaluacionPorTerreno(cada=int(args.eval_cada), **kw_entorno))

    print(f"Entrenando {args.algo.upper()} en {nombre_disp} · {args.n_envs} robots en "
          f"{args.n_procesos} procesos · nivel inicial {nivel:.2f}")
    print(f"Resultados en: {salida.resolve()}")
    print(f"TensorBoard:   tensorboard --logdir {salida / 'tb'}")
    print(f"Verlo en 3D:   python ver_entrenamiento.py {salida}")
    t0 = time.time()
    try:
        modelo.learn(total_timesteps=int(args.pasos), callback=CallbackList(callbacks),
                     reset_num_timesteps=not args.continuar, progress_bar=True)
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido: guardando el modelo actual...")
    finally:
        modelo.save(salida / "modelo_final")
        venv.save(str(salida / "vecnormalize.pkl"))
        nivel_final = curriculo.nivel if curriculo else nivel
        (salida / "info_entrenamiento.json").write_text(json.dumps(dict(
            algo=args.algo, pasos=int(modelo.num_timesteps), nivel_final=nivel_final,
            modo_accion=args.modo_accion, minutos=round((time.time() - t0) / 60, 1),
        ), indent=2))
        venv.close()
        print(f"Modelo guardado en {salida / 'modelo_final.zip'} (nivel de terreno alcanzado: {nivel_final:.2f})")
        print(f"Para verlo:  python evaluar.py {salida / 'modelo_final.zip'}")


if __name__ == "__main__":
    main()
