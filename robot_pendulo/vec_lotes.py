"""VecEnv que simula varios robots en cada proceso.

`SubprocVecEnv` de Stable-Baselines3 usa un proceso por entorno y envía un mensaje
por entorno y paso. En Windows cada mensaje entre procesos cuesta del orden de
0,1 ms, más de lo que tarda en simularse un paso del robot. Agrupando K robots por
proceso se envía un único mensaje por proceso y paso, y la CPU se dedica a simular.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import signal
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper, VecEnv, VecEnvIndices
from stable_baselines3.common.vec_env.patch_gym import _patch_env


def _trabajador(remoto, remoto_padre, fabricas: CloudpickleWrapper) -> None:
    from stable_baselines3.common.env_util import is_wrapped

    # Ctrl+C solo debe interrumpir al proceso principal, que luego cierra los trabajadores
    # de forma ordenada (y así se guarda el modelo).
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    remoto_padre.close()
    envs = [_patch_env(f()) for f in fabricas.var]
    n = len(envs)
    while True:
        try:
            cmd, datos = remoto.recv()
        except (EOFError, KeyboardInterrupt):
            break
        try:
            if cmd == "step":
                obs, recompensas = [None] * n, np.zeros(n)
                hechos, infos, infos_reset = np.zeros(n, dtype=bool), [None] * n, [{}] * n
                for k, (env, accion) in enumerate(zip(envs, datos)):
                    o, r, term, trunc, info = env.step(accion)
                    hecho = term or trunc
                    info["TimeLimit.truncated"] = trunc and not term
                    if hecho:
                        info["terminal_observation"] = o
                        o, info_reset = env.reset()
                        infos_reset[k] = info_reset
                    obs[k], recompensas[k], hechos[k], infos[k] = o, r, hecho, info
                remoto.send((np.stack(obs), recompensas, hechos, infos, infos_reset))
            elif cmd == "reset":
                semillas, opciones = datos
                res = [env.reset(seed=s, **({"options": o} if o else {}))
                       for env, s, o in zip(envs, semillas, opciones)]
                remoto.send((np.stack([r[0] for r in res]), [r[1] for r in res]))
            elif cmd == "get_spaces":
                remoto.send((envs[0].observation_space, envs[0].action_space))
            elif cmd == "env_method":
                nombre, args, kwargs, locales = datos
                remoto.send([envs[k].get_wrapper_attr(nombre)(*args, **kwargs) for k in locales])
            elif cmd == "get_attr":
                nombre, locales = datos
                remoto.send([envs[k].get_wrapper_attr(nombre) for k in locales])
            elif cmd == "has_attr":
                nombre, locales = datos
                res = []
                for k in locales:
                    try:
                        envs[k].get_wrapper_attr(nombre)
                        res.append(True)
                    except AttributeError:
                        res.append(False)
                remoto.send(res)
            elif cmd == "set_attr":
                nombre, valor, locales = datos
                for k in locales:
                    setattr(envs[k], nombre, valor)
                remoto.send([None] * len(locales))
            elif cmd == "is_wrapped":
                clase, locales = datos
                remoto.send([is_wrapped(envs[k], clase) for k in locales])
            elif cmd == "render":
                remoto.send([env.render() for env in envs])
            elif cmd == "close":
                for env in envs:
                    env.close()
                remoto.close()
                break
            else:
                raise NotImplementedError(cmd)
        except (EOFError, BrokenPipeError, KeyboardInterrupt):
            break


class VecEnvPorLotes(VecEnv):
    """Como `SubprocVecEnv`, pero repartiendo los entornos en `n_procesos` procesos."""

    def __init__(self, env_fns: list[Callable[[], gym.Env]], n_procesos: int | None = None,
                 start_method: str | None = None):
        n_envs = len(env_fns)
        n_procesos = max(1, min(n_procesos or max(1, (os.cpu_count() or 2) // 2), n_envs))
        self.grupos = [g.tolist() for g in np.array_split(np.arange(n_envs), n_procesos)]
        self._proceso_de, self._local_de = {}, {}
        for p, grupo in enumerate(self.grupos):
            for local, i in enumerate(grupo):
                self._proceso_de[i], self._local_de[i] = p, local

        if start_method is None:
            start_method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
        ctx = mp.get_context(start_method)
        self.remotes, remotos_trabajo = zip(*[ctx.Pipe() for _ in range(n_procesos)])
        self.processes = []
        for rt, r, grupo in zip(remotos_trabajo, self.remotes, self.grupos):
            fabricas = CloudpickleWrapper([env_fns[i] for i in grupo])
            proceso = ctx.Process(target=_trabajador, args=(rt, r, fabricas), daemon=True)
            proceso.start()
            self.processes.append(proceso)
            rt.close()
        self.waiting = False
        self.closed = False
        self.remotes[0].send(("get_spaces", None))
        obs_space, act_space = self.remotes[0].recv()
        super().__init__(n_envs, obs_space, act_space)

    # ------------------------------------------------------------------ pasos
    def step_async(self, actions: np.ndarray) -> None:
        for remoto, grupo in zip(self.remotes, self.grupos):
            remoto.send(("step", actions[grupo[0]:grupo[-1] + 1]))
        self.waiting = True

    def step_wait(self):
        res = [r.recv() for r in self.remotes]
        self.waiting = False
        self.reset_infos = [i for r in res for i in r[4]]
        return (np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res]),
                np.concatenate([r[2] for r in res]), [i for r in res for i in r[3]])

    def reset(self):
        for remoto, grupo in zip(self.remotes, self.grupos):
            remoto.send(("reset", ([self._seeds[i] for i in grupo], [self._options[i] for i in grupo])))
        res = [r.recv() for r in self.remotes]
        self.reset_infos = [i for r in res for i in r[1]]
        self._reset_seeds()
        self._reset_options()
        return np.concatenate([r[0] for r in res])

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for r in self.remotes:
            try:
                if self.waiting:
                    r.recv()
                r.send(("close", None))
            except (EOFError, OSError):
                pass
        for p in self.processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        for r in self.remotes:
            r.close()

    # ------------------------------------------------------------------ utilidades
    def _a_procesos(self, cmd: str, datos: Callable[[list[int]], Any], indices: VecEnvIndices) -> list[Any]:
        indices = list(self._get_indices(indices))
        por_proceso: dict[int, list[int]] = {}
        for i in indices:
            por_proceso.setdefault(self._proceso_de[i], []).append(i)
        for p, lista in por_proceso.items():
            self.remotes[p].send((cmd, datos([self._local_de[i] for i in lista])))
        resultado = {}
        for p, lista in por_proceso.items():
            resultado.update(zip(lista, self.remotes[p].recv()))
        return [resultado[i] for i in indices]

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs):
        return self._a_procesos("env_method", lambda loc: (method_name, method_args, method_kwargs, loc), indices)

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None):
        return self._a_procesos("get_attr", lambda loc: (attr_name, loc), indices)

    def has_attr(self, attr_name: str) -> bool:
        return all(self._a_procesos("has_attr", lambda loc: (attr_name, loc), None))

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        self._a_procesos("set_attr", lambda loc: (attr_name, value, loc), indices)

    def env_is_wrapped(self, wrapper_class, indices: VecEnvIndices = None):
        return self._a_procesos("is_wrapped", lambda loc: (wrapper_class, loc), indices)

    def get_images(self):
        for r in self.remotes:
            r.send(("render", None))
        return [img for r in self.remotes for img in r.recv()]
