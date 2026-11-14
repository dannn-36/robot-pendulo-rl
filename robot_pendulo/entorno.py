"""Entorno Gymnasium: robot de 4 ruedas que debe mantener un péndulo invertido
en equilibrio sobre terrenos variados.

Observación (todo lo que un robot real puede medir con IMU + encoders):
    [0:3]   gravedad proyectada en el chasis (IMU)
    [3:6]   velocidad angular del chasis (giroscopio)
    [6:9]   velocidad lineal del chasis en su propio marco (odometría + IMU)
    [9:11]  ángulo y velocidad del pivote del péndulo (encoder)
    [11:13] inclinación del péndulo respecto a la vertical: cabeceo, lateral
    [13:15] velocidad angular del péndulo: cabeceo, lateral
    [15:19] velocidad de las 4 ruedas (encoders)
    [19:19+n] acción anterior
    [19+n]  velocidad de avance ordenada (m/s)
    [20+n]  velocidad de giro ordenada (rad/s, +izquierda); solo con comando_giro=True

Órdenes: velocidad de avance y giro. El giro se da como velocidad de giro constante o
como un rumbo a mantener; en ese caso el entorno calcula la velocidad de giro ordenada
con un control proporcional, giro = clip(k·(rumbo_ordenado − rumbo)), igual que haría el
robot real con su brújula/IMU. Durante el entrenamiento las órdenes cambian cada 5–10 s.

Acción: par normalizado [-1, 1] para cada rueda (modo "independiente", 4 valores)
o para cada lado izquierda/derecha (modo "diferencial", 2 valores).
"""
from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass, field

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .modelo import NOMBRES_RUEDAS, ParamsRobot, construir_mjcf
from .terreno import TIPOS_TERRENO, GeneradorTerreno


def _pesos_por_defecto() -> dict:
    return dict(
        vivo=0.5,          # bono por cada paso sin caer
        vertical=1.0,      # péndulo vertical (respecto al mundo, no al chasis)
        velocidad=1.5,     # seguir la velocidad de avance ordenada
        giro=0.5,          # seguir la velocidad de giro ordenada
        vel_pendulo=0.02,  # penaliza oscilaciones del péndulo
        accion=0.01,       # penaliza par excesivo (consumo)
        suavidad=0.05,     # penaliza cambios bruscos de par
        caida=10.0,        # penalización única al caer
    )


@dataclass
class ConfigEntorno:
    # --- Simulación
    dt_fisica: float = 0.005            # s (200 Hz)
    pasos_por_accion: int = 4           # control a 50 Hz
    duracion_episodio: float = 20.0     # s
    # --- Robot
    robot: ParamsRobot = field(default_factory=ParamsRobot)
    modo_accion: str = "independiente"  # "independiente" (4 ruedas) | "diferencial" (izq/der)
    # --- Terreno
    semitamano_terreno: float = 15.0    # m (el terreno mide 30 x 30 m)
    resolucion_terreno: float = 0.05    # m entre puntos del campo de alturas
    altura_max_terreno: float = 3.0     # m
    tipos_terreno: tuple = TIPOS_TERRENO
    pendiente_max_grados: float = 15.0  # con dificultad 1
    rugosidad_max: float = 0.03         # m de amplitud de baches con dificultad 1
    escalon_max: float = 0.05           # m de altura de escalón con dificultad 1
    episodios_por_terreno: int = 8      # se reutiliza el terreno (otra posición) para ir más rápido
    radio_aparicion: float = 3.0        # m alrededor del centro donde aparece el robot
    # --- Currículo: dificultad máxima del terreno en [0, 1].
    # Cada terreno se genera con dificultad = nivel * sqrt(U(0,1)).
    nivel: float = 1.0
    # --- Tarea
    vel_cmd_max: float = 0.5            # m/s
    prob_cmd_cero: float = 0.2          # probabilidad de pedir quedarse quieto
    comando_giro: bool = True           # False = modelos antiguos (sin orden de giro en la observación)
    giro_cmd_max: float = 0.6           # rad/s (girar derrapando 4 ruedas cuesta mucho con rozamiento alto)
    ganancia_rumbo: float = 1.0         # (rad/s) por rad de error de rumbo
    remuestrear_cmd: tuple = (5.0, 10.0)  # s entre cambios de órdenes (entrenamiento)
    inclinacion_max: float = 0.7        # rad (≈40°): si el péndulo pasa de aquí, cae
    # --- Robustez (sim-to-real)
    aleatorizar_dinamica: bool = True
    ruido_obs: float = 1.0              # multiplicador del ruido de sensores (0 = sin ruido)
    empujones: bool = True
    empujon_chasis_max: float = 25.0    # N
    empujon_pendulo_max: float = 3.0    # N
    # --- Recompensa
    pesos: dict = field(default_factory=_pesos_por_defecto)
    # Anchura de cada término gaussiano: exp(-(error / tolerancia)^2).
    tolerancia_vertical: float = 0.2    # rad
    tolerancia_vel: float = 0.32        # m/s
    tolerancia_giro: float = 0.5        # rad/s
    # --- Visualización
    ancho_video: int = 960
    alto_video: int = 540
    camara: str = "seguimiento"
    resolucion_visual: float | None = None  # m; p. ej. 0.1 dibuja un terreno más ligero (ver_entrenamiento.py)
    sombras: bool = True


# Desviación típica del ruido de cada bloque de la observación (con ruido_obs = 1).
_RUIDO = dict(grav=0.02, gyro=0.05, vel=0.05, piv=0.005, dpiv=0.05, incl=0.01, dincl=0.05, ruedas=0.2)


class RobotPenduloEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str | None = None, config: ConfigEntorno | None = None, **kwargs):
        super().__init__()
        cfg = config if config is not None else ConfigEntorno()
        self.cfg = dataclasses.replace(cfg, **kwargs)
        self.render_mode = render_mode

        c = self.cfg
        n = int(round(2 * c.semitamano_terreno / c.resolucion_terreno)) + 1
        self.terreno = GeneradorTerreno(n, c.semitamano_terreno, c.altura_max_terreno,
                                        c.pendiente_max_grados, c.rugosidad_max, c.escalon_max)
        # Terreno visual reducido: se toma 1 de cada `paso_visual` puntos del terreno de la física.
        self.paso_visual = 0
        n_visual = None
        if c.resolucion_visual:
            k = max(1, int(round(c.resolucion_visual / c.resolucion_terreno)))
            while (n - 1) % k:
                k -= 1
            if k > 1:
                self.paso_visual, n_visual = k, (n - 1) // k + 1
        xml = construir_mjcf(c.robot, n, c.semitamano_terreno, c.altura_max_terreno, c.dt_fisica, n_visual)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.dt = c.dt_fisica * c.pasos_por_accion
        self.max_pasos = int(round(c.duracion_episodio / self.dt))

        m = self.model
        nid = lambda t, s: mujoco.mj_name2id(m, t, s)  # noqa: E731
        self.id_chasis = nid(mujoco.mjtObj.mjOBJ_BODY, "chasis")
        self.id_pendulo = nid(mujoco.mjtObj.mjOBJ_BODY, "pendulo")
        self.id_hfield = nid(mujoco.mjtObj.mjOBJ_HFIELD, "terreno")
        self.id_geom_terreno = nid(mujoco.mjtObj.mjOBJ_GEOM, "terreno")
        self.ids_geom_ruedas = [nid(mujoco.mjtObj.mjOBJ_GEOM, f"rueda_{r}") for r in NOMBRES_RUEDAS]
        j_piv = nid(mujoco.mjtObj.mjOBJ_JOINT, "pivote")
        self.qpos_piv, self.qvel_piv = m.jnt_qposadr[j_piv], m.jnt_dofadr[j_piv]
        self.qvel_ruedas = np.array([m.jnt_dofadr[nid(mujoco.mjtObj.mjOBJ_JOINT, f"rueda_{r}")]
                                     for r in NOMBRES_RUEDAS])
        self.adr_hfield = m.hfield_adr[self.id_hfield]
        # El terreno que se dibuja (el de la física, o su copia reducida si existe)
        self.id_hfield_dibujo = self.id_hfield
        if self.paso_visual:
            self.id_hfield_dibujo = nid(mujoco.mjtObj.mjOBJ_HFIELD, "terreno_visual")
        if not c.sombras:
            m.light_castshadow[:] = 0
        self.ruedas_local = np.array([m.body_pos[nid(mujoco.mjtObj.mjOBJ_BODY, f"rueda_{r}")]
                                      for r in NOMBRES_RUEDAS])

        # Valores nominales para la aleatorización de dinámica.
        self._nominal = dict(
            body_mass=m.body_mass.copy(), body_inertia=m.body_inertia.copy(),
            geom_friction=m.geom_friction.copy(), dof_damping=m.dof_damping.copy(),
            gainprm=m.actuator_gainprm.copy(), biasprm=m.actuator_biasprm.copy(),
            forcerange=m.actuator_forcerange.copy(),
        )

        if c.modo_accion == "independiente":
            self.n_acciones = 4
            self._mapa_accion = np.arange(4)
        elif c.modo_accion == "diferencial":
            self.n_acciones = 2
            self._mapa_accion = np.array([0, 1, 0, 1])  # di, dd, ti, td -> izq, der, izq, der
        else:
            raise ValueError("modo_accion debe ser 'independiente' o 'diferencial'")
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_acciones,), dtype=np.float32)
        n_obs = 20 + self.n_acciones + (1 if c.comando_giro else 0)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_obs,), dtype=np.float32)
        self._escala_ruido = np.concatenate([
            np.full(3, _RUIDO["grav"]), np.full(3, _RUIDO["gyro"]), np.full(3, _RUIDO["vel"]),
            [_RUIDO["piv"], _RUIDO["dpiv"]], np.full(2, _RUIDO["incl"]), np.full(2, _RUIDO["dincl"]),
            np.full(4, _RUIDO["ruedas"]), np.zeros(n_obs - 19),
        ]).astype(np.float32) * c.ruido_obs

        # Estado del episodio
        self.tipo_terreno = "plano"
        self.dificultad = 0.0
        self.vel_cmd = 0.0
        self.giro_cmd = 0.0
        self.rumbo_cmd: float | None = None   # None = se ordena una velocidad de giro fija
        self._comandos_fijos = False
        self._proximo_cmd = 0
        self._ultimo_cambio_cmd = 0
        self._suma_err_vel = 0.0
        self._n_err_vel = 0
        self.pasos = 0
        self._episodios_en_terreno = 10 ** 9
        self._accion_previa = np.zeros(self.n_acciones)
        self._empujon_restante = 0
        self._proximo_empujon = 0
        self._cuerpo_empujado = self.id_chasis
        self._hfield_pendiente_subir = True
        self._renderer = None
        self._visor = None
        self._t_ultimo_frame = None
        self._vel_c = np.zeros(6)
        self._vel_p = np.zeros(6)
        self._obs = np.zeros(n_obs)
        self._ultimo_estado: dict = {}
        self.texto_extra = ""  # línea adicional en la ventana 3D (p. ej. el checkpoint cargado)

    # ================================================================== API pública
    def fijar_nivel(self, nivel: float) -> None:
        """Ajusta la dificultad máxima del terreno (lo usa el currículo)."""
        self.cfg.nivel = float(np.clip(nivel, 0.0, 1.0))
        self._episodios_en_terreno = 10 ** 9  # fuerza terreno nuevo en el próximo reset

    def fijar_vel_cmd(self, v: float) -> None:
        self.vel_cmd = float(np.clip(v, -self.cfg.vel_cmd_max, self.cfg.vel_cmd_max))

    def girar_rumbo(self, delta: float) -> None:
        """Cambia el rumbo ordenado `delta` radianes (+ = a la izquierda)."""
        if not self.cfg.comando_giro:
            return
        base = self.rumbo_cmd if self.rumbo_cmd is not None else self._rumbo_actual()
        self.rumbo_cmd = _envolver(base + delta)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        opciones = options or {}
        c, rng = self.cfg, self.np_random

        # --- Terreno (se regenera cada `episodios_por_terreno` o si se pide uno concreto)
        forzar = ("tipo_terreno" in opciones or "dificultad" in opciones or "alturas" in opciones
                  or seed is not None)
        if forzar or self._episodios_en_terreno >= c.episodios_por_terreno:
            if "alturas" in opciones:
                # Terreno propio: matriz en metros (ver terreno.cargar_mapa_alturas).
                # La dificultad aquí solo regula la fuerza de los empujones.
                self.tipo_terreno = "personalizado"
                self.dificultad = float(opciones.get("dificultad", c.nivel))
                self.terreno.usar_alturas(opciones["alturas"], opciones.get("tamano_alturas"))
            else:
                self.tipo_terreno = opciones.get("tipo_terreno") or str(rng.choice(c.tipos_terreno))
                self.dificultad = float(opciones.get("dificultad", c.nivel * np.sqrt(rng.uniform())))
                self.terreno.generar(self.tipo_terreno, self.dificultad, rng)
            n = self.terreno.n
            self.model.hfield_data[self.adr_hfield:self.adr_hfield + n * n] = self.terreno.datos_hfield()
            if self.paso_visual:
                k = self.paso_visual
                vis = (self.terreno.alturas[::k, ::k] / c.altura_max_terreno).astype(np.float32).ravel()
                adr = self.model.hfield_adr[self.id_hfield_dibujo]
                self.model.hfield_data[adr:adr + vis.size] = vis
            self._hfield_pendiente_subir = True
            self._episodios_en_terreno = 0
        self._episodios_en_terreno += 1

        mujoco.mj_resetData(self.model, self.data)
        self._aleatorizar_dinamica()
        self._colocar_robot(opciones.get("posicion"), opciones.get("rumbo"))
        mujoco.mj_forward(self.model, self.data)
        self.pasos = 0

        # --- Órdenes: aleatorias (y cambiantes) salvo que se fije alguna
        self._comandos_fijos = any(k in opciones for k in ("vel_cmd", "giro_cmd", "rumbo_cmd", "mantener_rumbo"))
        self._muestrear_comandos()
        if "vel_cmd" in opciones:
            self.fijar_vel_cmd(opciones["vel_cmd"])
        if c.comando_giro:
            if "giro_cmd" in opciones:
                self.rumbo_cmd = None
                self.giro_cmd = float(np.clip(opciones["giro_cmd"], -c.giro_cmd_max, c.giro_cmd_max))
            elif "rumbo_cmd" in opciones:
                self.rumbo_cmd = _envolver(float(opciones["rumbo_cmd"]))
            elif opciones.get("mantener_rumbo"):
                self.rumbo_cmd = self._rumbo_actual()

        self._accion_previa = np.zeros(self.n_acciones)
        self._empujon_restante = 0
        self._proximo_empujon = int(rng.integers(75, 200))
        self._suma_err_vel, self._n_err_vel = 0.0, 0

        obs, estado = self._observar()
        if self.render_mode == "human":
            self.render()
        return obs, self._info(estado)

    def step(self, accion):
        c, w = self.cfg, self.cfg.pesos
        a = np.clip(np.asarray(accion, dtype=np.float64).reshape(-1), -1.0, 1.0)
        v_ref, w_ref = self.vel_cmd, self.giro_cmd   # órdenes que veía la política al actuar
        self.data.ctrl[:] = a[self._mapa_accion]
        self._gestionar_empujones()
        mujoco.mj_step(self.model, self.data, nstep=c.pasos_por_accion)
        self.pasos += 1
        if not self._comandos_fijos and self.pasos >= self._proximo_cmd:
            self._muestrear_comandos()

        obs, e = self._observar()

        # --- Recompensa
        r_vertical = math.exp(-(e["inclinacion"] / c.tolerancia_vertical) ** 2)
        r_vel = math.exp(-((e["vel_adelante"] - v_ref) ** 2) / c.tolerancia_vel ** 2)
        r_giro = math.exp(-((e["vel_giro"] - w_ref) ** 2) / c.tolerancia_giro ** 2)
        p_vel_pend = e["dincl_cabeceo"] ** 2 + e["dincl_lateral"] ** 2
        p_accion = float(np.mean(a ** 2))
        p_suav = float(np.mean((a - self._accion_previa) ** 2))
        recompensa = (w["vivo"] + w["vertical"] * r_vertical + w["velocidad"] * r_vel
                      + w["giro"] * r_giro - w["vel_pendulo"] * p_vel_pend
                      - w["accion"] * p_accion - w["suavidad"] * p_suav)

        # --- Fin de episodio
        estable = bool(np.all(np.isfinite(obs)))
        cayo = (not estable or e["inclinacion"] > c.inclinacion_max or e["vuelco"] > 1.0)
        fuera = max(abs(e["x"]), abs(e["y"])) > c.semitamano_terreno - 1.5
        terminated = bool(cayo)
        truncated = bool(not cayo and (self.pasos >= self.max_pasos or fuera))
        if cayo:
            recompensa -= w["caida"]
        if not estable:  # simulación inestable (NaN): no contaminar el aprendizaje
            obs = np.zeros_like(obs)
            recompensa = -w["caida"]

        # Error de velocidad en régimen permanente (2 s después de cada cambio de orden)
        if self.pasos - self._ultimo_cambio_cmd >= int(2.0 / self.dt):
            self._suma_err_vel += abs(e["vel_adelante"] - v_ref)
            self._n_err_vel += 1

        self._accion_previa = a
        info = self._info(e)
        info.update(r_vertical=r_vertical, r_vel=r_vel, r_giro=r_giro, cayo=cayo)
        if terminated or truncated:
            info["error_vel_episodio"] = (self._suma_err_vel / self._n_err_vel
                                          if self._n_err_vel else float("nan"))
        if self.render_mode == "human":
            self.render()
        return obs, float(recompensa), terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        if self.render_mode == "human":
            self._render_visor()
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._visor is not None:
            self._visor.close()
            self._visor = None

    def visor_abierto(self) -> bool:
        return self._visor is not None and self._visor.is_running()

    # ================================================================== internos
    def _rumbo_actual(self) -> float:
        R = self.data.xmat[self.id_chasis]           # matriz 3x3 aplanada por filas
        return math.atan2(R[3], R[0])

    def _muestrear_comandos(self) -> None:
        c, rng = self.cfg, self.np_random
        self.vel_cmd = 0.0 if rng.uniform() < c.prob_cmd_cero else float(rng.uniform(-c.vel_cmd_max, c.vel_cmd_max))
        self.giro_cmd, self.rumbo_cmd = 0.0, None
        if c.comando_giro:
            u = rng.uniform()
            if u < 0.4:      # seguir recto: mantener el rumbo actual
                self.rumbo_cmd = self._rumbo_actual()
            elif u < 0.7:    # girar hasta otro rumbo y seguir recto
                self.rumbo_cmd = _envolver(self._rumbo_actual() + rng.uniform(-math.pi, math.pi))
            else:            # girar a velocidad constante
                self.giro_cmd = float(rng.uniform(-c.giro_cmd_max, c.giro_cmd_max))
        self._proximo_cmd = self.pasos + int(rng.uniform(*c.remuestrear_cmd) / self.dt)
        self._ultimo_cambio_cmd = self.pasos

    def _aleatorizar_dinamica(self) -> None:
        m, nom, rng = self.model, self._nominal, self.np_random
        m.body_mass[:] = nom["body_mass"]
        m.body_inertia[:] = nom["body_inertia"]
        m.geom_friction[:] = nom["geom_friction"]
        m.dof_damping[:] = nom["dof_damping"]
        m.actuator_gainprm[:] = nom["gainprm"]
        m.actuator_biasprm[:] = nom["biasprm"]
        m.actuator_forcerange[:] = nom["forcerange"]
        if not self.cfg.aleatorizar_dinamica:
            return
        for cuerpo, (lo, hi) in ((self.id_chasis, (0.8, 1.25)), (self.id_pendulo, (0.75, 1.3))):
            f = rng.uniform(lo, hi)
            m.body_mass[cuerpo] *= f
            m.body_inertia[cuerpo] *= f
        mu = rng.uniform(0.5, 1.2)                      # hierba mojada ... asfalto
        m.geom_friction[self.id_geom_terreno, 0] = mu
        m.geom_friction[self.ids_geom_ruedas, 0] = mu
        f_par = rng.uniform(0.8, 1.2, size=4)           # motores no idénticos
        f_vel = rng.uniform(0.85, 1.15, size=4)
        par = nom["gainprm"][:, 0] * f_par
        m.actuator_gainprm[:, 0] = par
        m.actuator_biasprm[:, 2] = nom["biasprm"][:, 2] * f_par / f_vel
        m.actuator_forcerange[:, 0] = -par
        m.actuator_forcerange[:, 1] = par
        m.dof_damping[self.qvel_piv] *= rng.uniform(0.5, 2.0)

    def _colocar_robot(self, posicion=None, rumbo=None) -> None:
        """Coloca el robot apoyado sobre el terreno, alineado con su pendiente,
        y con el péndulo casi vertical respecto al mundo."""
        c, rng, ter = self.cfg, self.np_random, self.terreno
        if posicion is None:
            r = c.radio_aparicion * np.sqrt(rng.uniform())
            ang = rng.uniform(-np.pi, np.pi)
            x0, y0 = r * np.cos(ang), r * np.sin(ang)
        else:
            x0, y0 = posicion
        psi = rng.uniform(-np.pi, np.pi) if rumbo is None else float(rumbo)

        gx, gy = ter.gradiente_en(x0, y0)
        cps, sps = np.cos(psi), np.sin(psi)
        pend_adelante = gx * cps + gy * sps       # sube hacia delante
        pend_lateral = -gx * sps + gy * cps       # sube hacia la izquierda
        theta = -np.arctan(pend_adelante)         # cabeceo (rotación en y)
        phi = np.arctan(pend_lateral)             # alabeo (rotación en x)

        q = np.zeros(4)
        tmp = np.zeros(4)
        mujoco.mju_mulQuat(tmp, [np.cos(psi / 2), 0, 0, np.sin(psi / 2)],
                           [np.cos(theta / 2), 0, np.sin(theta / 2), 0])
        mujoco.mju_mulQuat(q, tmp, [np.cos(phi / 2), np.sin(phi / 2), 0, 0])
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        R = R.reshape(3, 3)

        radio = c.robot.radio_rueda
        z0 = 0.0
        for p_local in self.ruedas_local:
            p = R @ p_local
            h = ter.altura_max_en(x0 + p[0], y0 + p[1], radio * 0.8)
            z0 = max(z0, h + radio + 0.005 - p[2])

        d = self.data
        d.qpos[0:3] = (x0, y0, z0)
        d.qpos[3:7] = q
        d.qpos[self.qpos_piv] = -theta + rng.uniform(-0.05, 0.05)
        d.qvel[:] = 0.0
        d.qvel[self.qvel_piv] = rng.uniform(-0.1, 0.1)

    def _gestionar_empujones(self) -> None:
        c, rng, d = self.cfg, self.np_random, self.data
        if self._empujon_restante > 0:
            self._empujon_restante -= 1
            if self._empujon_restante == 0:
                d.xfrc_applied[self._cuerpo_empujado, :] = 0.0
            return
        if not c.empujones or self.dificultad <= 0 or self.pasos < self._proximo_empujon:
            return
        if rng.uniform() < 0.5:
            self._cuerpo_empujado, fmax = self.id_chasis, c.empujon_chasis_max
        else:
            self._cuerpo_empujado, fmax = self.id_pendulo, c.empujon_pendulo_max
        f = fmax * self.dificultad * rng.uniform(0.3, 1.0)
        ang = rng.uniform(-np.pi, np.pi)
        d.xfrc_applied[self._cuerpo_empujado, :3] = (f * np.cos(ang), f * np.sin(ang), 0.0)
        self._empujon_restante = int(rng.integers(5, 11))  # 0.1–0.2 s
        self._proximo_empujon = self.pasos + int(rng.integers(100, 250))

    def _observar(self):
        m, d, o = self.model, self.data, self._obs
        R = d.xmat[self.id_chasis].reshape(3, 3)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.id_chasis, self._vel_c, 1)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.id_pendulo, self._vel_p, 0)
        w_cuerpo, v_cuerpo = self._vel_c[:3], self._vel_c[3:]

        # Marco "rumbo": como el mundo pero girado con la orientación horizontal del robot.
        psi = math.atan2(R[1, 0], R[0, 0])
        cps, sps = math.cos(psi), math.sin(psi)
        rp = d.xmat[self.id_pendulo]                     # eje z del péndulo = columna 2
        ex, ey, ez = rp[2], rp[5], rp[8]
        incl_cab = math.atan2(cps * ex + sps * ey, ez)
        incl_lat = math.atan2(-sps * ex + cps * ey, ez)
        wx, wy = self._vel_p[0], self._vel_p[1]
        dincl_cab = -sps * wx + cps * wy                 # derivada de incl_cab (giro en y del rumbo)
        dincl_lat = -(cps * wx + sps * wy)               # derivada de incl_lat (giro en -x del rumbo)
        v_mundo = R @ v_cuerpo
        if self.rumbo_cmd is not None:                   # mantener rumbo -> velocidad de giro ordenada
            gmax = self.cfg.giro_cmd_max
            self.giro_cmd = max(-gmax, min(gmax, self.cfg.ganancia_rumbo * _envolver(self.rumbo_cmd - psi)))

        o[0:3] = -R[2, :]                                # gravedad en el marco del chasis
        o[3:6] = w_cuerpo
        o[6:9] = v_cuerpo
        o[9] = d.qpos[self.qpos_piv]
        o[10] = d.qvel[self.qvel_piv]
        o[11], o[12], o[13], o[14] = incl_cab, incl_lat, dincl_cab, dincl_lat
        o[15:19] = d.qvel[self.qvel_ruedas]
        n = self.n_acciones
        o[19:19 + n] = self._accion_previa
        o[19 + n] = self.vel_cmd
        if self.cfg.comando_giro:
            o[20 + n] = self.giro_cmd
        obs = o.astype(np.float32)
        if self.cfg.ruido_obs > 0:
            obs += self.np_random.standard_normal(obs.shape[0]).astype(np.float32) * self._escala_ruido

        estado = dict(
            inclinacion=math.acos(min(1.0, max(-1.0, ez))),
            incl_cabeceo=incl_cab, incl_lateral=incl_lat,
            dincl_cabeceo=dincl_cab, dincl_lateral=dincl_lat,
            vel_adelante=cps * v_mundo[0] + sps * v_mundo[1],
            vel_giro=float(R[2, :] @ w_cuerpo),
            vuelco=math.acos(min(1.0, max(-1.0, R[2, 2]))),
            x=float(d.qpos[0]), y=float(d.qpos[1]), rumbo=psi,
        )
        self._ultimo_estado = estado
        return obs, estado

    def _info(self, e: dict) -> dict:
        return dict(e, tipo_terreno=self.tipo_terreno, dificultad=self.dificultad,
                    vel_cmd=self.vel_cmd, giro_cmd=self.giro_cmd, rumbo_cmd=self.rumbo_cmd,
                    nivel=self.cfg.nivel)

    # ------------------------------------------------------------------ render
    def _render_rgb(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.cfg.alto_video, self.cfg.ancho_video)
            self._hfield_pendiente_subir = False  # el contexto nuevo ya sube el terreno actual
        if self._hfield_pendiente_subir:
            self._renderer._gl_context.make_current()
            mujoco.mjr_uploadHField(self.model, self._renderer._mjr_context, self.id_hfield_dibujo)
            self._hfield_pendiente_subir = False
        self._renderer.update_scene(self.data, camera=self.cfg.camara)
        return self._renderer.render()

    def _render_visor(self):
        import mujoco.viewer

        if self._visor is None:
            self._visor = mujoco.viewer.launch_passive(self.model, self.data, key_callback=self._tecla)
            with self._visor.lock():
                self._visor.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self._visor.cam.trackbodyid = self.id_chasis
                self._visor.cam.distance = 2.8
                self._visor.cam.elevation = -18
                self._visor.cam.azimuth = 110
            self._hfield_pendiente_subir = False
        if not self._visor.is_running():
            return
        if self._hfield_pendiente_subir:
            self._visor.update_hfield(self.id_hfield_dibujo)
            self._hfield_pendiente_subir = False
        try:
            self._visor.set_texts((
                mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                # La fuente de MuJoCo solo tiene ASCII
                "Terreno\nDificultad\nVel. ordenada\nGiro ordenado\nInclinacion"
                + ("\nModelo" if self.texto_extra else ""),
                f"{self.tipo_terreno}\n{self.dificultad:.2f}\n{self.vel_cmd:+.2f} m/s  (flechas arriba/abajo)\n"
                + (f"rumbo {np.degrees(self.rumbo_cmd):+.0f} grados  (flechas izq/der)" if self.rumbo_cmd is not None
                   else f"{self.giro_cmd:+.2f} rad/s") + "\n"
                f"{np.degrees(self._ultimo_estado.get('inclinacion', 0.0)):.1f} grados"
                + (f"\n{self.texto_extra}" if self.texto_extra else ""),
            ))
        except Exception:  # noqa: BLE001  (versiones antiguas de mujoco)
            pass
        self._visor.sync()
        # Mantener tiempo real
        ahora = time.perf_counter()
        if self._t_ultimo_frame is not None:
            espera = self.dt - (ahora - self._t_ultimo_frame)
            if espera > 0:
                time.sleep(espera)
        self._t_ultimo_frame = time.perf_counter()

    def _tecla(self, codigo: int) -> None:
        # Códigos GLFW: 265 arriba, 264 abajo, 263 izquierda, 262 derecha.
        # Al usar el teclado, las órdenes dejan de cambiar solas.
        if codigo in (265, 264):
            self._comandos_fijos = True
            self.fijar_vel_cmd(self.vel_cmd + (0.1 if codigo == 265 else -0.1))
        elif codigo in (263, 262):
            self._comandos_fijos = True
            self.girar_rumbo(math.radians(15 if codigo == 263 else -15))


def _envolver(angulo: float) -> float:
    """Lleva un ángulo a [-pi, pi)."""
    return (angulo + math.pi) % (2 * math.pi) - math.pi
