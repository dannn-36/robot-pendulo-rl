"""Modelo físico (MJCF de MuJoCo) del robot de 4 ruedas con péndulo invertido.

Convención de ejes del robot:
    +x  hacia delante
    +y  hacia la izquierda
    +z  hacia arriba

El péndulo gira sobre un pivote en el eje y del chasis (se inclina hacia
delante / atrás), igual que un carro-péndulo clásico. El robot lo equilibra
acelerando las ruedas hacia el lado al que cae el péndulo.
"""
from __future__ import annotations

from dataclasses import dataclass

# Orden de las ruedas (y de los motores / acciones en modo "independiente").
#   di = delantera izquierda, dd = delantera derecha,
#   ti = trasera izquierda,   td = trasera derecha
NOMBRES_RUEDAS = ("di", "dd", "ti", "td")


@dataclass
class ParamsRobot:
    """Parámetros físicos nominales. Ajústalos a tu robot real."""

    # Chasis (caja)
    largo_chasis: float = 0.40       # m (eje x)
    ancho_chasis: float = 0.28       # m (eje y)
    alto_chasis: float = 0.08        # m (eje z)
    masa_chasis: float = 3.0         # kg (incluye batería, electrónica...)

    # Ruedas
    radio_rueda: float = 0.10        # m
    ancho_rueda: float = 0.04        # m
    masa_rueda: float = 0.30         # kg
    distancia_ejes: float = 0.30     # m entre eje delantero y trasero
    holgura_rueda: float = 0.005     # m entre chasis y rueda

    # Péndulo (barra + masa en la punta)
    largo_pendulo: float = 0.60      # m
    radio_barra: float = 0.012       # m
    masa_barra: float = 0.10         # kg
    masa_punta: float = 0.30         # kg
    radio_punta: float = 0.035       # m
    amortiguamiento_pivote: float = 0.002  # N·m·s/rad
    limite_pivote: float = 1.45      # rad (tope mecánico)

    # Motores de corriente continua (modelo par-velocidad lineal):
    #   par = par_max * (u - w / vel_vacio),  limitado a [-par_max, par_max]
    par_max: float = 1.5             # N·m por rueda (par de arranque)
    vel_vacio: float = 30.0          # rad/s sin carga (≈ 3 m/s con r=0.1)
    friccion_rueda: float = 1.0      # coeficiente de rozamiento nominal


def construir_mjcf(p: ParamsRobot, n_terreno: int, semitamano: float,
                   altura_max: float, dt: float, n_visual: int | None = None) -> str:
    """Devuelve el XML MJCF con el robot y un campo de alturas vacío.

    Con `n_visual`, el terreno de la física queda oculto (grupo 3) y se dibuja en su
    lugar una copia más gruesa de n_visual x n_visual puntos, mucho más barata de renderizar.
    """
    if n_visual:
        hfield_visual = (f'<hfield name="terreno_visual" nrow="{n_visual}" ncol="{n_visual}" '
                         f'size="{semitamano} {semitamano} {altura_max} 0.5"/>')
        geom_visual = ('<geom name="terreno_visual" type="hfield" hfield="terreno_visual" '
                       'material="terreno" contype="0" conaffinity="0" group="0"/>')
        grupo_fisico = ' group="3"'
    else:
        hfield_visual = geom_visual = grupo_fisico = ""
    lx, ly, lz = p.largo_chasis / 2, p.ancho_chasis / 2, p.alto_chasis / 2
    R, w = p.radio_rueda, p.ancho_rueda
    dx = p.distancia_ejes / 2
    dy = ly + p.holgura_rueda + w / 2
    L = p.largo_pendulo

    # Inercia de un disco macizo girando sobre su eje (y) y sobre un diámetro.
    i_eje = 0.5 * p.masa_rueda * R ** 2
    i_diam = p.masa_rueda * (3 * R ** 2 + w ** 2) / 12

    posiciones = {"di": (+dx, +dy), "dd": (+dx, -dy), "ti": (-dx, +dy), "td": (-dx, -dy)}
    ruedas_xml = []
    for nombre in NOMBRES_RUEDAS:
        x, y = posiciones[nombre]
        ruedas_xml.append(f"""
      <body name="rueda_{nombre}" pos="{x} {y} 0">
        <joint name="rueda_{nombre}" type="hinge" axis="0 1 0" damping="0.002" armature="0.0005"/>
        <inertial pos="0 0 0" mass="{p.masa_rueda}" diaginertia="{i_diam} {i_eje} {i_diam}"/>
        <!-- colisión: esfera (contacto estable con el campo de alturas) -->
        <geom name="rueda_{nombre}" class="contacto_rueda" type="sphere" size="{R}"/>
        <!-- visual: cilindro con radios -->
        <geom class="visual" type="cylinder" size="{R} {w / 2}" euler="1.5708 0 0" rgba=".12 .12 .12 1"/>
        <geom class="visual" type="box" size="{R * 0.8} {w / 2 + 0.001} {R * 0.12}" rgba=".85 .65 .1 1"/>
      </body>""")

    motores_xml = []
    for nombre in NOMBRES_RUEDAS:
        motores_xml.append(
            f'<general name="motor_{nombre}" joint="rueda_{nombre}" ctrlrange="-1 1" '
            f'gaintype="fixed" gainprm="{p.par_max}" biastype="affine" '
            f'biasprm="0 0 {-p.par_max / p.vel_vacio}" forcerange="{-p.par_max} {p.par_max}"/>'
        )

    return f"""
<mujoco model="robot_pendulo">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{dt}" integrator="implicitfast"/>
  <statistic center="0 0 0.4" extent="2.5"/>

  <visual>
    <global offwidth="1280" offheight="720" azimuth="120" elevation="-20"/>
    <quality shadowsize="4096"/>
    <map znear="0.01" zfar="100"/>
    <headlight ambient=".35 .35 .35" diffuse=".5 .5 .5"/>
  </visual>

  <asset>
    <hfield name="terreno" nrow="{n_terreno}" ncol="{n_terreno}"
            size="{semitamano} {semitamano} {altura_max} 0.5"/>
    {hfield_visual}
    <texture name="cuadros" type="2d" builtin="checker" width="512" height="512"
             rgb1=".42 .50 .38" rgb2=".36 .44 .33" mark="edge" markrgb=".3 .35 .28"/>
    <material name="terreno" texture="cuadros" texrepeat="1 1" texuniform="true" reflectance="0"/>
    <texture name="cielo" type="skybox" builtin="gradient" rgb1=".55 .7 .9" rgb2=".1 .15 .25"
             width="512" height="512"/>
  </asset>

  <default>
    <!-- contype/conaffinity: el robot choca con el terreno pero no consigo mismo -->
    <default class="robot">
      <geom contype="2" conaffinity="1" condim="3" friction="0.6 0.005 0.0001"/>
    </default>
    <default class="contacto_rueda">
      <geom contype="2" conaffinity="1" condim="3" group="3" rgba="1 0 0 .3"
            friction="{p.friccion_rueda} 0.005 0.0001" mass="0"/>
    </default>
    <default class="visual">
      <geom contype="0" conaffinity="0" group="1" mass="0"/>
    </default>
  </default>

  <worldbody>
    <light name="sol" pos="0 0 20" dir="0.3 0.3 -1" directional="true" castshadow="true"
           diffuse=".7 .7 .7"/>
    <geom name="terreno" type="hfield" hfield="terreno" material="terreno"
          contype="1" conaffinity="1" condim="3" friction="1 0.005 0.0001"{grupo_fisico}/>
    {geom_visual}

    <body name="chasis" pos="0 0 0.3">
      <freejoint name="base"/>
      <camera name="seguimiento" mode="trackcom" pos="0.6 -2.3 1.1" xyaxes="0.97 0.25 0 -0.11 0.42 0.9"/>
      <camera name="lateral" mode="trackcom" pos="0 -2.6 0.6" xyaxes="1 0 0 0 0.15 0.99"/>
      <site name="imu" pos="0 0 0" size="0.01"/>
      <geom name="chasis" class="robot" type="box" size="{lx} {ly} {lz}" mass="{p.masa_chasis}"
            rgba=".15 .3 .65 1"/>
      <geom class="visual" type="box" size="{lx * 0.6} {ly * 0.7} 0.004" pos="0 0 {lz}" rgba=".9 .9 .92 1"/>
{''.join(ruedas_xml)}

      <body name="pendulo" pos="0 0 {lz}">
        <joint name="pivote" type="hinge" axis="0 1 0" range="{-p.limite_pivote} {p.limite_pivote}"
               damping="{p.amortiguamiento_pivote}"/>
        <geom class="visual" type="cylinder" size="0.02 0.015" euler="1.5708 0 0" rgba=".3 .3 .3 1"/>
        <geom name="barra" class="robot" type="capsule" fromto="0 0 0 0 0 {L}" size="{p.radio_barra}"
              mass="{p.masa_barra}" rgba=".8 .8 .82 1"/>
        <geom name="punta" class="robot" type="sphere" pos="0 0 {L}" size="{p.radio_punta}"
              mass="{p.masa_punta}" rgba=".85 .2 .15 1"/>
        <site name="punta" pos="0 0 {L}" size="0.01"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    {chr(10).join('    ' + m for m in motores_xml).strip()}
  </actuator>
</mujoco>
"""
