# Robot de 4 ruedas con péndulo invertido — simulador para aprendizaje por refuerzo

Entorno de simulación (MuJoCo + Gymnasium) para entrenar con RL un robot de 4 ruedas
que debe mantener un péndulo invertido en equilibrio sobre terrenos variados
(plano, rampas, colinas, rugoso, escalones y mixto), con empujones, ruido de sensores
y variaciones físicas para que la política sea robusta.

```
robot_pendulo/
  modelo.py       # robot en MJCF: chasis, 4 ruedas motorizadas, péndulo (parámetros en ParamsRobot)
  terreno.py      # generador procedural de terrenos (heightfield)
  entorno.py      # entorno Gymnasium: observación, acción, recompensa, aleatorización
  utilidades.py   # controlador PD de referencia y carga de modelos
  vec_lotes.py    # simulación en paralelo con varios robots por proceso
entrenar.py       # entrenamiento PPO/SAC con currículo de terreno (CPU o GPU)
evaluar.py        # ventana 3D, vídeo y estadísticas por terreno
ver_entrenamiento.py # ventana 3D que sigue un entrenamiento en curso (último checkpoint)
probar_entorno.py # comprobaciones rápidas + galería de terrenos
crear_terreno_ejemplo.py # genera terrenos/pista_ejemplo.png (plantilla de terreno propio)
diagnostico.py    # ¿está listo este ordenador para la demo? gráfica, modelos, velocidad, calidad
instalar.bat      # crea/repara .venv, instala las librerías y lanza el diagnóstico
```

## Instalación (Windows, Python 3.10 – 3.14)

**Lo más fácil (p. ej. en otro portátil):** doble clic en `instalar.bat`. Busca un Python
compatible, crea el entorno `.venv` (o lo rehace si se copió de otro ordenador: un `.venv`
copiado no funciona, apunta al Python del PC original), instala las librerías y ejecuta
`diagnostico.py`, que dice qué calidad gráfica usar. A mano:

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

Con GPU NVIDIA (p. ej. RTX 2080), instala primero PyTorch con CUDA; si no, sáltate este paso
y se instalará la versión solo-CPU:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

```bash
pip install -r requirements.txt
```

Comprueba que todo funciona (API, velocidad de simulación y `terrenos.png`):

```bash
python probar_entorno.py
```

## Uso

**1. Ver el simulador con un controlador clásico (sin RL)** — útil para entender el problema:

```bash
python evaluar.py --pd --terreno colinas
```

En la ventana: flechas ↑/↓ cambian la velocidad ordenada y ←/→ giran el rumbo ordenado
15° (en cuanto tocas una flecha, las órdenes dejan de cambiar solas); doble clic sobre una
pieza y Ctrl + arrastrar con el botón derecho para empujarla.

**Calidad del dibujo** (`--calidad alta|media|baja`, solo cambia lo que se ve; la física y
los resultados son idénticos). Por defecto la ventana usa `media` (terreno dibujado a 10 cm)
y los vídeos `alta` (5 cm). La `alta` tarda ~35 ms por imagen incluso en una RTX 2080, más
que los 20 ms de cada paso de control, y la `media` ~3 ms. En un portátil sin gráfica
dedicada usa `--calidad baja` (20 cm y sin sombras).

### En un portátil (p. ej. para una presentación)
1. `instalar.bat`, o `python diagnostico.py` si ya está instalado. Al final recomienda la
   calidad y los comandos para la demo.
2. **Cargador conectado**: con batería Windows baja mucho la velocidad de CPU y gráfica.
3. Si el diagnóstico dice que Python usa la gráfica integrada teniendo una NVIDIA/AMD
   dedicada: `python diagnostico.py --gpu-dedicada` (equivale a Configuración > Pantalla >
   Gráficos > python.exe > Alto rendimiento), reabre la terminal y vuelve a comprobar.
4. Si el proyecto está en OneDrive, marca la carpeta como "Mantener siempre en este
   dispositivo" para que los modelos no dependan de internet.
5. La ruta de la carpeta debe ser corta (≤ 100 caracteres, p. ej. `C:\robot-pendulo-rl`):
   PyTorch instala archivos con rutas de ~150 caracteres y Windows corta en 260, así que
   en una carpeta muy anidada `pip` falla a mitad. `instalar.bat` lo comprueba antes.
6. Lleva de respaldo vídeos grabados con `--video` en un PC con buena gráfica.

**2. Entrenar** (por defecto PPO, 30 M pasos, 48 robots repartidos en un proceso por
núcleo físico; la red se entrena en la GPU si PyTorch la detecta):

```bash
python entrenar.py
```

Sigue el progreso en otra terminal (curvas de recompensa, nivel del currículo y
tasa de éxito por terreno en `eval/exito_*`):

```bash
tensorboard --logdir runs
```

Y para **ver el entrenamiento en 3D mientras entrena** (carga el último checkpoint, que se
guarda cada 1 M pasos, y lo va actualizando; cerrar la ventana no detiene el entrenamiento).
Dibuja el terreno a 10 cm y sin sombras: el de 5 cm satura la GPU (100 %) y frenaba el
entrenamiento de ~3.900 a ~250 pasos/s; así usa ~20 % de GPU y no lo frena. La física es la misma:

```bash
python ver_entrenamiento.py
```

Puedes parar con Ctrl+C: el modelo se guarda igualmente. Para continuar:

```bash
python entrenar.py --continuar runs/ppo_XXXX/modelo_final.zip --pasos 10e6
```

**3. Evaluar la política entrenada:**

```bash
python evaluar.py runs/ppo_XXXX/modelo_final.zip
```

```bash
python evaluar.py runs/ppo_XXXX/modelo_final.zip --sin-ventana --episodios 60
```

```bash
python evaluar.py runs/ppo_XXXX/modelo_final.zip --terreno rugoso --video rugoso.mp4 --episodios 2
```

Compara con el PD clásico usando las mismas semillas:

```bash
python evaluar.py --pd --sin-ventana --episodios 60
```

## Cómo está planteado el problema

### Robot (`ParamsRobot` en `modelo.py`)
- Chasis de 40 × 28 × 8 cm y 3 kg, ruedas de 10 cm de radio.
- 4 motores DC independientes con curva par-velocidad real: `par = par_max·(u − ω/ω_vacío)`
  (1,5 N·m de arranque, 30 rad/s en vacío).
- Péndulo de 60 cm con 0,3 kg en la punta, articulado en el eje lateral del chasis:
  cae hacia delante o hacia atrás, y el robot lo equilibra acelerando hacia ese lado.
  **Cambia estos valores por los de tu robot real** antes de entrenar.

### Acción
Par normalizado `[-1, 1]` por rueda (4 valores). Con `--modo-accion diferencial` son
2 valores (lado izquierdo / derecho). Control a 50 Hz, física a 200 Hz.

### Observación (25 valores, lo que mediría un robot real con IMU + encoders)
gravedad en el marco del chasis (3), giroscopio (3), velocidad del chasis (3),
ángulo y velocidad del encoder del pivote (2), inclinación del péndulo respecto a la
vertical del mundo (2), su velocidad angular (2), velocidad de las 4 ruedas (4),
acción anterior (4), velocidad de avance ordenada (1) y velocidad de giro ordenada (1).
Todo con ruido gaussiano.

### Órdenes
Velocidad de avance (hasta ±0,5 m/s) y giro. El giro se ordena como velocidad de giro
(hasta ±0,6 rad/s) o como un **rumbo a mantener**: entonces el entorno calcula
`giro = clip(1,0·(rumbo_ordenado − rumbo))`, igual que haría el robot real con su IMU.
Así va recto aunque los motores no sean idénticos o lo empujen. Durante el entrenamiento
las órdenes cambian cada 5–10 s (40 % seguir recto, 30 % girar a otro rumbo, 30 % girar a
velocidad constante; 20 % de las veces, velocidad 0).

### Recompensa (pesos en `ConfigEntorno.pesos`)
`+0,5` por paso vivo · `+1,0·exp(−(inclinación/0,2)²)` péndulo vertical ·
`+1,5·exp(−((v − v_ordenada)/0,32)²)` seguir la velocidad · `+0,5·exp(−((ω − ω_ordenada)/0,5)²)`
seguir el giro · penalizaciones por oscilar, par excesivo y cambios bruscos · `−10` al caer.
El episodio termina si el péndulo pasa de 40° o el chasis vuelca; dura 20 s como máximo.

La inclinación se mide **respecto a la vertical del mundo**, no del chasis: en una
rampa el chasis se inclina pero el péndulo debe seguir vertical.

### Terreno y currículo
- 6 tipos de terreno de 30 × 30 m (resolución 5 cm). Con dificultad 1: pendientes de
  hasta 15°, baches de ±3 cm, escalones de hasta 5 cm (configurable en `ConfigEntorno`).
- `entrenar.py` empieza con nivel 0 (plano) y sube el nivel 0,05 cada vez que el 80 % de
  los episodios son un éxito (lo baja si baja del 40 %). Cada terreno se genera con
  dificultad `nivel·√U(0,1)`, así se siguen practicando los fáciles.
- **Éxito = no caer _y_ obedecer**: el error medio de velocidad del episodio (medido 2 s
  después de cada cambio de orden) debe ser menor de 0,2 m/s (`--umbral-error-vel`).
  Con solo "no caer", la política llegaba a la dificultad máxima en 4,5 M pasos y a partir
  de ahí aprendía a quedarse casi quieta (lo más seguro): a los 2 M pasos seguía bien la
  velocidad y a los 30 M ya no (`runs/ppo_giro_30M_v1`).

### Robustez (sim-to-real)
En cada episodio cambian: masa del chasis (±20 %) y del péndulo (±25 %), rozamiento
del suelo (0,5–1,2), par y velocidad de cada motor (±15–20 %) y amortiguamiento del
pivote. Además hay empujones aleatorios al chasis (hasta 25 N) y al péndulo (hasta 3 N).
Todo se puede desactivar: `RobotPenduloEnv(aleatorizar_dinamica=False, empujones=False, ruido_obs=0)`.

## Probar en tu propio terreno

Dibuja un **mapa de alturas** en cualquier editor de imágenes (GIMP, Krita, Photoshop...):
escala de grises, negro = suelo más bajo, blanco = `--altura-blanco` metros. Se interpreta
como un mapa visto desde arriba: **arriba = norte (+y), derecha = este (+x)**. Usa PNG de
16 bits si puedes (8 bits solo da 256 niveles de altura). También valen matrices con las
alturas en metros en `.npy` o `.csv` (primera fila = norte).

```bash
python evaluar.py runs/ppo_10M_verificacion/modelo_final.zip --mapa mi_terreno.png --tamano 20 --altura-blanco 0.3 --inicio -8 0 --rumbo 0 --vel-cmd 0.4 --duracion 40
```

- `--tamano`: metros que mide el lado mayor del mapa (máx. 30). Se reescala a la malla de
  5 cm de la simulación; fuera del mapa se prolonga la altura del borde.
- `--inicio X Y` y `--rumbo` (grados, 0 = mirando a la derecha del mapa): dónde sale el robot.
- `--dificultad 0` o `--sin-perturbaciones` quitan los empujones.

Hay una pista de ejemplo con rampas de 8°, meseta con baches, grava, escalones de 3 cm,
ondulaciones y pendiente lateral (`crear_terreno_ejemplo.py`, úsalo como plantilla):

```bash
python crear_terreno_ejemplo.py
```

```bash
python evaluar.py runs/ppo_10M_verificacion/modelo_final.zip --mapa terrenos/pista_ejemplo.png --tamano 24 --altura-blanco 0.6 --inicio -10 0 --rumbo 0 --vel-cmd 0.5 --duracion 50
```

Desde Python: `env.reset(options={"alturas": cargar_mapa_alturas("mi_terreno.png", 0.3), "tamano_alturas": 20})`
(con `from robot_pendulo.terreno import cargar_mapa_alturas`).

Con `--mapa` o `--vel-cmd`, el robot mantiene el rumbo inicial (va recto); `evaluar.py`
muestra cuánto **avanza** en esa dirección y cuánto se **desvía** a los lados.

### Estado de los modelos entrenados (13/11/2026)

| Modelo | Éxito en terreno aleatorio (60 ep.) | Sigue la velocidad | Pista de ejemplo |
|---|---|---|---|
| `runs/ppo_10M_verificacion` (sin orden de giro) | **92 %** | no | se para al pie de la rampa |
| `runs/ppo_giro_30M_v1` | 77 % | no (deriva hacia delante) | se para al pie de la rampa |
| `runs/ppo_giro_v2_30M/checkpoints/modelo_7999872_steps.zip` | 80 % | **sí, en llano** | se para al pie de la rampa |
| `runs/ppo_giro_v2_30M` (final) | 43 % (empeoró al final) | parcial | se para al pie de la rampa |

**Limitación conocida: ningún modelo sube pendientes sostenidas**, ni siquiera de 4°, y en
una cuesta constante bajan aunque se les pida subir; el PD clásico sí las sube. Los
terrenos de entrenamiento tienen pendientes que cambian de dirección cada 2–4 m, así que
la política nunca aprendió a aplicar el par extra que hace falta para mantenerse en una
cuesta. La solución es añadir cuestas largas al generador de terreno y reentrenar.

## Usar el entorno desde tu propio código

```python
import gymnasium as gym
import robot_pendulo  # registra "RobotPendulo-v0"

env = gym.make("RobotPendulo-v0", render_mode="human", nivel=0.5, modo_accion="diferencial")
obs, info = env.reset(seed=0, options={"tipo_terreno": "rampas", "vel_cmd": 0.3})
```

Opciones de `reset`: `tipo_terreno`, `dificultad`, `vel_cmd`, `posicion=(x, y)`, `rumbo` (radianes),
`alturas` (matriz en metros) y `tamano_alturas` (m).
Cualquier campo de `ConfigEntorno` se puede pasar como argumento al crear el entorno.

## Rendimiento y GPU
La física (MuJoCo) siempre corre en la CPU; la GPU solo entrena la red neuronal.
Medido en un Ryzen 5 2600 + RTX 2080:

| | pasos/s | 30 M pasos |
|---|---|---|
| CPU | ~3.500 | ~2 h 25 min |
| GPU (RTX 2080) | ~4.100 | ~2 h |

Con la GPU la actualización de la red baja de 0,78 s a 0,24 s por iteración, pero
simular los robots sigue siendo lo más lento, así que lo que más acelera es tener más
núcleos de CPU. `--dispositivo cpu` fuerza la CPU.

`entrenar.py` agrupa varios robots en cada proceso (`robot_pendulo/vec_lotes.py`): en
Windows cada mensaje entre procesos cuesta más que simular un paso, y con un robot por
proceso (`SubprocVecEnv` normal) el entrenamiento iba a la mitad de velocidad.

## Consejos
- Si en TensorBoard `curriculo/nivel` se estanca, entrena más pasos o baja la
  dificultad máxima (`pendiente_max_grados`, `rugosidad_max`, `escalon_max`).
- Para llevarlo a un robot real, exporta solo la red de la política (`modelo.policy`)
  junto con la media/varianza de `vecnormalize.pkl`, y replica exactamente el orden y
  las unidades de la observación.
