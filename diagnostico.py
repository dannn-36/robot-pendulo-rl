"""Comprueba si este ordenador está listo para la demo y qué calidad gráfica usar.

    python diagnostico.py                 # informe completo (menos de 1 minuto)
    python diagnostico.py --gpu-dedicada  # pide a Windows que Python use la gráfica dedicada

Revisa Python y librerías, si los modelos cargan y funcionan, qué tarjeta gráfica usa
de verdad Python (en portátiles suele ser la integrada aunque haya una NVIDIA), si el
portátil va con batería, la velocidad de la física y cuánto tarda cada imagen 3D en
cada calidad. Al final recomienda los comandos para la demo.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # consolas Windows sin UTF-8
    sys.stdout.reconfigure(errors="replace")

RAIZ = Path(__file__).resolve().parent
# PyTorch instala archivos con rutas de hasta ~150 caracteres dentro de .venv, y Windows
# (sin las rutas largas activadas) no admite más de 259 en total: si la carpeta del
# proyecto tiene una ruta más larga que esto, pip falla a mitad de instalar torch.
LARGO_MAX_CARPETA = 100
avisos: list[str] = []
fallos: list[str] = []


def titulo(texto: str) -> None:
    print(f"\n=== {texto}")


def ok(texto: str) -> None:
    print(f"  OK     {texto}")


def aviso(texto: str) -> None:
    print(f"  AVISO  {texto}")
    avisos.append(texto)


def fallo(texto: str) -> None:
    print(f"  FALLO  {texto}")
    fallos.append(texto)


def ejecutables_python() -> list[str]:
    """python.exe del .venv y el Python base: en Windows el del .venv lanza el base,
    que es el proceso que realmente dibuja."""
    rutas = [sys.executable, getattr(sys, "_base_executable", sys.executable)]
    return list(dict.fromkeys(str(Path(r).resolve()) for r in rutas))


# ---------------------------------------------------------------------- secciones
def sistema() -> None:
    titulo("Sistema")
    print(f"  Python {platform.python_version()} en {', '.join(ejecutables_python())}")
    print(f"  {platform.platform()} · {os.cpu_count()} hilos de CPU · {platform.processor()}")
    if not (3, 10) <= sys.version_info[:2] <= (3, 14):
        aviso(f"Python {platform.python_version()} no está probado: usa Python 3.10–3.14 (mejor 3.13 o 3.14)")
    if sys.prefix == sys.base_prefix:
        aviso("No estás dentro del entorno .venv: actívalo con .venv\\Scripts\\activate (o ejecuta instalar.bat)")
    if "onedrive" in str(RAIZ).lower():
        print("  La carpeta del proyecto está dentro de OneDrive.")

    if sys.platform == "win32":
        class EstadoEnergia(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                        ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                        ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]

        e = EstadoEnergia()
        if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(e)):
            if e.ACLineStatus == 0:
                aviso("El portátil va con BATERÍA: Windows baja la velocidad de CPU y gráfica. "
                      "Conecta el cargador para la demo")
            elif e.ACLineStatus == 1:
                ok("Conectado al cargador")
            if e.SystemStatusFlag == 1:
                aviso("El ahorro de batería de Windows está activado: desactívalo para la demo")


def librerias() -> bool:
    titulo("Librerías")
    todo_ok = True
    for nombre in ("numpy", "mujoco", "gymnasium", "torch", "stable_baselines3", "imageio"):
        try:
            mod = __import__(nombre)
            print(f"  {nombre:18s} {getattr(mod, '__version__', '?')}")
        except Exception as e:  # noqa: BLE001
            fallo(f"No se puede importar {nombre}: {type(e).__name__}: {e}. "
                  "Ejecuta: pip install -r requirements.txt")
            todo_ok = False
    try:
        import numpy as np

        if int(np.__version__.split(".")[0]) < 2:
            fallo(f"numpy {np.__version__}: los modelos se guardaron con numpy 2. "
                  "Ejecuta: pip install -U \"numpy>=2\"")
            todo_ok = False
    except ImportError:
        pass
    return todo_ok


def archivos() -> list[Path]:
    titulo("Modelos entrenados")
    modelos = sorted((RAIZ / "runs").glob("*/modelo_final.zip"))
    if not modelos:
        fallo("No hay runs/*/modelo_final.zip: copia la carpeta runs/ completa (o haz git pull)")
        return []
    # OneDrive "Archivos a petición": archivos que solo están en la nube y se descargan
    # al abrirlos. Sin internet durante la demo, fallarían.
    solo_nube = 0x400000 | 0x40000 | 0x1000  # RECALL_ON_DATA_ACCESS | RECALL_ON_OPEN | OFFLINE
    en_nube = [f for f in (RAIZ / "runs").rglob("*")
               if f.is_file() and getattr(f.stat(), "st_file_attributes", 0) & solo_nube]
    if en_nube:
        aviso(f"{len(en_nube)} archivos de runs/ están solo en la nube de OneDrive. Clic derecho en la "
              "carpeta del proyecto > 'Mantener siempre en este dispositivo', por si no hay internet")
    for m in modelos:
        print(f"  {m.relative_to(RAIZ)}")
    return modelos


def tarjetas_graficas() -> list[str]:
    if sys.platform != "win32":
        return []
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command",
                            "(Get-CimInstance Win32_VideoController).Name"],
                           capture_output=True, text=True, timeout=20)
        return [linea.strip() for linea in r.stdout.splitlines() if linea.strip()]
    except Exception:  # noqa: BLE001
        return []


def es_dedicada(nombre: str) -> bool:
    n = nombre.lower()
    return "nvidia" in n or "radeon rx" in n or "radeon pro" in n or "arc a" in n


def opengl() -> None:
    titulo("Tarjeta gráfica (OpenGL)")
    tarjetas = tarjetas_graficas()
    for t in tarjetas:
        print(f"  Instalada: {t}{'  (dedicada)' if es_dedicada(t) else ''}")
    try:
        import mujoco
        from OpenGL import GL

        ctx = mujoco.GLContext(64, 64)
        ctx.make_current()
        leer = lambda c: (GL.glGetString(c) or b"?").decode(errors="replace")  # noqa: E731
        render, version = leer(GL.GL_RENDERER), leer(GL.GL_VERSION)
        ctx.free()
    except Exception as e:  # noqa: BLE001
        fallo(f"No se pudo abrir OpenGL: {type(e).__name__}: {e}. Actualiza el driver de la tarjeta gráfica "
              "(no vale el 'Adaptador de pantalla básico de Microsoft'). Con escritorio remoto tampoco funciona")
        return
    print(f"  Python dibuja con: {render} (OpenGL {version.split()[0]})")
    if "gdi generic" in render.lower() or "basic render" in render.lower() or "llvmpipe" in render.lower():
        fallo("OpenGL va por software (sin driver de gráfica): instala el driver del fabricante")
    elif any(es_dedicada(t) for t in tarjetas) and not es_dedicada(render):
        aviso("El portátil tiene gráfica dedicada pero Python usa la integrada. Ejecuta: "
              "python diagnostico.py --gpu-dedicada  (y vuelve a lanzar este diagnóstico para comprobarlo)")
    else:
        ok("Driver de gráfica correcto")


def fisica() -> None:
    import numpy as np

    from robot_pendulo import RobotPenduloEnv

    titulo("Física")
    env = RobotPenduloEnv()
    env.reset(seed=0, options=dict(tipo_terreno="colinas", dificultad=1.0))
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < 3.0:
        _, _, term, trunc, _ = env.step(np.zeros(4))
        n += 1
        if term or trunc:
            env.reset()
    pasos_s = n / (time.perf_counter() - t0)
    x = pasos_s * env.dt
    env.close()
    (ok if x >= 5 else aviso)(f"{pasos_s:,.0f} pasos/s = {x:.0f}x tiempo real"
                              + ("" if x >= 5 else " (justo: cierra otros programas y conecta el cargador)"))


def probar_modelos(modelos: list[Path]) -> None:
    import numpy as np
    import torch

    from robot_pendulo import RobotPenduloEnv
    from robot_pendulo.utilidades import cargar_modelo, cargar_normalizador, kwargs_para_modelo

    torch.set_num_threads(1)
    for ruta in modelos:
        nombre = ruta.parent.name
        try:
            t0 = time.perf_counter()
            modelo, vn = cargar_modelo(ruta), cargar_normalizador(ruta)
            t_carga = time.perf_counter() - t0
            env = RobotPenduloEnv(**kwargs_para_modelo(modelo))
            obs, info = env.reset(seed=3, options=dict(tipo_terreno="plano", dificultad=0.0, vel_cmd=0.0))
            t0, pasos = time.perf_counter(), 0
            for pasos in range(1, 251):  # 5 s
                o = vn.normalize_obs(obs) if vn is not None else obs
                accion, _ = modelo.predict(o, deterministic=True)
                obs, _, term, trunc, info = env.step(accion)
                if term or trunc:
                    break
            ms = (time.perf_counter() - t0) / pasos * 1e3
            env.close()
            if not np.all(np.isfinite(obs)):
                fallo(f"{nombre}: la simulación dio NaN")
            elif info["cayo"]:
                aviso(f"{nombre}: carga, pero el péndulo cayó en terreno plano a los {pasos * env.dt:.1f} s")
            else:
                ok(f"{nombre}: carga en {t_carga:.1f} s, aguanta en plano, {ms:.1f} ms por paso (límite 20 ms)")
        except Exception as e:  # noqa: BLE001
            fallo(f"{nombre}: {type(e).__name__}: {e}")


def probar_render() -> str | None:
    import numpy as np

    from robot_pendulo import RobotPenduloEnv
    from robot_pendulo.utilidades import CALIDADES

    titulo("Dibujo 3D (960x540, como un vídeo)")
    tiempos = {}
    for calidad, kw in CALIDADES.items():
        try:
            env = RobotPenduloEnv(render_mode="rgb_array", **kw)
            env.reset(seed=0, options=dict(tipo_terreno="colinas", dificultad=1.0))
            for _ in range(3):
                env.render()
            t0 = time.perf_counter()
            for _ in range(30):
                env.step(np.zeros(4))
                env.render()
            tiempos[calidad] = (time.perf_counter() - t0) / 30 * 1e3
            env.close()
            print(f"  calidad {calidad:5s}: {tiempos[calidad]:6.1f} ms por imagen")
        except Exception as e:  # noqa: BLE001
            fallo(f"calidad {calidad}: {type(e).__name__}: {e}")
    if not tiempos:
        return None
    # La ventana suele ser más grande que 960x540: se deja margen sobre los 20 ms de cada paso.
    for calidad in ("alta", "media", "baja"):
        if calidad in tiempos and tiempos[calidad] <= 8.0:
            ok(f"Recomendado: --calidad {calidad}")
            return calidad
    aviso("Incluso la calidad baja va lenta: la ventana irá a saltos (la física sigue bien). "
          "Graba los vídeos de la demo en un PC con mejor gráfica y llévalos de respaldo")
    return "baja"


def rutas_largas_activadas() -> bool:
    if sys.platform != "win32":
        return True
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as k:
            return winreg.QueryValueEx(k, "LongPathsEnabled")[0] == 1
    except OSError:
        return False


def comprobar_ruta() -> bool:
    """Antes de instalar: ¿cabe PyTorch en .venv sin pasar del límite de rutas de Windows?"""
    largo = len(str(RAIZ)) + 1
    if largo <= LARGO_MAX_CARPETA or rutas_largas_activadas():
        return True
    print(f"\nLa ruta de esta carpeta es demasiado larga ({largo} caracteres, máximo {LARGO_MAX_CARPETA}):\n"
          f"  {RAIZ}\n"
          "PyTorch instala archivos con rutas de ~150 caracteres y Windows no admite más de 260 en\n"
          "total, así que la instalación fallaría. Soluciones (cualquiera de las dos):\n"
          "  1. Mueve la carpeta a una ruta corta, p. ej. C:\\robot-pendulo-rl, y ejecuta allí instalar.bat\n"
          "  2. Activa las rutas largas de Windows: abre PowerShell como administrador, ejecuta\n"
          "       New-ItemProperty -Path HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem "
          "-Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force\n"
          "     y vuelve a ejecutar instalar.bat")
    return False


def usar_gpu_dedicada() -> None:
    """Lo mismo que Configuración > Pantalla > Gráficos > python.exe > Alto rendimiento."""
    import winreg

    clave = r"Software\Microsoft\DirectX\UserGpuPreferences"
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, clave) as k:
        for exe in ejecutables_python():
            winreg.SetValueEx(k, exe, 0, winreg.REG_SZ, "GpuPreference=2;")
            print(f"Alto rendimiento (gráfica dedicada) para {exe}")
    print("\nCierra y vuelve a abrir la terminal y ejecuta: python diagnostico.py\n"
          "En 'Python dibuja con' debería salir la NVIDIA/AMD dedicada. Si no, en el Panel de control\n"
          "de NVIDIA > Administrar configuración 3D > Configuración de programa, añade esos python.exe\n"
          "y elige 'Procesador NVIDIA de alto rendimiento'.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu-dedicada", action="store_true",
                   help="configura Windows para que Python use la gráfica dedicada (portátiles con 2 gráficas)")
    p.add_argument("--comprobar-ruta", action="store_true",
                   help="solo comprobar que la ruta de la carpeta no es demasiado larga para instalar PyTorch")
    args = p.parse_args()
    os.chdir(RAIZ)
    if args.gpu_dedicada:
        usar_gpu_dedicada()
        return
    if args.comprobar_ruta:
        sys.exit(0 if comprobar_ruta() else 1)

    sistema()
    if not librerias():
        print("\nArregla las librerías antes de seguir.")
        sys.exit(1)
    modelos = archivos()
    probar_modelos(modelos)
    opengl()
    fisica()
    calidad = probar_render()

    titulo("Resumen")
    if not fallos and not avisos:
        print("  Todo bien.")
    for f in fallos:
        print(f"  FALLO  {f}")
    for a in avisos:
        print(f"  AVISO  {a}")
    if calidad:
        c = "" if calidad == "media" else f" --calidad {calidad}"
        print("\n  Comandos para la demo:")
        print(f"    python evaluar.py --pd --terreno colinas{c}")
        print(f"    python evaluar.py runs/ppo_10M_verificacion/modelo_final.zip{c}")
    sys.exit(1 if fallos else 0)


if __name__ == "__main__":
    main()
