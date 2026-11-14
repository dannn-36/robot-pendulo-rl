"""Crea una pista de ejemplo como mapa de alturas (PNG de 16 bits) para probar el modelo.

    python crear_terreno_ejemplo.py
    python evaluar.py runs/ppo_10M_verificacion/modelo_final.zip --mapa terrenos/pista_ejemplo.png --tamano 24 --altura-blanco 0.6 --inicio -10 0 --rumbo 0 --vel-cmd 0.5 --duracion 50

La pista va de oeste a este (de izquierda a derecha en la imagen) por el centro del
mapa. El robot sale en x = -10 m mirando al este y la recorre en este orden:

    salida plana -> rampa de subida 8° -> meseta con baches -> rampa de bajada 8°
    -> grava (baches de ±2 cm) -> escalones de 3 cm -> ondulaciones -> pendiente lateral 6° -> meta

Úsalo como plantilla: cada tramo es una función de x (y de y) en metros. También puedes
dibujar tu propio mapa en cualquier editor de imágenes (GIMP, Photoshop, Krita...):
escala de grises, negro = suelo más bajo, blanco = `--altura-blanco` metros.
"""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np

LADO = 24.0          # m que mide el mapa
RES = 0.05           # m por píxel
ALTURA_BLANCO = 0.6  # m que representa el blanco en el PNG


def pista(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Altura (m) de la pista en cada punto."""
    t8 = np.tan(np.radians(8))
    h = np.full_like(x, 0.10)
    # Rampa de subida y bajada con meseta
    h += np.clip((x + 8.0) * t8, 0, 2.0 * t8) - np.clip((x + 4.0) * t8, 0, 2.0 * t8)
    meseta = (x > -6) & (x < -4)
    h += meseta * 0.015 * np.sin(2 * np.pi * x / 0.6) * np.sin(2 * np.pi * y / 0.7)
    # Grava: baches aleatorios de ±2 cm (celdas de 15 cm)
    grava = (x > -2) & (x < 0)
    ruido = rng.uniform(-0.02, 0.02, size=x.shape)
    k = 3  # 3 píxeles = 15 cm
    ruido = ruido[::k, ::k].repeat(k, axis=0).repeat(k, axis=1)[:x.shape[0], :x.shape[1]]
    h += grava * ruido
    # Escalones de 3 cm: sube, sube, baja, baja
    h += 0.03 * ((x > 0.5).astype(float) + (x > 1.0) - (x > 1.5) - (x > 2.0))
    # Ondulaciones de 5 cm de amplitud
    h += ((x > 2.0) & (x < 5.0)) * 0.05 * np.sin(2 * np.pi * (x - 2.0) / 1.5)
    # Pendiente lateral de 6° (sube hacia el norte), con transición suave
    lateral = np.clip((x - 5.0) / 0.5, 0, 1) * np.clip((8.5 - x) / 0.5, 0, 1)
    h += lateral * np.tan(np.radians(6)) * y
    return h


def main() -> None:
    rng = np.random.default_rng(0)
    n = int(round(LADO / RES)) + 1
    coords = np.linspace(-LADO / 2, LADO / 2, n)
    x, y = np.meshgrid(coords, coords)   # fila = y (sur -> norte), columna = x (oeste -> este)

    h_pista = pista(x, y, rng)
    # Fuera del pasillo central (|y| > 2 m): colinas suaves
    colinas = 0.10 + 0.08 * np.sin(x / 2.3) * np.cos(y / 1.9) + 0.05 * np.sin((x + y) / 1.3)
    peso = np.clip((3.0 - np.abs(y)) / 1.0, 0, 1)   # 1 en el pasillo, 0 a partir de |y| = 3 m
    peso = peso * peso * (3 - 2 * peso)
    h = peso * h_pista + (1 - peso) * colinas

    h -= h.min()
    assert h.max() <= ALTURA_BLANCO, f"desnivel {h.max():.2f} m > {ALTURA_BLANCO} m"
    # La primera fila de la imagen es el norte: se invierte el eje y
    img = np.flipud(np.round(h / ALTURA_BLANCO * 65535)).astype(np.uint16)
    salida = Path("terrenos") / "pista_ejemplo.png"
    salida.parent.mkdir(exist_ok=True)
    imageio.imwrite(salida, img)
    print(f"Guardado {salida} ({n}x{n} px, {RES * 100:.0f} cm/px, lado {LADO:.0f} m, "
          f"desnivel {h.max():.2f} m, blanco = {ALTURA_BLANCO} m)")
    print(f"Pruébalo con:\n  python evaluar.py runs/ppo_10M_verificacion/modelo_final.zip --mapa {salida.as_posix()} "
          f"--tamano {LADO:.0f} --altura-blanco {ALTURA_BLANCO} --inicio -10 0 --rumbo 0 --vel-cmd 0.5 --duracion 50")


if __name__ == "__main__":
    main()
