"""Generación procedural de terrenos para el campo de alturas (heightfield) de MuJoCo.

Cada terreno se genera con una `dificultad` en [0, 1]: 0 es plano y 1 usa los
valores máximos de pendiente, rugosidad y altura de escalón configurados.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

TIPOS_TERRENO = ("plano", "rampas", "colinas", "rugoso", "escalones", "mixto")


def cargar_mapa_alturas(ruta: str | Path, altura_blanco: float = 0.5) -> np.ndarray:
    """Lee un mapa de alturas propio y lo devuelve en metros, listo para `usar_alturas`.

    El archivo se interpreta como un mapa visto desde arriba: arriba = norte (+y),
    derecha = este (+x). Un robot con rumbo 0° mira hacia la derecha del mapa.

    - Imagen (.png, .jpg, .bmp, .tif): escala de grises, negro = 0 m y blanco =
      `altura_blanco` m. Un PNG de 16 bits da mucha más precisión que uno de 8.
    - .npy / .csv / .txt: matriz con las alturas directamente en metros.
    """
    ruta = Path(ruta)
    ext = ruta.suffix.lower()
    if ext == ".npy":
        a = np.load(ruta).astype(np.float64)
    elif ext in (".csv", ".txt"):
        a = np.loadtxt(ruta, delimiter="," if ext == ".csv" else None, ndmin=2)
    else:
        import imageio.v2 as imageio

        img = imageio.imread(ruta)
        tipo = img.dtype
        if img.ndim == 3:  # color o con transparencia -> gris
            img = img[..., :3].mean(axis=2)
        if tipo == np.uint8:
            maximo = 255.0
        elif tipo == np.uint16:
            maximo = 65535.0
        else:
            maximo = float(np.max(img)) or 1.0
        a = np.asarray(img, dtype=np.float64) / maximo * altura_blanco
    return np.flipud(a)  # la primera fila del archivo es el norte (+y)


class GeneradorTerreno:
    """Mantiene una matriz de alturas `alturas[fila, columna]`.

    La fila i corresponde a y = -L + i*res y la columna j a x = -L + j*res,
    que es como MuJoCo interpreta `hfield_data`.
    """

    def __init__(self, n: int, semitamano: float, altura_max: float,
                 pendiente_max_grados: float = 15.0, rugosidad_max: float = 0.03,
                 escalon_max: float = 0.05):
        self.n = n
        self.L = semitamano
        self.altura_max = altura_max
        self.res = 2 * semitamano / (n - 1)
        self.pendiente_max = np.deg2rad(pendiente_max_grados)
        self.rugosidad_max = rugosidad_max
        self.escalon_max = escalon_max
        self.alturas = np.zeros((n, n))

    # ------------------------------------------------------------------ generación
    def generar(self, tipo: str, dificultad: float, rng: np.random.Generator) -> np.ndarray:
        d = float(np.clip(dificultad, 0.0, 1.0))
        n = self.n
        if tipo == "plano" or d == 0.0:
            h = np.zeros((n, n))
        elif tipo == "rampas":
            # Planos inclinados enlazados (interpolación lineal de una malla gruesa).
            h = self._ruido(rng, celda=rng.uniform(2.0, 4.0), suave=False)
            h = self._a_pendiente(h, d * self.pendiente_max)
        elif tipo == "colinas":
            h = self._ruido(rng, celda=rng.uniform(4.0, 8.0)) \
                + 0.4 * self._ruido(rng, celda=rng.uniform(1.5, 3.0))
            h = self._a_pendiente(h, d * self.pendiente_max)
        elif tipo == "rugoso":
            # Baches de alta frecuencia (grava, piedras pequeñas) + ondulación suave.
            fino = self._ruido(rng, celda=rng.uniform(0.10, 0.25), suave=False)
            base = self._a_pendiente(self._ruido(rng, celda=5.0), 0.3 * d * self.pendiente_max)
            h = base + d * self.rugosidad_max * fino
        elif tipo == "escalones":
            # Bloques planos de distinta altura (bordillos, losas, tablones).
            bloques = self._bloques(rng, tam=rng.uniform(0.4, 1.5))
            h = d * self.escalon_max * bloques
        elif tipo == "mixto":
            col = self._a_pendiente(self._ruido(rng, celda=rng.uniform(4.0, 8.0)),
                                    0.7 * d * self.pendiente_max)
            fino = self._ruido(rng, celda=rng.uniform(0.10, 0.25), suave=False)
            bloques = self._bloques(rng, tam=rng.uniform(0.6, 2.0))
            h = col + 0.7 * d * self.rugosidad_max * fino + 0.5 * d * self.escalon_max * bloques
        else:
            raise ValueError(f"Tipo de terreno desconocido: {tipo!r}. Opciones: {TIPOS_TERRENO}")

        h = h - h.min()
        self.alturas = np.clip(h, 0.0, self.altura_max * 0.999)
        return self.alturas

    def usar_alturas(self, alturas: np.ndarray, tamano: float | None = None) -> np.ndarray:
        """Usa un mapa de alturas propio (en metros) en vez de uno generado.

        `alturas[fila, columna]`: la fila 0 es el borde sur (y mínima) y la columna 0 el
        borde oeste (x mínima). El mapa se centra en el origen y se reescala para que su
        lado mayor mida `tamano` metros (por defecto, todo el terreno); fuera de él se
        prolonga la altura del borde.
        """
        a = np.asarray(alturas, dtype=np.float64)
        if a.ndim != 2 or min(a.shape) < 2:
            raise ValueError(f"El mapa de alturas debe ser una matriz 2D, no {a.shape}")
        if not np.all(np.isfinite(a)):
            raise ValueError("El mapa de alturas contiene NaN o infinitos")
        filas, cols = a.shape
        lado = 2 * self.L if tamano is None else min(float(tamano), 2 * self.L)
        ancho = lado * min(1.0, (cols - 1) / (filas - 1))   # extensión en x
        alto = lado * min(1.0, (filas - 1) / (cols - 1))    # extensión en y

        # Interpolación bilineal del mapa sobre la malla del terreno
        coords = np.linspace(-self.L, self.L, self.n)
        fc = np.clip((coords + ancho / 2) / ancho * (cols - 1), 0, cols - 1)
        fr = np.clip((coords + alto / 2) / alto * (filas - 1), 0, filas - 1)
        c0 = np.minimum(fc.astype(int), cols - 2)
        r0 = np.minimum(fr.astype(int), filas - 2)
        tc, tr = fc - c0, fr - r0
        tmp = a[:, c0] * (1 - tc) + a[:, c0 + 1] * tc
        h = tmp[r0, :] * (1 - tr)[:, None] + tmp[r0 + 1, :] * tr[:, None]

        h = h - h.min()
        if h.max() >= self.altura_max:
            print(f"Aviso: el terreno tiene {h.max():.2f} m de desnivel; se recorta a {self.altura_max:.2f} m "
                  f"(sube `altura_max_terreno` si necesitas más).")
        self.alturas = np.clip(h, 0.0, self.altura_max * 0.999)
        return self.alturas

    def datos_hfield(self) -> np.ndarray:
        """Alturas normalizadas a [0, 1] tal como las espera MuJoCo."""
        return (self.alturas / self.altura_max).astype(np.float32).ravel()

    def _ruido(self, rng, celda: float, suave: bool = True) -> np.ndarray:
        """Ruido de valores: malla aleatoria cada `celda` metros, interpolada."""
        paso = max(celda / self.res, 1.0)
        m = int(np.ceil((self.n - 1) / paso)) + 2
        g = rng.uniform(-1.0, 1.0, size=(m, m)).astype(np.float32)
        t = np.arange(self.n, dtype=np.float32) / np.float32(paso)
        i0 = np.floor(t).astype(int)
        f = t - i0
        if suave:
            f = f * f * (3.0 - 2.0 * f)
        a = g[:, i0] * (1 - f) + g[:, i0 + 1] * f
        return a[i0, :] * (1 - f)[:, None] + a[i0 + 1, :] * f[:, None]

    def _bloques(self, rng, tam: float) -> np.ndarray:
        paso = max(tam / self.res, 1.0)
        m = int(np.ceil(self.n / paso)) + 1
        g = rng.uniform(0.0, 1.0, size=(m, m))
        idx = (np.arange(self.n) / paso).astype(int)
        return g[np.ix_(idx, idx)]

    def _a_pendiente(self, h: np.ndarray, pendiente_rad: float) -> np.ndarray:
        """Escala `h` para que su pendiente máxima sea `pendiente_rad`."""
        gx = np.diff(h, axis=1)[:-1, :] / self.res
        gy = np.diff(h, axis=0)[:, :-1] / self.res
        g = float(np.sqrt(gx * gx + gy * gy).max())
        return h * np.float32(np.tan(pendiente_rad) / g) if g > 1e-9 else h

    # ------------------------------------------------------------------ consultas
    def altura_en(self, x: float, y: float) -> float:
        """Altura del terreno en (x, y) con interpolación bilineal."""
        tope = self.n - 1.001
        c = min(max((x + self.L) / self.res, 0.0), tope)
        r = min(max((y + self.L) / self.res, 0.0), tope)
        c0, r0 = int(c), int(r)
        fc, fr = c - c0, r - r0
        h = self.alturas
        return float((h[r0, c0] * (1 - fc) + h[r0, c0 + 1] * fc) * (1 - fr)
                     + (h[r0 + 1, c0] * (1 - fc) + h[r0 + 1, c0 + 1] * fc) * fr)

    def altura_max_en(self, x: float, y: float, radio: float) -> float:
        """Altura máxima del terreno dentro de un cuadrado de semilado `radio`."""
        idx = lambda v: int(min(max(v / self.res, 0.0), self.n - 1))  # noqa: E731
        c0, c1 = idx(x - radio + self.L), idx(x + radio + self.L) + 1
        r0, r1 = idx(y - radio + self.L), idx(y + radio + self.L) + 1
        return float(self.alturas[r0:r1, c0:c1].max())

    def gradiente_en(self, x: float, y: float, d: float = 0.25) -> tuple[float, float]:
        """(dh/dx, dh/dy) promediado en una ventana de ±d metros."""
        gx = (self.altura_en(x + d, y) - self.altura_en(x - d, y)) / (2 * d)
        gy = (self.altura_en(x, y + d) - self.altura_en(x, y - d)) / (2 * d)
        return gx, gy
