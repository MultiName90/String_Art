# 1st_main.py — String Art (análisis + vídeo línea a línea)
# Autor: Fran + ChatGPT
# Requisitos: numpy, pillow, pandas, imageio
#   pip install numpy pillow pandas imageio

import os
import math
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
import imageio.v2 as imageio


# =========================
# ===== CONFIG GLOBAL =====
# =========================
INPUT_DIR = "input"
OUTPUT_DIR = "output"

# Crea carpetas si no existen
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parámetros principales
W = H = 512               # tamaño de trabajo (cuadrado)
N_PEGS = 180              # número de clavos
K_LINES = 2500             # número de hilos a trazar
THREAD_WIDTH_PX = 1       # grosor visual del hilo (px) en el modelo
THREAD_ALPHA = 0.03       # opacidad/aporte por hilo (0-1)
FRAME_EVERY = 5           # añade 1 frame al vídeo cada N líneas
DRAW_FIRST_N_LINES = None # None = todas; o limita para vídeos ligeros

# Imagen personalizada (si quieres usar tu foto)
USE_CUSTOM_IMAGE = True
INPUT_IMAGE_PATH = os.path.join(INPUT_DIR, "mi_foto.png")

if USE_CUSTOM_IMAGE and not os.path.exists(INPUT_IMAGE_PATH):
    print(f"⚠ No se encontró {INPUT_IMAGE_PATH}. "
          f"Coloca tu imagen en '{INPUT_DIR}/' o desactiva USE_CUSTOM_IMAGE. "
          f"Se usará una silueta de ejemplo.")


# =========================
# ===== UTILIDADES IMG ====
# =========================
def load_target_image(w: int, h: int, use_custom: bool, path: str) -> np.ndarray:
    """
    Devuelve matriz [H,W] en [0,1] donde 1 = zona a oscurecer con hilos (negro).
    Si no hay imagen personalizada, genera una silueta simple.
    """
    if use_custom and path and os.path.exists(path):
        im = Image.open(path).convert("L")
        im = ImageOps.fit(im, (w, h), method=Image.Resampling.LANCZOS)
        arr = np.array(im).astype(np.float32) / 255.0
        arr = 1.0 - arr   # invertimos: 1 = oscuro (hilo), 0 = claro (fondo)
        return arr

    # Silueta de ejemplo (gato)
    img = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(img)
    cx, cy = w // 2, h // 2 + 20
    rx, ry = int(w * 0.18), int(h * 0.16)
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=0)
    draw.polygon([(cx - 40, cy - ry + 10), (cx - 80, cy - ry - 45), (cx - 10, cy - ry + 15)], fill=0)
    draw.polygon([(cx + 40, cy - ry + 10), (cx + 80, cy - ry - 45), (cx + 10, cy - ry + 15)], fill=0)
    # ojos recortados
    draw.ellipse([cx - 35, cy - 10, cx - 10, cy + 15], fill=255)
    draw.ellipse([cx + 10, cy - 10, cx + 35, cy + 15], fill=255)
    # suavizado básico
    img_small = img.resize((w // 2, h // 2), Image.Resampling.BICUBIC)
    img = img_small.resize((w, h), Image.Resampling.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = 1.0 - arr
    return arr


def make_pegs(n_pegs: int, w: int, h: int, radius_ratio: float = 0.46) -> np.ndarray:
    """Devuelve array [N,2] con coordenadas (x,y) de clavos en circunferencia."""
    R = min(w, h) * radius_ratio
    center = np.array([w / 2, h / 2], dtype=np.float32)
    thetas = 2 * np.pi * np.arange(n_pegs) / n_pegs
    xs = center[0] + R * np.cos(thetas)
    ys = center[1] + R * np.sin(thetas)
    return np.stack([xs, ys], axis=1).astype(np.float32)


def line_mask(w: int, h: int, p0: np.ndarray, p1: np.ndarray, width: int = 1) -> np.ndarray:
    """Máscara binaria de una línea entre p0 y p1 sobre imagen WxH."""
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    d.line([tuple(p0), tuple(p1)], fill=255, width=width)
    return (np.array(im).astype(np.float32) / 255.0)


# =========================
# ===== SOLVER GREEDY =====
# =========================
def string_art_greedy(
    target: np.ndarray,
    pegs: np.ndarray,
    k_lines: int,
    thread_alpha: float,
    thread_px: int,
    start_idx: int = 0
):
    """
    target: [H,W] en [0,1], 1 = oscuro que queremos cubrir
    Devuelve: render, residual, sequence [(from,to), ...]
    """
    h, w = target.shape
    residual = target.copy()
    render = np.zeros_like(target, dtype=np.float32)
    sequence = []
    current = start_idx

    for _ in range(k_lines):
        best_gain = -1.0
        best_j = None
        best_mask = None

        # probar todas las posibles siguientes conexiones
        for j in range(len(pegs)):
            if j == current:
                continue
            m = line_mask(w, h, pegs[current], pegs[j], width=thread_px)
            gain = float((residual * m).sum())
            if gain > best_gain:
                best_gain = gain
                best_j = j
                best_mask = m

        # aplicar la mejor línea encontrada
        render = np.clip(render + thread_alpha * best_mask, 0, 1)
        residual = np.clip(residual - thread_alpha * best_mask, 0, 1)
        sequence.append((current, best_j))
        current = best_j

    return render, residual, sequence


# =========================
# ===== EXPORT/VIDEO  =====
# =========================
def export_sequence(sequence, pegs, w, h,
                    csv_path, svg_path,
                    gif_path, mp4_path,
                    frame_every=5, draw_first_n=None):
    """Guarda CSV, SVG y animación (GIF y MP4 si es posible)."""
    # CSV
    pd.DataFrame(sequence, columns=["from_peg", "to_peg"]).to_csv(csv_path, index=False)

    # SVG
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n']
    for (x, y) in pegs:
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1" fill="black" />\n')
    for (a, b) in sequence:
        x1, y1 = pegs[a]
        x2, y2 = pegs[b]
        parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="black" stroke-width="0.6"/>\n')
    parts.append('</svg>\n')
    with open(svg_path, "wb") as f:
        f.write("".join(parts).encode("utf-8"))

    # GIF/MP4
    base = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(base)
    for (x, y) in pegs:
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(0, 0, 0))

    frames = []
    current_img = base.copy()
    d = ImageDraw.Draw(current_img)
    steps = sequence if draw_first_n is None else sequence[:draw_first_n]

    for idx, (a, b) in enumerate(steps, start=1):
        d.line([tuple(pegs[a]), tuple(pegs[b])], width=1, fill=(0, 0, 0))
        if idx == 1 or (idx % frame_every == 0) or (idx == len(steps)):
            frames.append(current_img.copy())

    imageio.mimsave(gif_path, [np.array(f) for f in frames], duration=0.05)

    # MP4 (si el backend lo permite)
    try:
        imageio.mimsave(mp4_path, [np.array(f) for f in frames], fps=20)
    except Exception:
        # No interrumpe la ejecución si no es posible
        pass


# =========================
# ========= MAIN ==========
# =========================
def main():
    # Rutas de salida
    csv_path = os.path.join(OUTPUT_DIR, "string_art_sequence.csv")
    svg_path = os.path.join(OUTPUT_DIR, "string_art.svg")
    gif_path = os.path.join(OUTPUT_DIR, "string_art_video.gif")
    mp4_path = os.path.join(OUTPUT_DIR, "string_art_video.mp4")

    # Carga imagen objetivo y genera clavos
    target = load_target_image(W, H, USE_CUSTOM_IMAGE, INPUT_IMAGE_PATH)
    pegs = make_pegs(N_PEGS, W, H)

    # Resolver (greedy)
    render, residual, sequence = string_art_greedy(
        target, pegs,
        k_lines=K_LINES,
        thread_alpha=THREAD_ALPHA,
        thread_px=THREAD_WIDTH_PX,
        start_idx=0
    )

    # Exportar secuencia y animación
    export_sequence(sequence, pegs, W, H, csv_path, svg_path, gif_path, mp4_path,
                    frame_every=FRAME_EVERY, draw_first_n=DRAW_FIRST_N_LINES)

    print("✅ Listo.")
    print(f"CSV: {csv_path}")
    print(f"SVG: {svg_path}")
    print(f"GIF: {gif_path}")
    print(f"MP4: {mp4_path} (si no se generó, usa el GIF)")


if __name__ == "__main__":
    main()
