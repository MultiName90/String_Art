# 1st_main.py — String Art (greedy) con heartbeat, logs y snapshots
# Requisitos: numpy, pillow, pandas, imageio
#   pip install numpy pillow pandas imageio

import os
import math
import json
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
import imageio.v2 as imageio


# =========================
# ===== CONFIG GLOBAL =====
# =========================
INPUT_DIR = "input"
OUTPUT_DIR = "output"
PROGRESS_DIR = os.path.join(OUTPUT_DIR, "progress")
HEARTBEAT_FILE = os.path.join(OUTPUT_DIR, "heartbeat.json")

# Crea carpetas si no existen
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROGRESS_DIR, exist_ok=True)

# Parámetros principales
W = H = 512               # tamaño de trabajo (cuadrado)
N_PEGS = 180              # número de clavos
K_LINES = 2500             # número de hilos a trazar
THREAD_WIDTH_PX = 1       # grosor visual del hilo (px) en el modelo
THREAD_ALPHA = 0.03       # opacidad/aporte por hilo (0-1)

# Vídeo / animación
FRAME_EVERY = 5           # añade 1 frame al GIF/MP4 cada N líneas
DRAW_FIRST_N_LINES = None # None = todas; o limita para vídeos ligeros

# Instrumentación / debug
LOG_EVERY = 20            # log a consola + heartbeat cada N líneas
SNAPSHOT_EVERY = 100      # guardar PNG de progreso cada N líneas

# Imagen personalizada (si quieres usar tu foto)
USE_CUSTOM_IMAGE = True
INPUT_IMAGE_PATH = os.path.join(INPUT_DIR, "mi_foto.png")
if USE_CUSTOM_IMAGE and not os.path.exists(INPUT_IMAGE_PATH):
    print(f"⚠ No se encontró {INPUT_IMAGE_PATH}. "
          f"Coloca tu imagen en '{INPUT_DIR}/' o desactiva USE_CUSTOM_IMAGE. "
          f"Se usará una silueta de ejemplo.")

# Rutas de salida
CSV_PATH = os.path.join(OUTPUT_DIR, "string_art_sequence.csv")
SVG_PATH = os.path.join(OUTPUT_DIR, "string_art.svg")
GIF_PATH = os.path.join(OUTPUT_DIR, "string_art_video.gif")
MP4_PATH = os.path.join(OUTPUT_DIR, "string_art_video.mp4")


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


def fmt_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# =========================
# ===== SOLVER GREEDY =====
# =========================
def string_art_greedy(
    target: np.ndarray,
    pegs: np.ndarray,
    k_lines: int,
    thread_alpha: float,
    thread_px: int,
    start_idx: int,
    progress_cb=None,
    snapshot_every: int | None = None
):
    """
    target: [H,W] en [0,1], 1 = oscuro que queremos cubrir
    progress_cb: callback opcional -> progress_cb(iter_idx, total, frm, to, best_gain, elapsed_s, render_snapshot_or_None)
    snapshot_every: si se define, enviará un render.copy() cada N iteraciones
    """
    t0 = time.perf_counter()
    h, w = target.shape
    residual = target.copy()
    render = np.zeros_like(target, dtype=np.float32)
    sequence = []
    current = start_idx

    for k in range(k_lines):
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
        prev, current = current, best_j

        # callback de progreso / snapshot
        if progress_cb:
            snap = None
            iter_idx = k + 1
            if snapshot_every and (iter_idx % snapshot_every == 0 or iter_idx == k_lines):
                snap = render.copy()
            elapsed_s = time.perf_counter() - t0
            progress_cb(iter_idx, k_lines, prev, current, best_gain, elapsed_s, snap)

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
        pass


# =========================
# ========= MAIN ==========
# =========================
def main():
    print("=== String Art (greedy) — debug en vivo ===")
    print(f"Clavos: {N_PEGS} | Hilos: {K_LINES} | Alpha: {THREAD_ALPHA} | Width(px): {THREAD_WIDTH_PX}")
    print(f"Log cada {LOG_EVERY} líneas | Snapshot cada {SNAPSHOT_EVERY} | FrameEvery={FRAME_EVERY}")
    print(f"Imagen: {'custom' if USE_CUSTOM_IMAGE else 'silueta'} -> {INPUT_IMAGE_PATH if USE_CUSTOM_IMAGE else '-'}")
    print("Salida en:", OUTPUT_DIR, "\n", flush=True)

    target = load_target_image(W, H, USE_CUSTOM_IMAGE, INPUT_IMAGE_PATH)
    pegs = make_pegs(N_PEGS, W, H)

    t0 = time.perf_counter()

    def progress_cb(iter_idx, total, frm, to, best_gain, elapsed_s, snap):
        # métricas
        lps = iter_idx / max(1e-9, elapsed_s)  # lines per second
        eta_s = (total - iter_idx) * (elapsed_s / iter_idx)

        # log consola
        if iter_idx == 1 or (iter_idx % LOG_EVERY == 0) or (iter_idx == total):
            print(f"[{iter_idx:4d}/{total}] gain={best_gain:.1f} | "
                  f"elapsed={elapsed_s:7.2f}s | lps={lps:5.2f} | ETA={fmt_eta(eta_s)}",
                  flush=True)

            # heartbeat
            hb = {
                "iter": int(iter_idx),
                "total": int(total),
                "elapsed_s": round(elapsed_s, 3),
                "lps": round(lps, 3),
                "eta_s": int(max(0, eta_s)),
                "last_gain": float(best_gain),
                "last_from": int(frm),
                "last_to": int(to),
                "outputs": {
                    "csv": CSV_PATH, "svg": SVG_PATH, "gif": GIF_PATH, "mp4": MP4_PATH
                }
            }
            with open(HEARTBEAT_FILE, "w", encoding="utf-8") as f:
                json.dump(hb, f, ensure_ascii=False, indent=2)

        # snapshot
        if snap is not None:
            # guardamos imagen del render actual (0..1) como PNG
            snap_img = (np.clip(snap, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(snap_img, mode="L").save(
                os.path.join(PROGRESS_DIR, f"render_{iter_idx:04d}.png")
            )

    # Ejecuta solver
    render, residual, sequence = string_art_greedy(
        target, pegs,
        k_lines=K_LINES,
        thread_alpha=THREAD_ALPHA,
        thread_px=THREAD_WIDTH_PX,
        start_idx=0,
        progress_cb=progress_cb,
        snapshot_every=SNAPSHOT_EVERY
    )

    # Exporta resultados
    export_sequence(sequence, pegs, W, H, CSV_PATH, SVG_PATH, GIF_PATH, MP4_PATH,
                    frame_every=FRAME_EVERY, draw_first_n=DRAW_FIRST_N_LINES)

    elapsed = time.perf_counter() - t0
    print("\n✅ Listo.")
    print(f"Tiempo total: {elapsed:.2f}s")
    print(f"CSV: {CSV_PATH}")
    print(f"SVG: {SVG_PATH}")
    print(f"GIF: {GIF_PATH}")
    print(f"MP4: {MP4_PATH} (si no se generó, usa el GIF)")


if __name__ == "__main__":
    main()
