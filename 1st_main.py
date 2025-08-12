# Re-ejecución completa tras reset del estado: generamos archivos y devolvemos rutas.
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import math
import pandas as pd
import os
import imageio.v2 as imageio

W = H = 512
N_PEGS = 180
K_LINES = 500           # reducimos ligeramente para asegurar ejecución estable
THREAD_WIDTH_PX = 1
THREAD_ALPHA = 0.03
FRAME_EVERY = 5
DRAW_FIRST_N_LINES = None
USE_CUSTOM_IMAGE = False
INPUT_IMAGE_PATH = "/mnt/data/tu_imagen.png"

def load_target_image(w, h, use_custom=False, path=None):
    if use_custom and path and os.path.exists(path):
        im = Image.open(path).convert("L")
        im = ImageOps.fit(im, (w, h), method=Image.Resampling.LANCZOS)
        arr = np.array(im).astype(np.float32) / 255.0
        arr = 1.0 - arr
        return arr
    else:
        img = Image.new("L", (w, h), 255)
        draw = ImageDraw.Draw(img)
        cx, cy = w//2, h//2 + 20
        rx, ry = int(w*0.18), int(h*0.16)
        draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=0)
        draw.polygon([(cx-40, cy-ry+10), (cx-80, cy-ry-45), (cx-10, cy-ry+15)], fill=0)
        draw.polygon([(cx+40, cy-ry+10), (cx+80, cy-ry-45), (cx+10, cy-ry+15)], fill=0)
        draw.ellipse([cx-35, cy-10, cx-10, cy+15], fill=255)
        draw.ellipse([cx+10, cy-10, cx+35, cy+15], fill=255)
        img_small = img.resize((w//2, h//2), Image.Resampling.BICUBIC)
        img = img_small.resize((w, h), Image.Resampling.BICUBIC)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = 1.0 - arr
        return arr

def make_pegs(n_pegs, w, h, radius_ratio=0.46):
    R = min(w, h) * radius_ratio
    center = np.array([w/2, h/2], dtype=np.float32)
    thetas = 2 * np.pi * np.arange(n_pegs) / n_pegs
    xs = center[0] + R * np.cos(thetas)
    ys = center[1] + R * np.sin(thetas)
    return np.stack([xs, ys], axis=1).astype(np.float32)

def line_mask(w, h, p0, p1, width=1):
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    d.line([tuple(p0), tuple(p1)], fill=255, width=width)
    return (np.array(im).astype(np.float32) / 255.0)

def string_art_greedy(target, pegs, k_lines=500, thread_alpha=0.03, thread_px=1, start_idx=0):
    h, w = target.shape
    residual = target.copy()
    render = np.zeros_like(target, dtype=np.float32)
    sequence = []
    current = start_idx
    for _ in range(k_lines):
        best_gain = -1.0
        best_j = None
        best_mask = None
        for j in range(len(pegs)):
            if j == current:
                continue
            m = line_mask(w, h, pegs[current], pegs[j], width=thread_px)
            gain = float((residual * m).sum())
            if gain > best_gain:
                best_gain = gain
                best_j = j
                best_mask = m
        render = np.clip(render + thread_alpha * best_mask, 0, 1)
        residual = np.clip(residual - thread_alpha * best_mask, 0, 1)
        sequence.append((current, best_j))
        current = best_j
    return render, residual, sequence

def make_animation(pegs, sequence, w, h, frame_every=5, draw_first_n=None, out_gif="/mnt/data/string_art_video.gif", out_mp4="/mnt/data/string_art_video.mp4"):
    frames = []
    base = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(base)
    for (x, y) in pegs:
        draw.ellipse([x-2, y-2, x+2, y+2], fill=(0, 0, 0))
    current_img = base.copy()
    d = ImageDraw.Draw(current_img)
    steps = sequence if draw_first_n is None else sequence[:draw_first_n]
    for idx, (a, b) in enumerate(steps, start=1):
        d.line([tuple(pegs[a]), tuple(pegs[b])], width=1, fill=(0, 0, 0))
        if idx == 1 or (idx % frame_every == 0) or (idx == len(steps)):
            frames.append(current_img.copy())
    imageio.mimsave(out_gif, [np.array(f) for f in frames], duration=0.05)
    mp4_ok = False
    try:
        imageio.mimsave(out_mp4, [np.array(f) for f in frames], fps=20)
        mp4_ok = True
    except Exception:
        mp4_ok = False
    return out_gif, (out_mp4 if mp4_ok else None)

# Ejecutar
target = load_target_image(W, H, use_custom=USE_CUSTOM_IMAGE, path=INPUT_IMAGE_PATH)
pegs = make_pegs(N_PEGS, W, H)
render, residual, sequence = string_art_greedy(target, pegs, k_lines=K_LINES, thread_alpha=THREAD_ALPHA, thread_px=THREAD_WIDTH_PX, start_idx=0)

# Exportar CSV y SVG
csv_path = "/mnt/data/string_art_sequence.csv"
pd.DataFrame(sequence, columns=["from_peg", "to_peg"]).to_csv(csv_path, index=False)
svg_path = "/mnt/data/string_art.svg"
parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n']
for (x,y) in pegs:
    parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1" fill="black" />\n')
for (a,b) in sequence:
    x1,y1 = pegs[a]
    x2,y2 = pegs[b]
    parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="black" stroke-width="0.6"/>\n')
parts.append('</svg>\n')
with open(svg_path, "wb") as f:
    f.write("".join(parts).encode("utf-8"))

# Animación
gif_path, mp4_path = make_animation(pegs, sequence, W, H, frame_every=FRAME_EVERY, draw_first_n=DRAW_FIRST_N_LINES)

print({"csv": csv_path, "svg": svg_path, "gif": gif_path, "mp4": mp4_path})
