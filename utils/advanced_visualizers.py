"""Advanced forensic visualizers — v4.0.

Provides rich visual evidence for the analysis results:
  - Face region heatmap (forehead / eyes / nose / cheeks / mouth / jaw)
  - Blending boundary detection (face-swap edge artifacts)
  - 3D FFT power spectrum (Plotly surface)
  - Multi-panel forensic grid (ELA + FFT + Noise SRM + NPR map)
  - Face mesh landmark overlay (MediaPipe or OpenCV fallback)
  - Compression artifact map (block-DCT variance)
  - Half-face asymmetry map (left vs right swap artifacts)
  - Frequency anomaly radial chart (Plotly polar)
  - Audio forensic panel (waveform + LFCC + mel spectrogram)
  - Suspicious frame contact sheet for video
"""
from __future__ import annotations

import io
import logging
import math
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw, ImageFilter, ImageChops, ImageEnhance
import plotly.graph_objects as go
import plotly.express as px

matplotlib.use("Agg")
log = logging.getLogger("deepfake.visualizers")

# ── Colour palettes ───────────────────────────────────────────────────────────
_DARK_BG = "#04070f"
_GRID_COLOR = "rgba(51,65,85,0.5)"
_FONT_COLOR = "#94a3b8"


# ══════════════════════════════════════════════════════════════════════════════
# 1.  FACE REGION HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
_FACE_REGIONS = {
    # name: (x_frac_start, x_frac_end, y_frac_start, y_frac_end)
    "Forehead":  (0.20, 0.80, 0.04, 0.22),
    "Left Eye":  (0.10, 0.44, 0.22, 0.40),
    "Right Eye": (0.56, 0.90, 0.22, 0.40),
    "Nose":      (0.30, 0.70, 0.38, 0.62),
    "L-Cheek":   (0.05, 0.38, 0.45, 0.72),
    "R-Cheek":   (0.62, 0.95, 0.45, 0.72),
    "Mouth":     (0.28, 0.72, 0.62, 0.82),
    "Jaw/Chin":  (0.20, 0.80, 0.80, 0.97),
}


def _srm_noise(gray: np.ndarray) -> np.ndarray:
    """Simple SRM high-pass residual for noise analysis."""
    import cv2
    k = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32) / 4
    return cv2.filter2D(gray.astype(np.float32), -1, k)


def face_region_heatmap(
    img: Image.Image,
    face_box: Optional[Tuple[int, int, int, int]] = None,
) -> go.Figure:
    """Plotly bar chart showing per-region noise suspicion score."""
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    import cv2
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    noise = np.abs(_srm_noise(gray))

    if face_box is not None:
        x1, y1, x2, y2 = [int(v) for v in face_box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.width, x2), min(img.height, y2)
        bw, bh = x2 - x1, y2 - y1
        origin = (x1, y1)
    else:
        bw, bh = img.width, img.height
        origin = (0, 0)

    scores, names = [], []
    for name, (fx0, fx1, fy0, fy1) in _FACE_REGIONS.items():
        rx0 = origin[0] + int(fx0 * bw)
        rx1 = origin[0] + int(fx1 * bw)
        ry0 = origin[1] + int(fy0 * bh)
        ry1 = origin[1] + int(fy1 * bh)
        rx0, rx1 = max(0, rx0), min(gray.shape[1], rx1)
        ry0, ry1 = max(0, ry0), min(gray.shape[0], ry1)
        patch = noise[ry0:ry1, rx0:rx1]
        scores.append(float(patch.mean()) if patch.size else 0.0)
        names.append(name)

    # Normalize to 0-100.
    m = max(scores) + 1e-9
    pcts = [s / m * 100 for s in scores]
    colors = [
        f"hsl({int(120 * (1 - p / 100))},80%,55%)" for p in pcts
    ]

    fig = go.Figure(go.Bar(
        x=names, y=pcts,
        marker_color=colors,
        text=[f"{p:.0f}" for p in pcts],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text="🔬 Per-Region Noise Suspicion Score", font_color="#38bdf8"),
        yaxis_title="Relative noise anomaly (%)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,14,32,0.6)",
        font_color=_FONT_COLOR,
        xaxis=dict(gridcolor=_GRID_COLOR),
        yaxis=dict(gridcolor=_GRID_COLOR, range=[0, 115]),
        height=280,
        margin=dict(t=50, b=40, l=50, r=20),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 2.  BLENDING BOUNDARY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def blending_boundary_map(img: Image.Image) -> Image.Image:
    """Detect face-swap blending boundaries using gradient discontinuity.

    Face-swapping tools blend the source face onto the target with a mask.
    The mask boundary shows characteristic gradient magnitude spikes.
    This function highlights those boundaries in orange.
    """
    import cv2
    rgb = np.asarray(img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Multi-scale gradient magnitude.
    edges = np.zeros_like(gray, dtype=np.float32)
    for ksize in (3, 5, 9):
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        mag = np.sqrt(sobelx ** 2 + sobely ** 2).astype(np.float32)
        edges += mag / (ksize ** 2)

    # Laplacian of Gaussian for blob-like boundary detection.
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
    log_resp = np.abs(cv2.Laplacian(blurred, cv2.CV_64F)).astype(np.float32)
    combined = 0.6 * edges + 0.4 * log_resp

    # Normalize and threshold.
    combined -= combined.min()
    combined /= (combined.max() + 1e-9)

    # Apply thermal colourmap (orange = boundary, dark = homogeneous).
    heat = plt.cm.hot(combined)[:, :, :3]
    heat_uint8 = (heat * 255).astype(np.uint8)
    overlay = Image.fromarray(heat_uint8).resize(img.size, Image.LANCZOS)

    # Blend with original.
    base = img.convert("RGB")
    return Image.blend(base, overlay, alpha=0.55)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  3D FFT POWER SPECTRUM (Plotly surface)
# ══════════════════════════════════════════════════════════════════════════════
def fft_3d_surface(img: Image.Image, crop: int = 128) -> go.Figure:
    """Interactive 3D Plotly surface of the log FFT power spectrum.

    GAN outputs show characteristic peaks at regular intervals.
    Diffusion models show smooth roll-off. Real images show 1/f slope.
    """
    import cv2
    rgb = np.asarray(img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)

    # Centre-crop for speed.
    h, w = gray.shape
    s = min(h, w, crop)
    y0, x0 = (h - s) // 2, (w - s) // 2
    patch = gray[y0:y0 + s, x0:x0 + s]

    win = np.outer(np.hanning(s), np.hanning(s))
    f = np.fft.fftshift(np.fft.fft2(patch * win))
    power = np.log(np.abs(f) ** 2 + 1e-12)

    # Downsample for rendering speed.
    step = max(1, s // 64)
    Z = power[::step, ::step]
    X = np.linspace(-1, 1, Z.shape[1])
    Y = np.linspace(-1, 1, Z.shape[0])
    XX, YY = np.meshgrid(X, Y)

    fig = go.Figure(go.Surface(
        x=XX, y=YY, z=Z,
        colorscale="Plasma",
        showscale=True,
        colorbar=dict(title="log power", tickfont_color=_FONT_COLOR),
        contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="#38bdf8", project_z=True)),
    ))
    fig.update_layout(
        title=dict(text="📡 3D FFT Power Spectrum — Generator Fingerprint", font_color="#38bdf8"),
        scene=dict(
            xaxis=dict(title="u-frequency", gridcolor=_GRID_COLOR, color=_FONT_COLOR),
            yaxis=dict(title="v-frequency", gridcolor=_GRID_COLOR, color=_FONT_COLOR),
            zaxis=dict(title="log|F|²", gridcolor=_GRID_COLOR, color=_FONT_COLOR),
            bgcolor="rgba(4,7,15,0.95)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color=_FONT_COLOR,
        height=420,
        margin=dict(t=60, b=20, l=20, r=20),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 4.  FREQUENCY ANOMALY RADAR (Plotly polar)
# ══════════════════════════════════════════════════════════════════════════════
def frequency_radar(img: Image.Image, n_sectors: int = 16) -> go.Figure:
    """Polar radar of directional spectral energy.

    Real images: roughly isotropic (flat radar).
    GAN images: spiky — energy concentrates at discrete angles from checkerboard grids.
    """
    import cv2
    gray = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY).astype(np.float64)
    h, w = gray.shape
    s = min(h, w, 256)
    y0, x0 = (h - s) // 2, (w - s) // 2
    patch = gray[y0:y0 + s, x0:x0 + s]

    win = np.outer(np.hanning(s), np.hanning(s))
    f = np.fft.fftshift(np.fft.fft2(patch * win))
    power = np.abs(f) ** 2

    yy, xx = np.indices(power.shape)
    cx, cy = s // 2, s // 2
    angles = np.arctan2(yy - cy, xx - cx)
    radius = np.hypot(yy - cy, xx - cx)

    sector_energy = np.zeros(n_sectors)
    sector_angles = np.linspace(-np.pi, np.pi, n_sectors + 1)
    band = (radius > s * 0.1) & (radius < s * 0.48)
    for i in range(n_sectors):
        mask = band & (angles >= sector_angles[i]) & (angles < sector_angles[i + 1])
        sector_energy[i] = power[mask].mean() if mask.any() else 0.0

    # Normalize.
    sector_energy /= (sector_energy.max() + 1e-9)
    labels = [f"{int(np.degrees(a + np.pi / n_sectors))}°" for a in sector_angles[:-1]]

    fig = go.Figure(go.Scatterpolar(
        r=np.append(sector_energy, sector_energy[0]),
        theta=np.append(np.degrees(sector_angles[:-1]), np.degrees(sector_angles[0])),
        fill="toself",
        fillcolor="rgba(56,189,248,0.15)",
        line=dict(color="#38bdf8", width=2),
        name="Spectral energy",
    ))
    fig.update_layout(
        title=dict(text="🌐 Directional Spectral Energy (anisotropy → GAN/Diffusion)", font_color="#38bdf8"),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=_GRID_COLOR, color=_FONT_COLOR),
            angularaxis=dict(gridcolor=_GRID_COLOR, color=_FONT_COLOR),
            bgcolor="rgba(8,14,32,0.8)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color=_FONT_COLOR,
        height=340,
        showlegend=False,
        margin=dict(t=60, b=20, l=40, r=40),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 5.  FACE LANDMARK OVERLAY  (MediaPipe face mesh; OpenCV Haar fallback)
# ══════════════════════════════════════════════════════════════════════════════
def face_landmark_overlay(img: Image.Image) -> Image.Image:
    """Draw face mesh landmarks + bounding box on the image.

    Uses MediaPipe if available, otherwise MTCNN boxes.
    Shows:
      - 468-point face mesh (MediaPipe)
      - Coloured contours: jawline (red), eyes (cyan), brows (yellow),
        nose (green), lips (magenta)
    """
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        mp_styles = mp.solutions.drawing_styles

        rgb = np.asarray(img.convert("RGB"))
        out = rgb.copy()

        with mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=3,
            refine_landmarks=True, min_detection_confidence=0.4,
        ) as mesh:
            results = mesh.process(rgb)

        if not results.multi_face_landmarks:
            return _fallback_landmark(img)

        h, w = rgb.shape[:2]
        pil_out = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_out)

        CONTOUR_COLORS = {
            "FACE_OVAL": (239, 68, 68),
            "LEFT_EYE": (56, 189, 248),
            "RIGHT_EYE": (56, 189, 248),
            "LEFT_EYEBROW": (234, 179, 8),
            "RIGHT_EYEBROW": (234, 179, 8),
            "NOSE": (34, 197, 94),
            "LIPS": (216, 27, 96),
        }
        connections = {
            "FACE_OVAL": mp_face_mesh.FACEMESH_FACE_OVAL,
            "LEFT_EYE": mp_face_mesh.FACEMESH_LEFT_EYE,
            "RIGHT_EYE": mp_face_mesh.FACEMESH_RIGHT_EYE,
            "LEFT_EYEBROW": mp_face_mesh.FACEMESH_LEFT_EYEBROW,
            "RIGHT_EYEBROW": mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
            "NOSE": mp_face_mesh.FACEMESH_NOSE,
            "LIPS": mp_face_mesh.FACEMESH_LIPS,
        }

        for face_lm in results.multi_face_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in face_lm.landmark]
            # Dot every landmark.
            for p in pts:
                draw.ellipse([p[0]-1, p[1]-1, p[0]+1, p[1]+1],
                             fill=(255, 255, 255, 180))
            # Draw contours.
            for name, conn_set in connections.items():
                color = CONTOUR_COLORS[name]
                for start_idx, end_idx in conn_set:
                    p1 = pts[start_idx]
                    p2 = pts[end_idx]
                    draw.line([p1, p2], fill=color, width=1)

        return pil_out

    except ImportError:
        return _fallback_landmark(img)
    except Exception as e:
        log.debug("MediaPipe landmark failed: %s", e)
        return _fallback_landmark(img)


def _fallback_landmark(img: Image.Image) -> Image.Image:
    """Draw face boxes using MTCNN or Haar fallback."""
    try:
        from detector.models import get_face_detector
        detector = get_face_detector()
        faces = detector.detect(img)
    except Exception:
        return img

    out = img.copy()
    draw = ImageDraw.Draw(out)
    for f in faces:
        draw.rectangle(list(f.box), outline=(56, 189, 248), width=3)
        if f.landmarks is not None:
            for lm in f.landmarks:
                x, y = int(lm[0]), int(lm[1])
                draw.ellipse([x-4, y-4, x+4, y+4], fill=(239, 68, 68))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 6.  HALF-FACE ASYMMETRY MAP
# ══════════════════════════════════════════════════════════════════════════════
def half_face_asymmetry(img: Image.Image,
                        face_box: Optional[Tuple] = None) -> Image.Image:
    """Compare left vs right half of the face — face-swap introduces asymmetry.

    Returns a side-by-side image:
      Left panel: original face | Right panel: |left - mirror(right)| diff
    Bright regions in the diff = inconsistency between halves.
    """
    import cv2

    if face_box is not None:
        x1, y1, x2, y2 = [int(v) for v in face_box]
        face = img.crop((x1, y1, x2, y2))
    else:
        face = img

    face_rgb = face.convert("RGB").resize((256, 256), Image.LANCZOS)
    arr = np.asarray(face_rgb, dtype=np.float32)
    mid = arr.shape[1] // 2
    left = arr[:, :mid]
    right = arr[:, mid:][:, ::-1, :]  # mirror right half

    # Resize to same width.
    w = min(left.shape[1], right.shape[1])
    left, right = left[:, :w], right[:, :w]
    diff = np.abs(left - right)

    # Amplify.
    diff_norm = np.clip(diff * 3, 0, 255).astype(np.uint8)
    diff_heat = plt.cm.hot(diff_norm.mean(axis=2) / 255)[:, :, :3]
    diff_heat = (diff_heat * 255).astype(np.uint8)

    # Side-by-side.
    left_img = Image.fromarray(left.astype(np.uint8))
    diff_img = Image.fromarray(diff_heat)
    right_img = Image.fromarray(right.astype(np.uint8))

    combined = Image.new("RGB", (left_img.width + diff_img.width + right_img.width + 8, 256))
    combined.paste(left_img, (0, 0))
    combined.paste(diff_img, (left_img.width + 4, 0))
    combined.paste(right_img, (left_img.width + diff_img.width + 8, 0))

    # Label.
    draw = ImageDraw.Draw(combined)
    draw.rectangle([0, 0, left_img.width - 1, 14], fill=(0, 0, 80))
    draw.text((4, 2), "LEFT HALF", fill=(200, 200, 255))
    draw.rectangle([left_img.width + 4, 0, left_img.width + diff_img.width - 1, 14], fill=(80, 0, 0))
    draw.text((left_img.width + 8, 2), "ASYM DIFF", fill=(255, 180, 180))
    draw.rectangle([left_img.width + diff_img.width + 8, 0, combined.width - 1, 14], fill=(0, 0, 80))
    draw.text((left_img.width + diff_img.width + 12, 2), "RIGHT(mirror)", fill=(200, 200, 255))
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# 7.  COMPRESSION ARTIFACT MAP
# ══════════════════════════════════════════════════════════════════════════════
def compression_artifact_map(img: Image.Image, block: int = 8) -> Image.Image:
    """Block-DCT coefficient variance map — detects JPEG re-compression traces.

    AI-generated images often show uniform block structure because generators
    don't introduce real camera sensor noise that modulates block variance.
    Re-compressed regions show boundary artifacts at multiples of 8px.
    """
    import cv2
    from scipy.fft import dctn

    gray = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY).astype(np.float64)
    H, W = gray.shape
    grid_h = H // block
    grid_w = W // block
    var_map = np.zeros((grid_h, grid_w), dtype=np.float64)

    for i in range(grid_h):
        for j in range(grid_w):
            patch = gray[i * block:(i + 1) * block, j * block:(j + 1) * block]
            coeffs = dctn(patch, norm="ortho")
            # High-frequency AC coefficient variance.
            ac = coeffs.copy()
            ac[0, 0] = 0  # zero DC
            var_map[i, j] = float(ac.var())

    # Upsample to image size.
    var_up = np.kron(var_map, np.ones((block, block), dtype=np.float64))
    var_up = var_up[:H, :W]
    var_norm = (var_up - var_up.min()) / (var_up.max() - var_up.min() + 1e-9)

    heat = plt.cm.viridis(var_norm)[:, :, :3]
    heat_img = Image.fromarray((heat * 255).astype(np.uint8))
    base = img.convert("RGB").resize(heat_img.size, Image.LANCZOS)
    return Image.blend(base, heat_img, alpha=0.6)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  MULTI-PANEL FORENSIC GRID
# ══════════════════════════════════════════════════════════════════════════════
def forensic_grid(img: Image.Image,
                  face_box: Optional[Tuple] = None,
                  size: int = 256) -> Image.Image:
    """6-panel forensic grid image (no Streamlit dependencies).

    Panels:
      [0] Original RGB
      [1] Face landmark overlay
      [2] ELA (Error Level Analysis)
      [3] Blending boundary map
      [4] Noise SRM residual (amplified)
      [5] Half-face asymmetry diff
    """
    import cv2

    def _resize(i): return i.resize((size, size), Image.LANCZOS)

    panels = []

    # 0 — Original.
    panels.append(_resize(img.convert("RGB")))

    # 1 — Landmark overlay.
    try:
        panels.append(_resize(face_landmark_overlay(img)))
    except Exception:
        panels.append(_resize(img))

    # 2 — ELA.
    try:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "JPEG", quality=88)
        buf.seek(0)
        comp = Image.open(buf)
        ela = ImageChops.difference(img.convert("RGB"), comp)
        extrema = ela.getextrema()
        mx = max(e[1] for e in extrema) or 1
        ela = ImageEnhance.Brightness(ela).enhance(255 / mx * 1.5)
        panels.append(_resize(ela))
    except Exception:
        panels.append(_resize(img))

    # 3 — Blending boundary.
    try:
        panels.append(_resize(blending_boundary_map(img)))
    except Exception:
        panels.append(_resize(img))

    # 4 — SRM noise residual.
    try:
        gray = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
        res = np.abs(_srm_noise(gray))
        res_norm = np.clip(res / (res.max() + 1e-9) * 255, 0, 255).astype(np.uint8)
        heat = plt.cm.inferno(res_norm / 255)[:, :, :3]
        panels.append(_resize(Image.fromarray((heat * 255).astype(np.uint8))))
    except Exception:
        panels.append(_resize(img))

    # 5 — Half-face asymmetry.
    try:
        panels.append(_resize(half_face_asymmetry(img, face_box)))
    except Exception:
        panels.append(_resize(img))

    # Compose 2×3 grid.
    cols, rows = 3, 2
    labels = ["Original", "Face Mesh", "ELA", "Blend Boundary", "SRM Noise", "Asymmetry Diff"]
    pw, ph = size, size + 24  # extra height for label
    grid_w = cols * pw + (cols - 1) * 4
    grid_h = rows * ph + (rows - 1) * 4
    grid = Image.new("RGB", (grid_w, grid_h), (4, 7, 15))
    draw = ImageDraw.Draw(grid)

    for idx, (panel, label) in enumerate(zip(panels, labels)):
        col = idx % cols
        row = idx // cols
        x = col * (pw + 4)
        y = row * (ph + 4)
        grid.paste(panel, (x, y + 24))
        draw.rectangle([x, y, x + pw - 1, y + 22], fill=(8, 14, 32))
        draw.text((x + 4, y + 4), label, fill=(56, 189, 248))

    return grid


# ══════════════════════════════════════════════════════════════════════════════
# 9.  VIDEO SUSPICIOUS FRAME CONTACT SHEET
# ══════════════════════════════════════════════════════════════════════════════
def suspicious_frame_contact_sheet(
    frames: List[Tuple[float, Image.Image, float]],
    thumb: int = 160,
) -> Image.Image:
    """Grid of video frames coloured by fake probability.

    Args:
        frames: list of (timestamp_s, PIL_image, p_fake) tuples.
    Returns PIL contact sheet.
    """
    if not frames:
        return Image.new("RGB", (320, 80), (4, 7, 15))

    frames = sorted(frames, key=lambda f: f[2], reverse=True)[:12]
    cols = min(4, len(frames))
    rows = math.ceil(len(frames) / cols)
    cell_h = thumb + 36
    sheet = Image.new("RGB", (cols * (thumb + 4), rows * (cell_h + 4)), (4, 7, 15))
    draw = ImageDraw.Draw(sheet)

    for idx, (ts, frame, pf) in enumerate(frames):
        col = idx % cols
        row = idx // cols
        x = col * (thumb + 4)
        y = row * (cell_h + 4)
        thumb_img = frame.convert("RGB").resize((thumb, thumb), Image.LANCZOS)

        # Border colour by probability.
        r = int(pf * 239 + (1 - pf) * 34)
        g = int(pf * 68 + (1 - pf) * 197)
        b = int(pf * 68 + (1 - pf) * 94)

        # Border.
        bordered = Image.new("RGB", (thumb + 4, thumb + 4), (r, g, b))
        bordered.paste(thumb_img, (2, 2))
        sheet.paste(bordered, (x, y + 30))

        # Label.
        draw.rectangle([x, y, x + thumb + 3, y + 28], fill=(8, 14, 32))
        pct_str = f"t={ts:.1f}s  P={pf:.0%}"
        draw.text((x + 4, y + 6), pct_str, fill=(r, g, b))

    return sheet


# ══════════════════════════════════════════════════════════════════════════════
# 10.  AUDIO FORENSIC PANEL  (waveform + LFCC + mel)
# ══════════════════════════════════════════════════════════════════════════════
def audio_forensic_panel(audio_path: str, sr: int = 16000) -> go.Figure:
    """3-row Plotly figure: waveform + LFCC spectrogram + mel spectrogram."""
    try:
        import torchaudio
        import torch
        wave, orig_sr = torchaudio.load(audio_path)
        if orig_sr != sr:
            wave = torchaudio.functional.resample(wave, orig_sr, sr)
        wave = wave.mean(0)[:sr * 30]  # mono, max 30 s

        t = np.linspace(0, len(wave) / sr, len(wave))

        # Mel.
        mel_tf = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=1024,
                                                       hop_length=256, n_mels=64)
        mel_db = torchaudio.transforms.AmplitudeToDB()(mel_tf(wave.unsqueeze(0)))[0].numpy()

        # LFCC.
        lfcc_tf = torchaudio.transforms.LFCC(sample_rate=sr, n_filter=128, n_lfcc=40)
        lfcc = lfcc_tf(wave.unsqueeze(0))[0].numpy()

        from plotly.subplots import make_subplots
        fig = make_subplots(rows=3, cols=1, subplot_titles=[
            "Waveform", "LFCC Features (Voice Clone Fingerprint)", "Mel Spectrogram"
        ], vertical_spacing=0.1)

        # Row 1: waveform.
        step = max(1, len(wave) // 4000)
        fig.add_trace(go.Scatter(
            x=t[::step], y=wave.numpy()[::step],
            line=dict(color="#38bdf8", width=1),
            name="Waveform",
        ), row=1, col=1)

        # Row 2: LFCC heatmap.
        fig.add_trace(go.Heatmap(
            z=lfcc, colorscale="Plasma",
            showscale=False, name="LFCC",
        ), row=2, col=1)

        # Row 3: Mel heatmap.
        fig.add_trace(go.Heatmap(
            z=mel_db, colorscale="Magma",
            showscale=True,
            colorbar=dict(title="dB", len=0.3, y=0.1, tickfont_color=_FONT_COLOR),
            name="Mel",
        ), row=3, col=1)

        fig.update_layout(
            height=520,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(8,14,32,0.6)",
            font_color=_FONT_COLOR,
            showlegend=False,
            margin=dict(t=60, b=30, l=60, r=20),
        )
        for axis in ["xaxis", "xaxis2", "xaxis3", "yaxis", "yaxis2", "yaxis3"]:
            fig.update_layout(**{axis: dict(gridcolor=_GRID_COLOR)})
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Audio panel unavailable: {e}", x=0.5, y=0.5,
                           showarrow=False, font_color="#ef4444")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=200)
        return fig


# ══════════════════════════════════════════════════════════════════════════════
# 11.  NPR FEATURE VISUALIZER
# ══════════════════════════════════════════════════════════════════════════════
def npr_direction_chart(img: Image.Image) -> go.Figure:
    """Spider chart of NPR directional difference statistics.

    Real images: roughly isotropic (uniform web).
    GAN/Diffusion: anisotropic — energy spikes in upsampling directions.
    """
    try:
        from detector.experts.spectral import npr_features, NPR_DIRS, NPR_STATS
        feats = npr_features(img).reshape(NPR_DIRS, NPR_STATS)
        means = feats[:, 0]
        stds = feats[:, 1]
        dir_names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

        fig = go.Figure()
        # Mean differences.
        fig.add_trace(go.Scatterpolar(
            r=np.abs(means) / (np.abs(means).max() + 1e-9),
            theta=dir_names,
            fill="toself", fillcolor="rgba(56,189,248,0.12)",
            line=dict(color="#38bdf8"), name="|Mean diff|",
        ))
        # Std differences.
        fig.add_trace(go.Scatterpolar(
            r=stds / (stds.max() + 1e-9),
            theta=dir_names,
            fill="toself", fillcolor="rgba(192,132,252,0.12)",
            line=dict(color="#c084fc"), name="Std diff",
        ))
        fig.update_layout(
            title=dict(text="🔢 NPR Directional Pixel Differences (GAN grid → anisotropy)", font_color="#38bdf8"),
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor=_GRID_COLOR, color=_FONT_COLOR),
                angularaxis=dict(gridcolor=_GRID_COLOR, color=_FONT_COLOR),
                bgcolor="rgba(8,14,32,0.8)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color=_FONT_COLOR,
            height=360,
            showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=60, b=20, l=40, r=40),
        )
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"NPR chart unavailable: {e}", x=0.5, y=0.5,
                           showarrow=False, font_color="#ef4444")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=200)
        return fig
