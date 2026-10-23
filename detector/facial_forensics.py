import cv2
import numpy as np
from PIL import Image


def analyze_image_quality_and_noise(image_input) -> dict:
    """
    Computes spatial blurriness (Laplacian variance), skin chrominance variance,
    and High-Pass Spatial Rich Model (SRM) noise residual artifacts.
    """
    if isinstance(image_input, Image.Image):
        img_np = np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        img_np = image_input
    else:
        img_np = np.array(Image.open(image_input).convert("RGB"))

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # 1. Laplacian Blur Variance
    # AI faces often have over-smoothed skin regions
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 2. Skin Chrominance Variance (YCbCr)
    ycbcr = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
    cr_std = float(np.std(ycbcr[:, :, 1]))
    cb_std = float(np.std(ycbcr[:, :, 2]))
    chroma_variance = (cr_std + cb_std) / 2.0

    # 3. High-Pass Spatial Rich Model (SRM) Noise Filter
    srm_kernel = np.array([
        [0, 0, 0, 0, 0],
        [0, -1, 2, -1, 0],
        [0, 2, -4, 2, 0],
        [0, -1, 2, -1, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32) / 4.0

    srm_filtered = cv2.filter2D(gray, -1, srm_kernel)
    srm_noise_energy = float(np.std(srm_filtered))

    # Determine risk indicators
    is_suspicious_blur = laplacian_var < 50.0
    is_suspicious_chroma = chroma_variance < 8.0

    risk_score = 0.0
    if is_suspicious_blur:
        risk_score += 0.25
    if is_suspicious_chroma:
        risk_score += 0.25
    if srm_noise_energy < 3.0:
        risk_score += 0.25

    return {
        "laplacian_blur_score": f"{laplacian_var:.1f}",
        "chroma_variance": f"{chroma_variance:.1f}",
        "srm_noise_energy": f"{srm_noise_energy:.1f}",
        "is_suspicious_blur": is_suspicious_blur,
        "is_suspicious_chroma": is_suspicious_chroma,
        "heuristic_risk_score": float(risk_score)
    }
