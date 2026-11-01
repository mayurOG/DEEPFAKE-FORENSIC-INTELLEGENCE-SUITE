"""Optical flow inconsistency expert — v4.0 (NEW).

Face-swap deepfakes warp the source face onto the target face frame-by-frame.
This warping leaves characteristic inconsistencies in the optical flow field:
  * Abrupt motion discontinuities at the face boundary.
  * Unnaturally smooth or overly regular motion inside the face region.
  * Temporal flickering in the warp residual.

This expert computes dense optical flow between consecutive frames (using
OpenCV's Farneback algorithm — no extra dependency; RAFT is optional for
higher accuracy when torchvision is available) and measures:
  1. Boundary magnitude ratio: flow magnitude at the face edge vs interior.
  2. Flow regularity: how unnaturally smooth the flow is inside the face.
  3. Temporal consistency: frame-to-frame flip of the dominant motion direction.

The evidence is heuristic and calibrated to be conservative (low weight until
fusion weights are learned). It is most reliable on high-frame-rate videos
with a consistent face track.

Integration:
    Called from detector.pipeline in the video Tier-2 analysis block.
    Requires: a video path, the face detector, and at least 8 consecutive frames.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from detector.fusion import Evidence, skipped

log = logging.getLogger("deepfake.optical_flow")


def _farneback_flow(gray1: np.ndarray, gray2: np.ndarray) -> np.ndarray:
    """Dense optical flow via Farneback (OpenCV, always available)."""
    import cv2
    return cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0,
    )  # shape (H, W, 2)


def _raft_flow(frame1: np.ndarray, frame2: np.ndarray, device: str) -> Optional[np.ndarray]:
    """Dense optical flow via RAFT (torchvision) — higher accuracy.

    Returns None if torchvision or CUDA is unavailable so we fall back to
    Farneback gracefully.
    """
    try:
        import torch
        import torchvision.transforms.functional as TF
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights).to(device).eval()
        t = weights.transforms()

        def to_tensor(arr: np.ndarray):
            im = Image.fromarray(arr)
            return TF.to_tensor(im).unsqueeze(0).to(device)

        with torch.inference_mode():
            flows = model(t(to_tensor(frame1), to_tensor(frame2)))
        flow = flows[-1].squeeze(0).permute(1, 2, 0).cpu().numpy()
        return flow
    except Exception as e:
        log.debug("RAFT not available, falling back to Farneback: %s", e)
        return None


def _face_boundary_mask(
    gray: np.ndarray, face_box: Tuple[int, int, int, int],
    boundary_width: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (interior_mask, boundary_mask) for the face box."""
    H, W = gray.shape
    x1, y1, x2, y2 = [int(v) for v in face_box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    interior = np.zeros((H, W), dtype=bool)
    interior[y1 + boundary_width:y2 - boundary_width,
             x1 + boundary_width:x2 - boundary_width] = True
    face = np.zeros((H, W), dtype=bool)
    face[y1:y2, x1:x2] = True
    boundary = face & ~interior
    return interior, boundary


def _flow_magnitude(flow: np.ndarray) -> np.ndarray:
    return np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)


def _flow_regularity(mag: np.ndarray) -> float:
    """How unnaturally smooth the flow is (low variance = suspicious)."""
    if mag.size < 4:
        return 0.0
    return float(np.std(mag) / (np.mean(mag) + 1e-6))


def extract_flow_features(
    video_path: str,
    face_detector,
    max_frames: int = 32,
    max_seconds: float = 10.0,
    use_raft: bool = True,
    device: str = "cpu",
) -> Optional[dict]:
    """Extract optical flow features from a video.

    Returns a dict with:
        boundary_ratio: list of flow-magnitude boundary/interior ratios per pair.
        regularity: list of interior flow regularity scores per pair.
        direction_flips: fraction of frame pairs where dominant direction flips.
        n_pairs: number of valid consecutive-frame pairs analysed.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return None

    # Sample frames evenly (up to max_frames).
    n_target = min(max_frames, int(max_seconds * fps))
    if total > n_target:
        idxs = np.linspace(0, total - 1, n_target, dtype=int)
    else:
        idxs = np.arange(total, dtype=int)

    frames: List[np.ndarray] = []
    boxes: List[Optional[Tuple]] = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, fr = cap.read()
        if not ok or fr is None:
            continue
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        detected = face_detector.detect(pil)
        box = max(detected, key=lambda f: f.area).box if detected else None
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        boxes.append(box)
    cap.release()

    if len(frames) < 2:
        return None

    boundary_ratios, regularities, directions = [], [], []

    for i in range(len(frames) - 1):
        g1, g2 = frames[i], frames[i + 1]
        box = boxes[i] or boxes[i + 1]

        # Try RAFT first, fall back to Farneback.
        flow = None
        if use_raft:
            rgb1 = cv2.cvtColor(g1, cv2.COLOR_GRAY2RGB)
            rgb2 = cv2.cvtColor(g2, cv2.COLOR_GRAY2RGB)
            flow = _raft_flow(rgb1, rgb2, device)
        if flow is None:
            flow = _farneback_flow(g1, g2)

        mag = _flow_magnitude(flow)

        if box is not None:
            try:
                interior, boundary = _face_boundary_mask(g1, box)
                int_mag = mag[interior]
                bnd_mag = mag[boundary]
                if int_mag.size > 0 and bnd_mag.size > 0:
                    ratio = float(np.mean(bnd_mag) / (np.mean(int_mag) + 1e-6))
                    boundary_ratios.append(ratio)
                    regularities.append(_flow_regularity(int_mag))
            except Exception:
                pass

        # Dominant motion direction (angle quantised to 8 bins).
        angles = np.arctan2(flow[..., 1], flow[..., 0])
        dom = float(np.angle(np.exp(1j * angles).mean()))
        directions.append(dom)

    direction_flips = 0.0
    if len(directions) > 1:
        diffs = np.abs(np.diff(directions))
        diffs = np.minimum(diffs, 2 * np.pi - diffs)  # circular diff
        direction_flips = float((diffs > np.pi / 2).mean())

    return {
        "boundary_ratio": boundary_ratios,
        "regularity": regularities,
        "direction_flips": direction_flips,
        "n_pairs": len(frames) - 1,
    }


def optical_flow_evidence(
    video_path: str,
    face_detector,
    quality_factor: float = 1.0,
    use_raft: bool = True,
    device: str = "cpu",
) -> Evidence:
    """Optical flow inconsistency evidence for video deepfake detection."""
    try:
        feats = extract_flow_features(
            video_path, face_detector,
            use_raft=use_raft, device=device,
        )
    except Exception as e:
        log.warning("Optical flow analysis failed: %s", e)
        return skipped("optical_flow", f"extraction failed: {e}", True)

    if feats is None or feats["n_pairs"] < 4:
        return skipped("optical_flow", "too few frames for flow analysis", True)

    # Score components:
    # 1. High boundary ratio → abrupt face-edge discontinuity → suspicious.
    br = feats["boundary_ratio"]
    br_score = 0.0
    if br:
        mean_br = float(np.mean(br))
        # Genuine motion: boundary ≈ interior (ratio ≈ 1.0).
        # Face-swap: motion discontinuity → ratio often > 2.
        br_score = float(np.clip((mean_br - 1.5) * 1.2, -2.0, 2.0))

    # 2. Unnaturally low regularity (very uniform/smooth flow inside face) → suspicious.
    reg = feats["regularity"]
    reg_score = 0.0
    if reg:
        mean_reg = float(np.mean(reg))
        # Real face motion: moderate regularity (0.3–0.8).
        # Warped face: very smooth (low regularity coefficient of variation).
        reg_score = float(np.clip((0.25 - mean_reg) * 3.0, -1.5, 1.5))

    # 3. High direction flip rate → temporal jitter from frame-by-frame warp.
    flip_score = float(np.clip((feats["direction_flips"] - 0.3) * 2.5, -1.5, 1.5))

    # Combine scores (weighted blend).
    logit = 0.5 * br_score + 0.3 * reg_score + 0.2 * flip_score
    logit = float(np.clip(logit, -4.0, 4.0))

    reliability = min(1.0, feats["n_pairs"] / 12.0) * quality_factor
    reliability = float(np.clip(reliability, 0.0, 0.7))  # cap; heuristic

    return Evidence(
        name="optical_flow",
        logit=logit,
        reliability=reliability,
        heuristic=True,
        detail={
            "mean_boundary_ratio": round(float(np.mean(br)) if br else 0.0, 3),
            "mean_regularity": round(float(np.mean(reg)) if reg else 0.0, 3),
            "direction_flip_rate": round(feats["direction_flips"], 3),
            "n_pairs": feats["n_pairs"],
            "method": "raft" if use_raft else "farneback",
        },
    )
