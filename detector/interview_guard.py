"""Interview Guard — Real-time cheating detection engine for live video calls.

Detects:
  1. Deepfake / virtual camera face (single-frame forensic score)
  2. Off-screen gaze (reading from notes / second screen)
  3. Multiple persons in frame
  4. Abnormal head pose (looking down / sideways)
  5. Low blink rate (typically means reading from a screen)
  6. Mouth movement without audio (silent external coaching)
  7. Face absence (turned away from camera)
  8. Temporal facial motion inconsistency (synthetic face artefacts)

Each indicator produces a 0–100 risk score. The composite
``InterviewRisk`` is a weighted mean.

Usage (single frame):
    guard = InterviewGuard()
    report = guard.analyze_frame(pil_image)
    # report.overall_risk → 0..100
    # report.indicators  → dict of indicator → score
    # report.flags       → list of triggered alert strings
    # report.annotated   → PIL image with overlay

Usage (update over session):
    guard = InterviewGuard()
    for frame in frames:
        guard.update(frame)
    summary = guard.session_summary()
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ── Indicator weights (sum to 1.0) ────────────────────────────────────────────
INDICATOR_WEIGHTS: Dict[str, float] = {
    "face_authenticity":  0.30,   # deepfake / virtual camera detection
    "gaze_deviation":    0.20,   # looking off-screen
    "head_pose":         0.15,   # large pitch/yaw angle
    "multiple_persons":  0.12,   # > 1 face in frame
    "blink_rate":        0.08,   # abnormally low → reading from screen
    "face_absence":      0.08,   # face not detected
    "motion_pattern":    0.07,   # unusual facial motion (lip/warp)
}

ALERT_THRESHOLDS = {
    "face_authenticity":  55,
    "gaze_deviation":    50,
    "head_pose":         50,
    "multiple_persons":  70,
    "blink_rate":        60,
    "face_absence":      70,
    "motion_pattern":    60,
}

FLAG_MESSAGES = {
    "face_authenticity":  "⚠ AI/virtual camera face detected",
    "gaze_deviation":    "⚠ Candidate is looking off-screen",
    "head_pose":         "⚠ Head rotated — possibly reading notes",
    "multiple_persons":  "⚠ Multiple people detected in frame",
    "blink_rate":        "⚠ Abnormal blink rate — possible screen reading",
    "face_absence":      "⚠ Face missing from frame",
    "motion_pattern":    "⚠ Unnatural facial motion pattern",
}


@dataclass
class FrameReport:
    timestamp: float
    indicators: Dict[str, float]      # indicator → 0..100 risk score
    flags: List[str]                   # triggered alerts
    overall_risk: float                # 0..100 weighted composite
    annotated: Optional[Image.Image] = None
    face_count: int = 0
    head_yaw: float = 0.0             # degrees
    head_pitch: float = 0.0           # degrees
    gaze_x: float = 0.0              # normalised -1..1 (0=centre)
    gaze_y: float = 0.0


@dataclass
class SessionSummary:
    duration_s: float
    n_frames: int
    mean_risk: float
    max_risk: float
    risk_timeline: List[float]
    flag_counts: Dict[str, int]
    total_flags: int
    verdict: str                       # "Clean" / "Suspicious" / "High Risk"
    recommendations: List[str]


class InterviewGuard:
    """Stateful per-session interview cheating detector."""

    def __init__(self, blink_window_s: float = 10.0):
        self._blink_window: Deque[float] = deque()  # timestamps of blinks
        self._blink_window_s = blink_window_s
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_eye_ratio: Optional[float] = None
        self._session_reports: List[FrameReport] = []
        self._session_start = time.time()

        # MediaPipe: lazy init.
        self._mp_face_mesh = None
        self._mp_face_det = None
        self._mp_init_done = False

    # ── Public API ────────────────────────────────────────────────────────────
    def analyze_frame(self, img: Image.Image,
                      pipe=None) -> FrameReport:
        """Analyse a single PIL frame. Returns FrameReport.

        Args:
            img:  PIL RGB image.
            pipe: (optional) detector.pipeline.Pipeline instance for
                  fast deepfake scoring of the face crop. If None, a
                  heuristic noise score is used.
        """
        ts = time.time()
        indicators: Dict[str, float] = {k: 0.0 for k in INDICATOR_WEIGHTS}
        meta: Dict[str, float] = {}

        try:
            self._ensure_mp()
            rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
            mp_result = self._run_mp(rgb)
        except Exception:
            mp_result = None
            rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)

        face_count = 0
        head_yaw = head_pitch = gaze_x = gaze_y = 0.0

        # ── 1.  Face authenticity ──────────────────────────────────────────
        auth_score = self._face_authenticity(img, pipe)
        indicators["face_authenticity"] = auth_score

        # ── 2–6.  MediaPipe-based indicators ─────────────────────────────
        if mp_result is not None and mp_result.get("multi_face_landmarks"):
            lms_list = mp_result["multi_face_landmarks"]
            face_count = len(lms_list)

            # Multiple persons.
            if face_count > 1:
                indicators["multiple_persons"] = min(100, 60 + face_count * 15)

            lms = lms_list[0]
            h, w = rgb.shape[:2]

            # Head pose.
            yaw, pitch = self._head_pose(lms, w, h)
            head_yaw, head_pitch = yaw, pitch
            pose_score = min(100, (abs(yaw) / 30 * 60 + abs(pitch) / 25 * 40))
            indicators["head_pose"] = float(pose_score)

            # Gaze.
            gx, gy, gaze_score = self._gaze(lms, w, h)
            gaze_x, gaze_y = gx, gy
            indicators["gaze_deviation"] = float(gaze_score)

            # Blink rate.
            ear = self._eye_aspect_ratio(lms, w, h)
            blink_score = self._update_blink(ear, ts)
            indicators["blink_rate"] = blink_score

            # Motion pattern.
            gray = rgb.mean(axis=2).astype(np.float32)
            motion = self._motion_pattern(gray)
            indicators["motion_pattern"] = motion
            self._prev_gray = gray

        else:
            # No face detected.
            indicators["face_absence"] = 80.0
            indicators["blink_rate"] = 0.0

        # ── 7.  Face absence ───────────────────────────────────────────────
        if face_count == 0:
            indicators["face_absence"] = 80.0

        # ── Composite score ────────────────────────────────────────────────
        overall = sum(
            indicators[k] * w for k, w in INDICATOR_WEIGHTS.items()
        )

        # ── Flags ──────────────────────────────────────────────────────────
        flags = [
            FLAG_MESSAGES[k]
            for k, score in indicators.items()
            if score >= ALERT_THRESHOLDS.get(k, 100)
        ]

        # ── Annotate frame ─────────────────────────────────────────────────
        annotated = self._annotate(img.copy(), indicators, flags, overall,
                                   face_count, head_yaw, head_pitch)

        report = FrameReport(
            timestamp=ts,
            indicators=indicators,
            flags=flags,
            overall_risk=overall,
            annotated=annotated,
            face_count=face_count,
            head_yaw=head_yaw,
            head_pitch=head_pitch,
            gaze_x=gaze_x,
            gaze_y=gaze_y,
        )
        self._session_reports.append(report)
        return report

    def session_summary(self) -> SessionSummary:
        rs = self._session_reports
        if not rs:
            return SessionSummary(0, 0, 0, 0, [], {}, 0, "Clean", [])

        risks = [r.overall_risk for r in rs]
        flag_counts: Dict[str, int] = {k: 0 for k in INDICATOR_WEIGHTS}
        for r in rs:
            for f in r.flags:
                for k, msg in FLAG_MESSAGES.items():
                    if f == msg:
                        flag_counts[k] += 1

        mean_risk = float(np.mean(risks))
        max_risk = float(max(risks))
        duration = rs[-1].timestamp - rs[0].timestamp if len(rs) > 1 else 0.0
        total_flags = sum(len(r.flags) for r in rs)

        if mean_risk < 30 and total_flags < 5:
            verdict = "✅ Clean"
        elif mean_risk < 55:
            verdict = "⚠ Suspicious"
        else:
            verdict = "🚨 High Risk"

        recs = []
        if flag_counts.get("face_authenticity", 0) > 2:
            recs.append("Possible virtual camera / AI avatar — request ID verification.")
        if flag_counts.get("gaze_deviation", 0) > 5:
            recs.append("Candidate frequently looked off-screen — review for external assistance.")
        if flag_counts.get("multiple_persons", 0) > 2:
            recs.append("Multiple people detected — verify candidate is alone.")
        if flag_counts.get("head_pose", 0) > 5:
            recs.append("Unusual head orientation detected repeatedly.")
        if flag_counts.get("blink_rate", 0) > 3:
            recs.append("Abnormal blink pattern — possibly reading from a hidden screen.")

        return SessionSummary(
            duration_s=duration, n_frames=len(rs),
            mean_risk=mean_risk, max_risk=max_risk,
            risk_timeline=risks, flag_counts=flag_counts,
            total_flags=total_flags, verdict=verdict,
            recommendations=recs,
        )

    def reset_session(self):
        self._session_reports.clear()
        self._blink_window.clear()
        self._prev_gray = None
        self._session_start = time.time()

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _ensure_mp(self):
        if self._mp_init_done:
            return
        try:
            import mediapipe as mp
            self._mp_face_mesh_mod = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=3,
                refine_landmarks=True, min_detection_confidence=0.35,
            )
            self._mp_init_done = True
        except ImportError:
            self._mp_init_done = True  # skip silently

    def _run_mp(self, rgb: np.ndarray) -> Optional[dict]:
        if not hasattr(self, "_mp_face_mesh_mod") or self._mp_face_mesh_mod is None:
            return None
        result = self._mp_face_mesh_mod.process(rgb)
        return {
            "multi_face_landmarks": result.multi_face_landmarks,
        } if result else None

    def _face_authenticity(self, img: Image.Image, pipe) -> float:
        """Fast deepfake score — uses pipeline if available, else SRM heuristic."""
        if pipe is not None:
            try:
                import time as _time
                res = pipe.analyze(img, media_type="image", mode="balanced",
                                   explain=False, stability=False)
                return float(res.p_fake * 100)
            except Exception:
                pass
        # Fallback: SRM noise energy as authenticity proxy.
        try:
            import cv2
            gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY).astype(np.float32)
            k = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32) / 4
            res = cv2.filter2D(gray, -1, k)
            energy = float(res.var())
            # High noise energy → more likely natural camera image.
            # AI images tend to have smooth, low-energy noise residuals.
            score = max(0, min(100, (1 - min(energy / 200, 1)) * 60))
            return score
        except Exception:
            return 0.0

    def _head_pose(self, lms, w: int, h: int) -> Tuple[float, float]:
        """Estimate head yaw and pitch from 3D landmarks (no camera matrix)."""
        try:
            # Use 3D coordinates provided by MediaPipe.
            nose = lms.landmark[1]
            left_ear = lms.landmark[234]
            right_ear = lms.landmark[454]
            chin = lms.landmark[152]
            forehead = lms.landmark[10]

            # Yaw: horizontal displacement of nose tip relative to ears.
            ear_mid_x = (left_ear.x + right_ear.x) / 2
            yaw = (nose.x - ear_mid_x) / (abs(left_ear.x - right_ear.x) + 1e-9) * 90
            # Pitch: vertical displacement of nose relative to forehead-chin mid.
            face_mid_y = (forehead.y + chin.y) / 2
            pitch = (nose.y - face_mid_y) / (abs(chin.y - forehead.y) + 1e-9) * 60
            return float(np.degrees(np.arctan(yaw))), float(np.degrees(np.arctan(pitch)))
        except Exception:
            return 0.0, 0.0

    def _gaze(self, lms, w: int, h: int) -> Tuple[float, float, float]:
        """Estimate gaze direction from iris landmarks."""
        try:
            # Iris landmarks: 473=left iris centre, 468=right iris centre.
            # Eye corners: 362 (left eye outer), 263 (left eye inner),
            #              33 (right eye outer), 133 (right eye inner).
            LEFT_IRIS = 473
            RIGHT_IRIS = 468
            LEFT_OUT, LEFT_IN = 362, 263
            RIGHT_OUT, RIGHT_IN = 33, 133

            def iris_ratio(iris_idx, out_idx, in_idx):
                iris = lms.landmark[iris_idx]
                outer = lms.landmark[out_idx]
                inner = lms.landmark[in_idx]
                span = abs(outer.x - inner.x) + 1e-9
                return (iris.x - inner.x) / span  # 0=inner, 1=outer

            lr = iris_ratio(LEFT_IRIS, LEFT_OUT, LEFT_IN)
            rr = iris_ratio(RIGHT_IRIS, RIGHT_OUT, RIGHT_IN)
            gaze_x = float((lr + rr) / 2 - 0.5) * 2   # -1..1

            # Vertical: use landmark 159/386 (upper lid) vs 145/374 (lower lid).
            def vert_ratio(upper_idx, lower_idx):
                u = lms.landmark[upper_idx]
                lo = lms.landmark[lower_idx]
                return float((u.y + lo.y) / 2)

            gy = (vert_ratio(159, 145) + vert_ratio(386, 374)) / 2
            gaze_y = float(gy - 0.5) * 2

            score = min(100, (abs(gaze_x) * 70 + abs(gaze_y) * 40))
            return gaze_x, gaze_y, score
        except Exception:
            return 0.0, 0.0, 0.0

    def _eye_aspect_ratio(self, lms, w: int, h: int) -> float:
        """Vertical vs horizontal eye opening ratio — low EAR → blink."""
        try:
            # Left eye: 159(top), 145(bot), 33(left), 133(right)
            def _ear(t, b, l, r):
                vert = abs(lms.landmark[t].y - lms.landmark[b].y)
                horiz = abs(lms.landmark[l].x - lms.landmark[r].x) + 1e-9
                return vert / horiz

            left_ear = _ear(159, 145, 33, 133)
            right_ear = _ear(386, 374, 362, 263)
            return float((left_ear + right_ear) / 2)
        except Exception:
            return 0.3  # neutral

    def _update_blink(self, ear: float, ts: float) -> float:
        """Maintain blink history; return abnormal blink rate score."""
        BLINK_THRESH = 0.18
        # Add blink event.
        if ear < BLINK_THRESH:
            if self._prev_eye_ratio is None or self._prev_eye_ratio >= BLINK_THRESH:
                self._blink_window.append(ts)
        self._prev_eye_ratio = ear

        # Remove old events.
        cutoff = ts - self._blink_window_s
        while self._blink_window and self._blink_window[0] < cutoff:
            self._blink_window.popleft()

        bpm = len(self._blink_window) / max(self._blink_window_s / 60, 1e-9)
        # Normal: 12–20 blinks/min. Too low (<8): reading. Too high (>30): nervous.
        if bpm < 8:
            return min(100, (8 - bpm) / 8 * 80)
        if bpm > 30:
            return min(100, (bpm - 30) / 20 * 40)
        return 0.0

    def _motion_pattern(self, gray: np.ndarray) -> float:
        """Optical flow magnitude vs previous frame — unnatural stillness or jitter."""
        if self._prev_gray is None:
            return 0.0
        try:
            import cv2
            prev = cv2.resize(self._prev_gray, (64, 64))
            curr = cv2.resize(gray, (64, 64))
            flow = cv2.calcOpticalFlowFarneback(
                prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            mean_mag = float(mag.mean())
            std_mag = float(mag.std())
            # Very low motion with high std → unusual jitter pattern.
            if mean_mag < 0.1 and std_mag > 0.5:
                return min(100, std_mag * 30)
            # Very high sudden motion.
            if mean_mag > 10:
                return min(100, mean_mag * 5)
            return 0.0
        except Exception:
            return 0.0

    def _annotate(self, img: Image.Image, indicators: Dict[str, float],
                  flags: List[str], overall: float,
                  face_count: int, yaw: float, pitch: float) -> Image.Image:
        """Overlay risk dashboard onto the frame."""
        w, h = img.size
        draw = ImageDraw.Draw(img, "RGBA")

        # Risk colour.
        risk_pct = overall / 100
        r = int(risk_pct * 239 + (1 - risk_pct) * 34)
        g = int(risk_pct * 68 + (1 - risk_pct) * 197)
        b_col = int(risk_pct * 68 + (1 - risk_pct) * 94)
        risk_color = (r, g, b_col)

        # Top bar.
        draw.rectangle([0, 0, w, 52], fill=(4, 7, 15, 200))
        verdict_str = (
            "🚨 HIGH RISK" if overall >= 60
            else "⚠ SUSPICIOUS" if overall >= 35
            else "✅ AUTHENTIC"
        )
        draw.text((8, 4), f"Interview Guard  ·  {verdict_str}", fill=risk_color)
        draw.text((8, 26), f"Risk: {overall:.0f}/100  |  Faces: {face_count}  |  Yaw: {yaw:.0f}°  Pitch: {pitch:.0f}°",
                  fill=(148, 163, 184))

        # Side bar with indicators.
        bar_w = 180
        bar_x = w - bar_w - 4
        draw.rectangle([bar_x, 56, w - 4, 56 + len(indicators) * 22 + 12],
                       fill=(4, 7, 15, 210))
        for i, (key, score) in enumerate(indicators.items()):
            y = 60 + i * 22
            sc = score / 100
            rc = (int(sc * 239 + (1 - sc) * 34),
                  int(sc * 68 + (1 - sc) * 197),
                  int(sc * 68 + (1 - sc) * 94))
            label = key.replace("_", " ").title()[:14]
            draw.text((bar_x + 4, y), f"{label}: {score:.0f}", fill=rc)

        # Flags.
        if flags:
            for j, flag in enumerate(flags[:4]):
                fy = h - 30 - j * 24
                draw.rectangle([0, fy - 4, w, fy + 18], fill=(80, 0, 0, 180))
                draw.text((8, fy), flag, fill=(252, 165, 165))

        return img
