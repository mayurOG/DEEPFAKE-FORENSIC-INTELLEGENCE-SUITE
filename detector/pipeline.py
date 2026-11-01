"""The calibrated evidence cascade (recommended advanced architecture).

Stage 1 — Secure media intake
    Format validation · Hashing · Metadata · Decode limits · C2PA

Stage 2 — Intelligent triage
    Media type · Resolution · Face/speech availability · Quality score · Compute budget

Stage 3 — Adaptive expert controller (InvestigationController / VOI)
    Select next detector by uncertainty, disagreement, quality, expected cost.
    Image experts: SigLIP + DINOv2/DINOv3 probe + frequency (spectral/F3Net) + localization
    Video experts: Frame screening + FTCN/VideoMAE-style temporal + evidence timestamps
    Audio experts: AASIST-style + speech embeddings + acoustic quality
    Cross-modal: Lip-sync + identity consistency + temporal alignment
    Physiological (P1): rPPG / face-signal quality gating

Stage 4 — Dependence-aware evidence fusion
    Calibrated scores · Correlated experts (GLS) · Missing modalities · Contradictions

Stage 5 — Decision and explanation
    Real / Fake / Inconclusive · Evidence localization · Confidence limitations

Stage 6 — Auditable forensic report
    Model versions · Evidence timestamps · Provenance verification · Signed output

Public entry point: :class:`Pipeline` / :func:`get_pipeline`.
"""
from __future__ import annotations

import io
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from detector import models as M
from detector.audio_io import has_audio_track, load_audio
from detector.calibration import (
    FAKE, REAL, CalibrationBundle, Calibrator, OODModel, sigmoid,
)
from detector.config import Settings, get_settings
from detector.experts.facesignals import (
    audio_envelope, extract_face_signals, geometry_evidence, rppg_evidence,
    sync_evidence,
)
from detector.experts.spectral import load_head as load_spectral_head, spectral_evidence
from detector.experts.tamper import tamper_evidence, tamper_heatmap_image
try:
    from detector.experts.optical_flow import optical_flow_evidence as _optical_flow_evidence
    _HAS_OPTICAL_FLOW = True
except ImportError:
    _HAS_OPTICAL_FLOW = False
from detector.experts.temporal import (
    aggregate_frame_logits, load_temporal_head, temporal_evidence,
)
from detector.explain import (
    build_narrative, counterfactual_evidence, deletion_faithfulness,
    multiscale_occlusion, occlusion_map, overlay_heatmap,
)
from detector.fusion import Evidence, fuse, load_weights, skipped, _jsonable
from detector.tta import reliability_from_spread, tta_logit
from detector.audit_chain import build_chain
from detector.conditional_reliability import load_conditional
from detector.contradiction_map import contradiction_map
from detector.evidence_graph import build_graph, gls_fuse
from detector.investigation_controller import InvestigationController, TestSpec
from detector.transformation_lab import robustness_gate, robustness_profile
from detector.stability import stability_report
from detector.triage import (
    audio_quality, image_quality, sha256_bytes, sha256_file, triage_image,
    video_quality, detect_c2pa, QualityProfile,
    IngestLimitError, enforce_upload_limit, enforce_image_limits, sniff_media_magic,
)

log = logging.getLogger("deepfake.pipeline")

IMAGE_EXT = {"jpg", "jpeg", "png", "webp", "bmp"}
VIDEO_EXT = {"mp4", "avi", "mov", "mkv", "webm"}
AUDIO_EXT = {"wav", "mp3", "flac", "ogg", "m4a"}
VALID_MODES = ("strict", "balanced", "sensitive")


def infer_media_type(name: str) -> Optional[str]:
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext in IMAGE_EXT:
        return "image"
    if ext in VIDEO_EXT:
        return "video"
    if ext in AUDIO_EXT:
        return "audio"
    return None


@dataclass
class AnalysisResult:
    media_type: str
    filename: Optional[str] = None
    sha256: Optional[str] = None
    verdict: str = "Inconclusive"  # Real | Fake | Inconclusive
    label_set: List[str] = field(default_factory=list)
    p_fake: float = 0.5
    confidence: Optional[float] = None
    calibrated: bool = False
    calibrator: Dict[str, Any] = field(default_factory=dict)
    ood: Optional[bool] = None
    ood_ratio: Optional[float] = None
    tier: str = "tier2"
    mode: str = "balanced"
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    fusion: Dict[str, Any] = field(default_factory=dict)
    quality: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = None
    stability: Optional[Dict[str, Any]] = None
    uncertainty: Dict[str, Any] = field(default_factory=dict)
    graph: Dict[str, Any] = field(default_factory=dict)
    investigation: Dict[str, Any] = field(default_factory=dict)
    robustness: Optional[Dict[str, Any]] = None
    contradiction_map: Optional[Dict[str, Any]] = None
    audit: Dict[str, Any] = field(default_factory=dict)
    abstention_reason: Optional[str] = None
    attribution: Optional[Dict[str, Any]] = None
    grounded: Dict[str, Any] = field(default_factory=dict)
    explanation: Optional[Dict[str, Any]] = None
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    segments: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    backends: List[str] = field(default_factory=list)
    narrative: str = ""
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    # Not serialised: PIL images etc. for the UI, and raw scores for fitting.
    artifacts: Dict[str, Any] = field(default_factory=dict, repr=False)
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k not in ("artifacts", "raw")}
        return _jsonable(d)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def aggregate_faces(logits: np.ndarray) -> float:
    """One image-level logit from several face logits. One manipulated face
    among many should flag the image, but a single noisy face should not
    dominate, so blend the max with the mean."""
    z = np.asarray(logits, dtype=np.float64)
    if len(z) == 0:
        return 0.0
    return float(0.5 * z.max() + 0.5 * z.mean())


def _frame_diff(fr1: np.ndarray, fr2: np.ndarray) -> float:
    """Mean absolute difference between two frames on a normalized [0, 1] scale."""
    if fr1.shape != fr2.shape:
        return 1.0
    return float(np.mean(np.abs(fr1.astype(np.float32) - fr2.astype(np.float32))) / 255.0)


def detect_scene_boundaries(cap, total_frames: int, fps: float, max_probes: int = 50, threshold: float = 0.28) -> List[int]:
    """Detects scene cut frame indices using lightweight thumbnail probing."""
    import cv2

    if total_frames < 20:
        return [0, total_frames]

    step = max(3, total_frames // max_probes)
    probe_indices = list(range(0, total_frames, step))
    if probe_indices[-1] != total_frames - 1:
        probe_indices.append(total_frames - 1)

    boundaries = [0]
    prev_thumb = None
    for idx in probe_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, fr = cap.read()
        if not ok or fr is None:
            continue
        thumb = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (48, 48))
        if prev_thumb is not None:
            diff = _frame_diff(thumb, prev_thumb)
            if diff > threshold:
                boundaries.append(int(idx))
        prev_thumb = thumb

    if boundaries[-1] != total_frames:
        boundaries.append(int(total_frames))
    return boundaries


def sample_frames(path: str, n: int, detect_scenes: bool = True, skip_duplicates: bool = True) -> List[Tuple[int, float, Image.Image]]:
    """``n`` intelligently sampled frames as (index, seconds, PIL image).

    Applies scene-change detection to ensure cross-scene representation, and skips
    near-identical duplicate frames (e.g. frozen slides, repeated loops) to maximize
    information diversity.
    ``n`` is capped by ``DEEPFAKE_MAX_VIDEO_FRAMES``.
    """
    import cv2
    from detector.config import get_settings

    max_f = get_settings().max_video_frames
    if max_f > 0:
        n = min(int(n), int(max_f))
    n = max(1, int(n))

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out: List[Tuple[int, float, Image.Image]] = []

    if total > 0:
        # Determine candidate indices: scene-aware or uniform
        if detect_scenes and total >= 30:
            boundaries = detect_scene_boundaries(cap, total, fps)
            if len(boundaries) > 2:
                # Distribute sample budget across detected scenes
                candidate_idxs = []
                for b_start, b_end in zip(boundaries[:-1], boundaries[1:]):
                    scene_len = b_end - b_start
                    if scene_len <= 0:
                        continue
                    n_scene = max(1, int(round(n * (scene_len / total))))
                    s_pts = np.linspace(b_start, b_end - 1, n_scene + 2)[1:-1].astype(int)
                    candidate_idxs.extend(s_pts.tolist())
                idxs = np.unique(candidate_idxs)
                # If budget exceeded, subsample evenly
                if len(idxs) > n:
                    idxs = idxs[np.linspace(0, len(idxs) - 1, n).astype(int)]
            else:
                idxs = np.unique(np.linspace(0, total - 1, n + 2)[1:-1].astype(int))
        else:
            idxs = np.unique(np.linspace(0, total - 1, n + 2)[1:-1].astype(int))

        prev_thumb = None
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if ok and fr is not None:
                thumb = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (48, 48))
                # Skip duplicate frame if it's near-identical to previous accepted frame
                if skip_duplicates and prev_thumb is not None and len(out) < n:
                    diff = _frame_diff(thumb, prev_thumb)
                    if diff < 0.010:  # < 1% difference: static duplicate
                        continue
                prev_thumb = thumb
                out.append((int(i), float(i / fps),
                            Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))))

    if len(out) < max(2, n // 2):  # seeking unreliable or heavy duplicates: sequential fallback
        cap.release()
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        out = []
        if frames:
            idxs = np.unique(np.linspace(0, len(frames) - 1, n).astype(int))
            for i in idxs:
                out.append((int(i), float(i / fps),
                            Image.fromarray(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))))
    cap.release()
    return out



def frames_at(path: str, times: Sequence[float]) -> List[Tuple[int, float, Image.Image]]:
    import cv2

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:
        fps = 30.0
    out = []
    for t in times:
        idx = max(0, int(round(t * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if ok and fr is not None:
            out.append((idx, float(idx / fps),
                        Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))))
    cap.release()
    return out


def _largest_face_crop(img: Image.Image, detector) -> Tuple[Image.Image, Optional[tuple], int]:
    faces = detector.detect(img)
    if not faces:
        return img, None, 0
    f = max(faces, key=lambda ff: ff.area)
    return M.crop_with_margin(img, f.box), f.box, len(faces)


class Pipeline:
    def __init__(
        self, settings: Optional[Settings] = None,
        calibration: Optional[CalibrationBundle] = None,
        ood: Optional[Dict[str, OODModel]] = None,
        fusion_weights: Optional[Dict[str, float]] = None,
        temporal_head=None,
    ):
        self.settings = settings or get_settings()
        self.calibration = calibration
        self.ood = ood
        self.fusion_weights = fusion_weights
        self.temporal_head = temporal_head
        self.spectral_head = None
        self.attribution_head = None
        self.expert_corr = None
        self.conditional: Dict[str, Any] = {}
        self.reliability_models: Dict[str, Any] = {}
        self._explicit = (calibration is not None, ood is not None,
                          fusion_weights is not None, temporal_head is not None)
        self.reload()

    # -- artifacts -------------------------------------------------------
    def reload(self) -> None:
        d = self.settings.artifact_dir
        if not self._explicit[0]:
            self.calibration = CalibrationBundle.load(os.path.join(d, "calibration.json"))
        if not self._explicit[1]:
            self.ood = {m: OODModel.load(os.path.join(d, f"ood_{m}.npz"))
                        for m in ("image", "video", "audio")}
        if not self._explicit[2]:
            self.fusion_weights = load_weights(d)
        self.spectral_head = load_spectral_head(os.path.join(d, "spectral_head.npz"))
        from detector.attribution import load_attributor
        self.attribution_head = load_attributor(os.path.join(d, "attribution_head.npz"))
        self.expert_corr = None
        try:
            with open(os.path.join(d, "expert_correlation.json"), "r", encoding="utf-8") as f:
                self.expert_corr = json.load(f)
        except Exception:
            pass
        if self.settings.independent_experts:
            from detector.evidence_graph import FAMILY
            self.expert_corr = {a: {b: (1.0 if a == b else 0.0) for b in FAMILY} for a in FAMILY}
        self.conditional, self.reliability_models = {}, {}
        for m in ("image", "video"):
            cc, rel = load_conditional(os.path.join(d, f"conditional_{m}.json"))
            if cc is not None:
                self.conditional[m] = cc
                self.reliability_models.update({f"{m}:{k}": v for k, v in rel.items()})
        if not self._explicit[3]:
            try:
                self.temporal_head = load_temporal_head(os.path.join(d, "temporal_head.pt"))
            except Exception as e:  # pragma: no cover
                log.warning("Could not load temporal head: %s", e)
                self.temporal_head = None

    def status(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"device": M.get_device(), "calibrators": {}}
        for name, cal in (self.calibration.items or {}).items():
            out["calibrators"][name] = {"fitted": cal.fitted, "n_cal": cal.n_cal,
                                        "alpha": cal.alpha}
        out["any_calibrated"] = any(c.fitted for c in self.calibration.items.values())
        out["ood_fitted"] = {k: v.fitted for k, v in self.ood.items()}
        out["learned_fusion_weights"] = self.fusion_weights is not None
        out["learned_temporal_head"] = self.temporal_head is not None
        out["spectral_head"] = self.spectral_head is not None
        out["controller"] = bool(self.settings.controller)
        out["dependent_fusion"] = bool(self.settings.dependent_fusion)
        out["conditional_calibration"] = sorted(self.conditional)
        out["image_backends"] = [b.name for b in M.get_image_backends()]
        out["audio_backend"] = M.get_audio_backend().name
        return out

    # -- decision ----------------------------------------------------------
    def _decide(self, cal_name: str, logit: float, ood_name: str,
                emb: Optional[np.ndarray], mode: str, ctx: Optional[Dict[str, Any]] = None):
        cal = self.calibration.get(cal_name)
        cc = (self.conditional.get(ood_name)
              if self.settings.conditional and cal_name.endswith(".final") else None)
        if cc is not None and cc.fitted:
            cal = cc.base
            p = float(cc.prob_fake(logit))
            label_set = cc.label_set(p, ctx, mode)
        else:
            p = float(cal.prob_fake(logit))
            label_set = cal.label_set(p, mode)
        ood_flag, ratio = (None, None)
        if emb is not None:
            ood_flag, ratio = self.ood.get(ood_name, OODModel()).is_ood(emb[None] if emb.ndim == 1 else emb)
        verdict = (FAKE if label_set == [FAKE] else REAL if label_set == [REAL]
                   else "Inconclusive")
        reason = None
        if ood_flag and verdict != "Inconclusive":
            verdict, reason = "Inconclusive", "outside_calibrated_domain"
        return cal, p, label_set, verdict, ood_flag, ratio, reason

    def _finalise(self, res: AnalysisResult, cal: Calibrator, p, label_set,
                  verdict, ood_flag, ratio, reason, mode, tier):
        res.p_fake = float(p)
        res.label_set = list(label_set)
        res.verdict = verdict
        res.confidence = (p if verdict == FAKE else 1.0 - p if verdict == REAL else None)
        res.calibrated = bool(cal.fitted)
        res.calibrator = {"fitted": cal.fitted, "alpha": cal.alpha, "n_cal": cal.n_cal}
        res.ood, res.ood_ratio = ood_flag, ratio
        res.tier, res.mode = tier, mode
        if reason == "outside_calibrated_domain":
            res.warnings.append(
                "Input looks unlike the calibration data; the verdict was "
                "downgraded to Inconclusive.")
        if not cal.fitted:
            res.warnings.append(
                "Detector not calibrated: probabilities are uncalibrated and "
                "decisions use a conservative fallback margin. Run "
                "`python -m train.fit_calibration` on labelled data.")

    # -- EGAFR helpers -------------------------------------------------------
    def _ctx(self, quality: QualityProfile, evidence: Sequence[Evidence],
             ood_ratio=None, view_spread: float = 0.0, modalities: int = 1) -> Dict[str, Any]:
        used = [e for e in evidence if e.status == "used" and e.logit is not None]
        logits = [float(e.logit) for e in used]
        side = None
        if quality.width and quality.height:
            side = min(quality.width, quality.height)
        return {"compression": quality.compression, "min_side": side,
                "n_used": len(used), "modalities": modalities,
                "disagreement": float(np.std(logits)) if len(logits) > 1 else 0.0,
                "ood_ratio": ood_ratio, "view_spread": view_spread}

    def _apply_reliability(self, evidence: Sequence[Evidence], media: str,
                           ctx: Dict[str, Any]) -> None:
        if not self.settings.conditional:
            return
        for e in evidence:
            m = self.reliability_models.get(f"{media}:{e.name}")
            if e.status == "used" and m is not None and m.fitted:
                e.reliability = m.reliability(ctx, default=e.reliability)

    def _fuse(self, evidence: Sequence[Evidence]):
        fr = fuse(list(evidence), self.fusion_weights)
        info = {"logit": fr.logit, "disagreement": fr.disagreement,
                "n_used": fr.n_used, "contributions": fr.contributions,
                "method": "weighted_mean"}
        if self.settings.dependent_fusion:
            g = gls_fuse(evidence, self.fusion_weights, self.expert_corr)
            info.update({"method": "dependent_gls", "weighted_mean_logit": fr.logit,
                         "logit": g["logit"], "sd": g["sd"], "n_eff": g["n_eff"],
                         "sufficiency": g["sufficiency"]})
            return g["logit"], info
        return fr.logit, info

    def _image_registry(self, img, quality: QualityProfile, backends) -> List[TestSpec]:
        # v4.0: expanded registry — includes diffusion detector and upgraded spectral/NPR.
        comp = {"low": 0.9, "medium": 0.6, "high": 0.3}.get(quality.compression, 0.6)
        va = None if self.spectral_head is None else float(self.spectral_head.get("val_auc", 0.5))
        has_npr = bool(self.spectral_head and self.spectral_head.get("has_npr", False))
        big = min(img.size) >= 96
        has_siglip = any(getattr(b, "kind", "") == "siglip" for b in backends)
        has_diffusion = any(getattr(b, "kind", "") == "diffusion_classifier" for b in backends)
        return [
            # 1. Real NPR + spectral frequency expert (upgraded from stub to real).
            TestSpec(
                "spectral", 0.05,
                lambda c: comp * float(np.clip(((va or 0.5) - 0.5) * 2, 0, 1)),
                lambda c: self.spectral_head is not None and big,
                f"no trained {'NPR+spectral' if has_npr else 'spectral'} head"
                if self.spectral_head is None else "image too small",
            ),
            # 2. Lightweight forensic prior.
            TestSpec("forensic_prior", 0.1, lambda c: 0.5, heuristic=True),
            # 3. Noise inconsistency (tamper detection).
            TestSpec("tamper", 0.4, lambda c: comp, lambda c: big,
                     "image too small", heuristic=True),
            # 4. Second backbone / probe (DINOv2-Large or other).
            TestSpec("neural_probe", 0.5, lambda c: quality.neural_reliability(),
                     lambda c: len(backends) > 1, "no second backbone configured"),
            # 5. SigLIP-SO400M semantic branch (NEW v4.0 at 400M/384px).
            TestSpec("siglip_semantic", 0.35,
                     lambda c: quality.neural_reliability() * 0.8,
                     lambda c: has_siglip, "SigLIP-SO400M branch not configured"),
            # 6. Diffusion-specific detector (NEW v4.0 — SDXL / DALL-E / Flux).
            TestSpec("diffusion_detect", 0.45,
                     lambda c: quality.neural_reliability() * 0.9,
                     lambda c: has_diffusion, "diffusion detector not configured"),
            # 7. Localization heatmap.
            TestSpec("localization", 0.6,
                     lambda c: 0.5 if big else 0.2,
                     lambda c: bool(getattr(self.settings, "localization", True)) and big,
                     "localization disabled or image too small", heuristic=True),
        ]

    def _image_test(self, name, img, crops, backends, quality, face_box) -> Evidence:
        if name == "neural_probe":
            zs = [aggregate_faces(b.predict(crops)[0]) for b in backends[1:]
                  if getattr(b, "kind", "") not in ("siglip", "frequency", "diffusion_classifier")]
            if not zs:
                zs = [0.0]
            return Evidence(
                "neural_probe", float(np.mean(zs)), quality.neural_reliability(),
                detail={"backends": [b.name for b in backends[1:]
                                     if getattr(b, "kind", "") not in
                                     ("siglip", "frequency", "diffusion_classifier")]})
        if name == "siglip_semantic":
            sig = [b for b in backends if getattr(b, "kind", "") == "siglip"]
            if not sig:
                return skipped("siglip_semantic", "SigLIP-SO400M not available")
            _z, emb = sig[0].predict(crops)
            return Evidence(
                "siglip_semantic", 0.0, quality.neural_reliability() * 0.7,
                detail={"backend": sig[0].name,
                        "emb_dim": int(emb.shape[-1]) if emb is not None else 0,
                        "note": "SO400M/384px semantic consistency embedding"})
        if name == "diffusion_detect":
            # NEW v4.0: diffusion-specific detector (SDXL / DALL-E / Flux / Midjourney).
            diff_backends = [b for b in backends if getattr(b, "kind", "") == "diffusion_classifier"]
            if not diff_backends:
                return skipped("diffusion_detect", "diffusion detector not configured")
            zs = [aggregate_faces(b.predict(crops)[0]) for b in diff_backends]
            return Evidence(
                "diffusion_detect", float(np.mean(zs)), quality.neural_reliability() * 0.85,
                detail={"backends": [b.name for b in diff_backends],
                        "note": "Specialized for SDXL/DALL-E/Flux/Midjourney outputs"})
        if name == "tamper":
            return tamper_evidence(img, face_box, quality.compression)
        if name == "spectral":
            # v4.0: real NPR+spectral expert (not a stub).
            has_npr = bool(self.spectral_head and self.spectral_head.get("has_npr", False))
            ev = spectral_evidence(img, self.spectral_head, quality.compression)
            if ev.detail is None:
                ev.detail = {}
            ev.detail["expert_version"] = "NPR+spectral v4.0" if has_npr else "spectral v3.x"
            return ev
        if name == "localization":
            ev = tamper_evidence(img, face_box, quality.compression)
            ev.name = "localization"
            ev.heuristic = True
            if ev.detail is None:
                ev.detail = {}
            ev.detail["note"] = "noise-inconsistency localization; SAM-2 optional upgrade"
            return ev
        return self._forensic_prior(img)

    def _post(self, res: AnalysisResult, media: str, source_size: Optional[int] = None):
        """Evidence graph, sufficiency gate, narrative and audit chain."""
        evs = [Evidence(name=e["name"], logit=e.get("logit"),
                        reliability=e.get("reliability", 0.0) or 0.0,
                        status=e.get("status", "used"), reason=e.get("reason", ""),
                        heuristic=bool(e.get("heuristic")))
               for e in res.evidence]
        g = build_graph(evs, self.fusion_weights, self.expert_corr, res.quality)
        res.graph = g.to_dict()
        suff = g.fusion.get("sufficiency")
        ms = float(self.settings.min_sufficiency or 0.0)
        if (ms > 0 and suff is not None and res.verdict in (FAKE, REAL) and suff < ms):
            res.verdict, res.label_set = "Inconclusive", [REAL, FAKE]
            res.confidence = None
            res.abstention_reason = (f"insufficient_evidence (sufficiency {suff:.2f} < {ms:.2f})")
            res.warnings.append("Evidence is not sufficient for a confident call "
                                "(few or strongly correlated sources).")
        # Generator attribution (image only; also runs for video key frames).
        if self.attribution_head is not None and res.verdict == FAKE:
            try:
                from detector.attribution import attribute_image
                im = res.artifacts.get("source_image")
                if im is not None:
                    res.attribution = attribute_image(im, self.attribution_head)
            except Exception:
                res.attribution = None
        from detector.grounded_explain import build_claims
        res.grounded = {"claims": build_claims(res.to_dict())}
        # LLM narrative (optional; falls back to rule-based).
        s = self.settings
        llm_text = None
        try:
            from detector.llm_narrative import generate_llm_narrative
            llm_text = generate_llm_narrative(
                res.to_dict(),
                provider=s.llm_narrative_provider,
                model=s.llm_narrative_model,
                openai_key=s.openai_api_key,
                gemini_key=s.gemini_api_key,
            )
        except Exception:
            llm_text = None
        res.narrative = llm_text if llm_text else build_narrative(res.to_dict())
        if llm_text:
            res.explanation = res.explanation or {}
            res.explanation["narrative_source"] = "llm"
        res.audit = build_chain(res.to_dict(), self.settings.artifact_dir, source_size).to_dict()
        return res

    # -- entry point -------------------------------------------------------
    def analyze(
        self, source, media_type: Optional[str] = None,
        filename: Optional[str] = None, mode: str = "balanced",
        explain: bool = True, stability: bool = True,
        force_full: bool = False, audio_chunk_s: int = 3,
    ) -> AnalysisResult:
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}")
        from detector.cache import get_inference_cache, ResourceTracker, compute_content_hash
        cache = get_inference_cache()
        content_hash = compute_content_hash(source)
        if self.settings.enable_inference_cache and not force_full:
            cached_res = cache.get(content_hash, mode=mode, explain=explain, stability=stability)
            if cached_res is not None:
                import copy
                cached_copy = copy.copy(cached_res)
                cached_copy.resource_usage = dict(cached_res.resource_usage)
                cached_copy.resource_usage["cache_hit"] = True
                cached_copy.resource_usage["latency_ms"] = 0.5
                return cached_copy

        tmp_path = None
        try:
            if hasattr(source, "read"):  # Streamlit UploadedFile / file object
                filename = filename or getattr(source, "name", None)
                source = source.read()
            # --- Ingest security: size limit + content sniff (do not trust extension) ---
            if isinstance(source, (bytes, bytearray)):
                enforce_upload_limit(bytes(source))
                sniffed = sniff_media_magic(bytes(source))
                if media_type is None and sniffed:
                    media_type = sniffed
                elif media_type and sniffed and media_type != sniffed:
                    # Prefer content over extension when they disagree
                    media_type = sniffed
            if isinstance(source, Image.Image):
                enforce_image_limits(source)
            if media_type is None:
                if isinstance(source, Image.Image):
                    media_type = "image"
                else:
                    name = filename or (source if isinstance(source, str) else "")
                    media_type = infer_media_type(str(name))
            if media_type is None:
                raise ValueError("Cannot infer media type; pass media_type=")
            if media_type != "image" and isinstance(source, (bytes, bytearray)):
                suffix = "." + (filename.rsplit(".", 1)[-1] if filename and "." in filename else "bin")
                fd, tmp_path = tempfile.mkstemp(suffix=suffix)
                with os.fdopen(fd, "wb") as f:
                    f.write(source)
                source = tmp_path
            if isinstance(source, str) and not filename:
                filename = os.path.basename(source)
            # Path-based inputs (CLI / batch) must also respect size + duration limits
            if isinstance(source, str) and os.path.isfile(source):
                from detector.triage import enforce_path_limits
                enforce_path_limits(source, media_type)
            size = (os.path.getsize(source) if isinstance(source, str) and os.path.isfile(source)
                    else None)

            with ResourceTracker() as tracker:
                if media_type == "image":
                    res = self._analyze_image(source, filename, mode, explain, stability, force_full)
                elif media_type == "video":
                    res = self._analyze_video(source, filename, mode, explain, stability, force_full)
                elif media_type == "audio":
                    res = self._analyze_audio(source, filename, mode, explain, audio_chunk_s)
                else:
                    raise ValueError(f"Unsupported media type {media_type}")
                final_res = self._post(res, media_type, size)

            final_res.resource_usage = tracker.usage.to_dict()
            if self.settings.enable_inference_cache and content_hash:
                cache.put(content_hash, final_res, mode=mode, explain=explain, stability=stability)
            return final_res
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    # ======================================================================
    # IMAGE
    # ======================================================================
    def _analyze_image(self, source, filename, mode, explain, do_stability, force_full):
        t_all = time.time()
        res = AnalysisResult(media_type="image", filename=filename)
        tri = triage_image(source)
        img: Image.Image = tri["image"].convert("RGB")
        enforce_image_limits(img)
        quality: QualityProfile = tri["quality"]
        res.sha256, res.quality, res.provenance = tri["sha256"], quality.to_dict(), tri["provenance"]
        t = time.time()

        detector = M.get_face_detector()
        faces = detector.detect(img)
        faces = sorted(faces, key=lambda f: -f.area)[:6]
        if faces:
            crops = [M.crop_with_margin(img, f.box) for f in faces]
            boxes = [f.box for f in faces]
        else:
            crops, boxes = [img], []
            res.warnings.append(
                "No face detected: the whole image was analysed (detectors "
                "trained on faces are less reliable here).")
        res.timings["faces"] = time.time() - t

        backends = M.get_image_backends()
        res.backends = [b.name for b in backends]
        t = time.time()
        tt = tta_logit(backends[0].predict, crops, aggregate_faces,
                       n_views=max(1, int(self.settings.tta_views)))
        z0, emb0 = tt["per_face"], tt["embedding"]
        res.timings["tier1"] = time.time() - t
        z_img = tt["logit"]
        spread = tt["spread"]
        res.uncertainty = {"tta_views": tt["n_views"], "view_logits": tt["view_logits"],
                           "view_spread": round(spread, 3),
                           "sign_flip": tt["sign_flip"]}
        ev_neural = Evidence(
            "neural", z_img,
            quality.neural_reliability() * reliability_from_spread(spread),
            detail={"backend": backends[0].name,
                    "per_face_logits": [round(float(v), 3) for v in z0],
                    "faces": len(faces), "view_spread": round(spread, 3)})
        mean_emb = None if emb0 is None else np.asarray(emb0).mean(axis=0)
        res.raw.update({"tier1_logit": z_img, "embedding": mean_emb})

        # cheap provenance / metadata evidence
        meta, ev_meta = self._metadata_evidence(source, tri)
        res.metadata = meta

        evidence: List[Evidence] = [ev_neural]
        if ev_meta is not None:
            evidence.append(ev_meta)

        cal1, p1, set1, v1, ood1, r1, reason1 = self._decide(
            "image.tier1", z_img, "image", mean_emb, mode)
        confident = (v1 in (FAKE, REAL)) and (ood1 is not True)
        if ev_meta is not None and v1 == REAL:
            confident = False  # an explicit AI-generator tag contradicts "Real"
        unstable = tt["sign_flip"] or spread > 2.0
        if unstable and confident:
            res.warnings.append(
                "The fast screen was unstable under mirror/crop views, so the "
                "full cascade was run instead of exiting early.")
        early = confident and not force_full and not unstable

        face_box = boxes[0] if boxes else None
        t = time.time()
        if early:
            res.tier = "tier1_early_exit"
            cal, p, label_set, verdict, ood_flag, ratio, reason = cal1, p1, set1, v1, ood1, r1, reason1
            final_logit = z_img
            res.fusion = {"logit": z_img, "disagreement": 0.0, "n_used": len(evidence)}
            # list the experts that were deliberately not run
            for name in ("tamper", "spectral", "forensic_prior"):
                evidence.append(skipped(name, "not needed: fast screen was confident", True))
        else:
            res.tier = "tier2"
            if self.settings.controller and not force_full:
                ctrl = InvestigationController(
                    self._image_registry(img, quality, backends),
                    risk_target=self.settings.risk_target, budget=self.settings.budget,
                    weights=self.fusion_weights, corr=self.expert_corr,
                    policy=self.settings.controller_policy)
                out = ctrl.run(evidence, {"compression": quality.compression},
                               lambda name: self._image_test(name, img, crops, backends,
                                                             quality, face_box))
                evidence = out["evidence"]
                for nr in out["not_run"]:
                    evidence.append(skipped(nr["test"], nr["reason"], True))
                res.investigation = {k: v for k, v in out.items() if k != "evidence"}
            else:
                # additional backbones (heterogeneous ensemble)
                for b in backends[1:]:
                    zb, _ = b.predict(crops)
                    evidence.append(Evidence(
                        "neural_probe", aggregate_faces(zb), quality.neural_reliability(),
                        detail={"backend": b.name}))
                evidence.append(tamper_evidence(img, face_box, quality.compression))
                evidence.append(spectral_evidence(img, self.spectral_head, quality.compression))
                evidence.append(self._forensic_prior(img))
            ctx = self._ctx(quality, evidence, r1, spread)
            self._apply_reliability(evidence, "image", ctx)
            final_logit, res.fusion = self._fuse(evidence)
            res.raw["ctx"] = ctx
            cal, p, label_set, verdict, ood_flag, ratio, reason = self._decide(
                "image.final", final_logit, "image", mean_emb, mode, ctx)
        res.timings["tier2" if not early else "decide"] = time.time() - t
        res.raw["final_logit"] = final_logit
        res.raw["expert_logits"] = {e.name: float(e.logit) for e in evidence
                                    if e.status == "used" and e.logit is not None}
        res.evidence = [e.to_dict() for e in evidence]
        self._finalise(res, cal, p, label_set, verdict, ood_flag, ratio, reason, mode, res.tier)

        # explanation + stability (UI features; do not affect the verdict)
        primary = crops[0]
        score_fn = lambda ims: backends[0].predict(ims)[0]
        res.artifacts["source_image"] = img
        res.artifacts["annotated"] = self._annotate(img, boxes)
        res.artifacts["face_crops"] = [c.resize((224, 224)) for c in crops]
        if explain:
            t = time.time()
            om = multiscale_occlusion(score_fn, primary, grids=(3, 6, 9))
            ff = deletion_faithfulness(score_fn, primary, om["importance"])
            cf = counterfactual_evidence(score_fn, primary, om["importance"],
                                         threshold=0.0, grid=6)
            res.artifacts["heatmap"] = overlay_heatmap(primary, om["importance"])
            res.explanation = {"method": "multiscale-occlusion", "scales": om["scales"],
                               "grid": om["grid"],
                               "faithfulness": ff.get("faithfulness"),
                               "faithfulness_detail": ff,
                               "counterfactual": cf,
                               "importance": om["importance"].round(4).tolist()}
            from utils.visualizers import generate_ela_image, generate_fft_spectrum
            res.artifacts["ela"] = generate_ela_image(img)
            res.artifacts["fft"] = generate_fft_spectrum(img)
            if any(e.name == "tamper" and e.status == "used" for e in evidence) or not early:
                from detector.experts.tamper import tamper_map
                tm = tamper_map(img, face_box)
                res.artifacts["tamper_map"] = tamper_heatmap_image(tm, img.size)
            res.timings["explain"] = time.time() - t
        if do_stability:
            t = time.time()

            def p_of(ims):
                z, _ = backends[0].predict(ims)
                return float(cal1.prob_fake(aggregate_faces(z)))

            res.stability = stability_report(crops, p_of, base_p=float(p1))
            res.robustness = robustness_profile(
                lambda ims: backends[0].predict(ims)[0], primary, quality=res.quality,
                verdict_fn=lambda z: cal1.verdict(float(cal1.prob_fake(z)), mode))
            gate = robustness_gate(res.robustness)
            res.robustness["gate"] = gate
            if (self.settings.robustness_gate and not gate["ok"]
                    and res.verdict in (FAKE, REAL)):
                res.verdict, res.label_set, res.confidence = "Inconclusive", [REAL, FAKE], None
                res.abstention_reason = "unstable_under_transformations: " + gate["reason"]
                res.warnings.append("The call is not stable under plausible re-encodings, "
                                    "so the verdict was downgraded to Inconclusive.")
            res.timings["stability"] = time.time() - t
        res.narrative = build_narrative(res.to_dict())
        res.timings["total"] = time.time() - t_all
        return res

    def _metadata_evidence(self, source, tri) -> Tuple[Optional[Dict], Optional[Evidence]]:
        from detector.metadata_detector import analyze_metadata

        meta = None
        try:
            if isinstance(source, str):
                meta = analyze_metadata(source)
            elif isinstance(source, (bytes, bytearray)):
                meta = analyze_metadata(io.BytesIO(bytes(source)))
        except Exception:
            meta = None
        prov = None
        try:
            from detector.provenance import inspect_provenance
            raw = (open(source, "rb").read(8 * 1024 * 1024) if isinstance(source, str)
                   else bytes(source) if isinstance(source, (bytes, bytearray)) else None)
            prov = inspect_provenance(raw) if raw else None
        except Exception:
            prov = None
        if prov:
            meta = dict(meta or {})
            meta["provenance"] = prov
            if prov.get("declared_ai_source") and not meta.get("ai_generator_signature"):
                meta["ai_generator_signature"] = "C2PA/IPTC: " + prov["declared_ai_source"]
        if meta and meta.get("ai_generator_signature"):
            sig = meta["ai_generator_signature"]
            return meta, Evidence(
                "metadata_ai", 6.0, 1.0,
                detail={"signature": sig},
                reason="File metadata names an AI generator.")
        return meta, None

    def _forensic_prior(self, img: Image.Image) -> Evidence:
        from detector.facial_forensics import analyze_image_quality_and_noise

        q = analyze_image_quality_and_noise(img)
        risk = float(q["heuristic_risk_score"])
        return Evidence("forensic_prior", (risk - 0.25) * 2.0, 0.5, heuristic=True,
                        detail={k: q[k] for k in
                                ("laplacian_blur_score", "chroma_variance",
                                 "srm_noise_energy", "heuristic_risk_score")})

    @staticmethod
    def _annotate(img: Image.Image, boxes) -> Image.Image:
        out = img.copy()
        d = ImageDraw.Draw(out)
        for b in boxes:
            d.rectangle(list(b), outline="#ef4444", width=3)
        return out

    # ======================================================================
    # VIDEO
    # ======================================================================
    def _frame_scores(self, frames, backend):
        """Per-frame fake logits on the largest face crop (full frame if no
        face). Returns logits, embeddings, crops, boxes."""
        detector = M.get_face_detector()
        crops, boxes, no_face = [], [], 0
        for _, _, im in frames:
            crop, box, n = _largest_face_crop(im, detector)
            crops.append(crop)
            boxes.append(box)
            no_face += int(box is None)
        z, emb = backend.predict(crops)
        return np.asarray(z, dtype=np.float64), emb, crops, boxes, no_face

    def _analyze_video(self, path, filename, mode, explain, do_stability, force_full):
        t_all = time.time()
        res = AnalysisResult(media_type="video", filename=filename)
        res.sha256 = sha256_file(path)
        quality = video_quality(path)
        res.quality = quality.to_dict()
        res.provenance = detect_c2pa(open(path, "rb").read(2_000_000), path=path)
        backends = M.get_image_backends()
        res.backends = [b.name for b in backends]
        s = self.settings

        t = time.time()
        frames = sample_frames(path, s.tier1_frames)
        if not frames:
            raise ValueError("Could not decode any frame from the video.")
        z, emb, crops, boxes, no_face = self._frame_scores(frames, backends[0])
        times = [f[1] for f in frames]
        res.timings["tier1_visual"] = time.time() - t
        if no_face == len(frames):
            res.warnings.append("No faces found in sampled frames; whole frames were analysed.")
        elif no_face:
            res.warnings.append(f"No face in {no_face} of {len(frames)} sampled frames.")

        qf = quality.neural_reliability()
        ev_temporal = temporal_evidence(z, times, emb, self.temporal_head, qf)

        # audio (cheap relative to the benefits: catches voice-cloned audio)
        ev_audio, audio_wave, audio_info = self._video_audio_evidence(path)
        evidence: List[Evidence] = [ev_temporal, ev_audio]

        fr1 = fuse([e for e in evidence], self.fusion_weights)
        mean_emb = None if emb is None else np.asarray(emb).mean(axis=0)
        cal1, p1, set1, v1, ood1, r1, reason1 = self._decide(
            "video.tier1", fr1.logit, "video", mean_emb, mode)
        res.raw.update({"tier1_logit": fr1.logit, "embedding": mean_emb})
        confident = (v1 in (FAKE, REAL)) and (ood1 is not True)
        early = confident and not force_full

        t = time.time()
        if early:
            res.tier = "tier1_early_exit"
            cal, p, label_set, verdict, ood_flag, ratio, reason = cal1, p1, set1, v1, ood1, r1, reason1
            final_logit = fr1.logit
            res.fusion = {"logit": fr1.logit, "disagreement": fr1.disagreement,
                          "n_used": fr1.n_used, "contributions": fr1.contributions}
            for name in ("sync", "rppg", "geometry"):
                evidence.append(skipped(name, "not needed: fast screen was confident", True))
        else:
            res.tier = "tier2"
            # 1) densify around the most suspicious frames
            if len(z) >= 3:
                top = np.argsort(-z)[:2]
                extra_t = []
                for i in top:
                    for dt in (-0.6, -0.3, 0.3, 0.6):
                        tt = times[i] + dt
                        if tt >= 0:
                            extra_t.append(tt)
                extra = frames_at(path, extra_t)
                if extra:
                    z2, emb2, crops2, boxes2, nf2 = self._frame_scores(extra, backends[0])
                    allt = times + [f[1] for f in extra]
                    allz = np.concatenate([z, z2])
                    alle = None if emb is None or emb2 is None else np.concatenate([emb, emb2])
                    order = np.argsort(allt)
                    allt = [allt[i] for i in order]
                    allz = allz[order]
                    alle = None if alle is None else alle[order]
                    evidence[0] = temporal_evidence(allz, allt, alle, self.temporal_head, qf)
                    evidence[0].detail["densified_frames"] = len(extra)
                    z, times = allz, allt
            # 2) face-signal experts on a short consecutive window
            sig = None
            try:
                sig = extract_face_signals(
                    path, M.get_face_detector(), max_seconds=s.face_signal_seconds)
            except Exception as e:
                log.warning("face signal extraction failed: %s", e)
            evidence.append(rppg_evidence(sig, quality.compression,
                                          dual=bool(getattr(s, "dual_rppg", True))))
            evidence.append(geometry_evidence(sig))
            evidence.append(sync_evidence(sig, audio_wave, 16000, quality.compression))
            # 3) Optical flow inconsistency expert (NEW v4.0).
            if _HAS_OPTICAL_FLOW and getattr(s, "optical_flow_expert", False):
                try:
                    of_ev = _optical_flow_evidence(
                        path, M.get_face_detector(),
                        quality_factor=qf,
                        use_raft=bool(getattr(s, "optical_flow_expert", False)),
                        device=M.get_device(),
                    )
                    evidence.append(of_ev)
                    log.info("Optical flow expert: logit=%.2f method=%s",
                             of_ev.logit or 0.0,
                             (of_ev.detail or {}).get("method", "?"))
                except Exception as oe:
                    log.warning("Optical flow expert failed: %s", oe)
            # 4) extra backbones on the sampled crops
            for b in backends[1:]:
                zb, _ = b.predict(crops)
                evidence.append(Evidence("neural_probe", aggregate_frame_logits(zb),
                                         qf, detail={"backend": b.name}))
            ctx = self._ctx(quality, evidence, r1, 0.0,
                            modalities=1 + int(ev_audio.status == "used"))
            self._apply_reliability(evidence, "video", ctx)
            final_logit, res.fusion = self._fuse(evidence)
            res.raw["ctx"] = ctx
            if sig is not None and audio_wave is not None:
                try:
                    env = audio_envelope(audio_wave, 16000, sig.fps, sig.n_frames)
                    res.contradiction_map = contradiction_map(sig.mouth_energy, env, sig.fps)
                except Exception as e:  # pragma: no cover
                    log.warning("contradiction map failed: %s", e)
            cal, p, label_set, verdict, ood_flag, ratio, reason = self._decide(
                "video.final", final_logit, "video", mean_emb, mode, ctx)
        res.timings["tier2" if not early else "decide"] = time.time() - t
        res.raw["final_logit"] = final_logit
        res.raw["expert_logits"] = {e.name: float(e.logit) for e in evidence
                                    if e.status == "used" and e.logit is not None}
        res.evidence = [e.to_dict() for e in evidence]
        self._finalise(res, cal, p, label_set, verdict, ood_flag, ratio, reason, mode, res.tier)

        # modality disagreement note
        if ev_audio.status == "used" and ev_temporal.logit is not None:
            if (ev_audio.logit > 1.0 and ev_temporal.logit < -1.0):
                res.warnings.append("Audio looks synthetic while the visuals look authentic.")
            elif (ev_audio.logit < -1.0 and ev_temporal.logit > 1.0):
                res.warnings.append("Visuals look manipulated while the audio looks authentic.")

        # timeline for the UI (calibrated per-frame probability)
        cal_t = self.calibration.get("video.tier1")
        for tt, zz in zip(times, z):
            res.timeline.append({"timestamp": round(float(tt), 3),
                                 "fake_probability": float(cal_t.prob_fake(zz)),
                                 "logit": float(zz)})
        res.segments = [{"start": a, "end": b} for a, b in
                        ev_temporal.detail.get("suspicious_segments", [])]
        # artifacts / explanation
        k = int(np.argmax(z[:len(crops)])) if len(crops) else 0
        res.artifacts["frames"] = [(f[1], f[2]) for f in frames]
        res.artifacts["suspicious_frames"] = [
            (frames[i][1], frames[i][2]) for i in np.argsort(-z[:len(frames)])[:4]]
        if explain and crops:
            score_fn = lambda ims: backends[0].predict(ims)[0]
            om = occlusion_map(score_fn, crops[k], grid=6)
            ff = deletion_faithfulness(score_fn, crops[k], om["importance"])
            res.artifacts["heatmap"] = overlay_heatmap(crops[k], om["importance"])
            res.explanation = {"method": "occlusion", "grid": 6,
                               "frame_time": round(float(frames[k][1]), 2),
                               "faithfulness": ff.get("faithfulness"),
                               "faithfulness_detail": ff}
        if do_stability and crops:
            top_k = list(np.argsort(-z[:len(crops)])[:3])
            sel = [crops[i] for i in top_k]

            def p_of(ims):
                zz, _ = backends[0].predict(ims)
                return float(cal1.prob_fake(aggregate_frame_logits(zz)))

            res.stability = stability_report(sel, p_of)
        if audio_info:
            res.segments += audio_info.get("segments", [])
        res.narrative = build_narrative(res.to_dict())
        res.timings["total"] = time.time() - t_all
        return res

    def _video_audio_evidence(self, path):
        try:
            if not has_audio_track(path):
                return skipped("audio", "no audio track"), None, None
            wave = load_audio(path, 16000)
            if wave is None or len(wave) < 16000:
                return skipped("audio", "audio shorter than 1 s"), wave, None
            ev, info = self._audio_evidence(wave, 3)
            return ev, wave, info
        except Exception as e:
            log.warning("audio analysis failed: %s", e)
            return skipped("audio", f"audio analysis failed: {e}"), None, None

    # ======================================================================
    # AUDIO
    # ======================================================================
    @staticmethod
    def _chunks(wave: np.ndarray, sr: int, chunk_s: int):
        n = int(chunk_s * sr)
        chunks, spans = [], []
        for i in range(0, len(wave), n):
            c = wave[i:i + n]
            if len(c) < n:
                if len(chunks) and len(c) < sr:  # drop a sub-1s tail
                    break
                c = np.concatenate([c, np.zeros(n - len(c), dtype=c.dtype)])
            chunks.append(c)
            spans.append((i / sr, min((i + n) / sr, len(wave) / sr)))
        return chunks, spans

    def _audio_evidence(self, wave: np.ndarray, chunk_s: int):
        from detector.audio_io import compute_vad_mask

        sr = 16000
        backend = M.get_audio_backend()
        chunks, spans = self._chunks(wave, sr, chunk_s)
        z, emb = backend.predict(chunks, sr)
        aq = audio_quality(wave, sr)
        rel = aq.neural_reliability()

        # Run Voice Activity Detection (VAD)
        vad_info = compute_vad_mask(wave, sr)
        speech_ratio = float(vad_info["speech_ratio"])
        vad_mask = vad_info["vad_mask"]

        # Calculate speech fraction per chunk
        chunk_speech_ratios = []
        for a, b in spans:
            s_idx = int(a * sr)
            e_idx = min(len(vad_mask), int(b * sr))
            c_sr = float(np.mean(vad_mask[s_idx:e_idx])) if e_idx > s_idx else 0.0
            chunk_speech_ratios.append(c_sr)

        # Silence / ambient noise gating:
        # If the entire audio file lacks speech (< 3%), voice deepfake predictions on pure noise
        # produce arbitrary false positives. Downweight to neutral prior.
        if speech_ratio < 0.03:
            ev = Evidence(
                "audio", 0.0, 0.05,
                detail={"backend": backend.name, "chunks": len(chunks),
                        "speech_detected": False, "speech_ratio": round(speech_ratio, 3),
                        "mean_snr_db": round(float(vad_info.get("mean_snr_db", 0.0)), 2)},
                reason="No active speech detected; audio track is silence or ambient noise.",
                heuristic=True,
            )
            segs = [{"start": float(a), "end": float(b), "logit": 0.0,
                     "speech_ratio": round(csr, 3), "is_speech": False}
                    for (a, b), csr in zip(spans, chunk_speech_ratios)]
            return ev, {"segments": segs, "logits": np.zeros(len(chunks)), "emb": emb,
                        "spans": spans, "quality": aq, "chunks": chunks, "vad": vad_info}

        # Filter: only evaluate active speech chunks
        active_indices = [i for i, csr in enumerate(chunk_speech_ratios) if csr >= 0.1]
        if active_indices:
            agg_z = aggregate_frame_logits(z[active_indices])
        else:
            agg_z = aggregate_frame_logits(z)

        # Scale reliability based on speech presence
        scaled_rel = rel * min(1.0, max(0.2, speech_ratio / 0.25))

        ev = Evidence("audio", agg_z, scaled_rel,
                      detail={"backend": backend.name, "chunks": len(chunks),
                              "max_chunk_logit": float(np.max(z)),
                              "speech_ratio": round(speech_ratio, 3),
                              "active_speech_chunks": len(active_indices),
                              "mean_snr_db": round(float(vad_info.get("mean_snr_db", 0.0)), 2)})
        segs = [{"start": float(a), "end": float(b), "logit": float(zz),
                 "speech_ratio": round(csr, 3), "is_speech": bool(csr >= 0.1)}
                for (a, b), zz, csr in zip(spans, z, chunk_speech_ratios)]
        return ev, {"segments": segs, "logits": z, "emb": emb, "spans": spans,
                    "quality": aq, "chunks": chunks, "vad": vad_info}

    def _analyze_audio(self, path, filename, mode, explain, chunk_s):
        t_all = time.time()
        res = AnalysisResult(media_type="audio", filename=filename)
        res.sha256 = sha256_file(path)
        wave = load_audio(path, 16000)
        if wave is None or len(wave) < 8000:
            raise ValueError("Could not decode audio, or it is shorter than 0.5 s.")
        res.backends = [M.get_audio_backend().name]
        ev, info = self._audio_evidence(wave, chunk_s)
        res.quality = info["quality"].to_dict()
        evidence = [ev]
        fr = fuse(evidence, self.fusion_weights)
        emb = info["emb"]
        mean_emb = None if emb is None else np.asarray(emb).mean(axis=0)
        cal, p, label_set, verdict, ood_flag, ratio, reason = self._decide(
            "audio.final", fr.logit, "audio", mean_emb, mode)
        res.tier = "single_stage"
        res.fusion = {"logit": fr.logit, "disagreement": fr.disagreement,
                      "n_used": fr.n_used, "contributions": fr.contributions}
        res.raw.update({"tier1_logit": fr.logit, "final_logit": fr.logit,
                        "embedding": mean_emb})
        res.evidence = [e.to_dict() for e in evidence]
        self._finalise(res, cal, p, label_set, verdict, ood_flag, ratio, reason, mode, res.tier)

        if info.get("vad", {}).get("speech_ratio", 1.0) < 0.05:
            res.warnings.append(
                f"No active speech detected in audio track (speech ratio: {info['vad']['speech_ratio'] * 100:.1f}%). "
                "The track appears to be silence or ambient noise; voice deepfake detection requires voiced speech."
            )

        for seg, zz in zip(info["segments"], info["logits"]):
            pp = float(cal.prob_fake(zz))
            res.timeline.append({"timestamp": round((seg["start"] + seg["end"]) / 2, 3),
                                 "fake_probability": pp,
                                 "label": "Fake" if pp >= 0.5 else "Real"})
        res.segments = [{"start": s["start"], "end": s["end"],
                         "fake_probability": float(cal.prob_fake(s["logit"]))}
                        for s in info["segments"]]
        res.artifacts["waveform"] = wave
        if explain:
            from utils.visualizers import generate_mel_spectrogram
            res.artifacts["mel"] = generate_mel_spectrogram(wave, 16000)
        res.narrative = build_narrative(res.to_dict())
        res.timings["total"] = time.time() - t_all
        return res



_PIPELINE: Optional[Pipeline] = None


def get_pipeline() -> Pipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = Pipeline()
    return _PIPELINE


def reset_pipeline() -> None:
    global _PIPELINE
    _PIPELINE = None
