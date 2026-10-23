"""Per-expert implementation & readiness audit (P0).

Produces a table (JSON + markdown) answering, for each detector branch:

* is the code path implemented?
* is a trained checkpoint present?
* does it load successfully?
* training / calibration status (if known)
* is it a heuristic, a frozen pretrained model, or a trained head?
* approximate role in the cascade

This does **not** claim real-world accuracy. It only documents what the
running software can actually invoke.

    python -m scripts.expert_audit --out artifacts/expert_audit.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger("deepfake.audit")


def _exists(path: str) -> bool:
    return bool(path) and os.path.isfile(path)


def _try_load_image_backends() -> List[Dict[str, Any]]:
    rows = []
    try:
        from detector import models as M
        from detector.config import get_settings
        s = get_settings()
        t0 = time.time()
        backends = M.get_image_backends()
        load_s = round(time.time() - t0, 2)
        for b in backends:
            row = {
                "name": getattr(b, "name", type(b).__name__),
                "kind": getattr(b, "kind", "unknown"),
                "implemented": True,
                "loaded": True,
                "load_seconds": load_s,
                "checkpoint": getattr(b, "model_id", None)
                    or getattr(b, "embed_model_id", None)
                    or getattr(b, "probe_path", None),
                "role": "Tier-1 / semantic visual evidence",
                "heuristic": False,
                "notes": "",
            }
            if row["kind"] == "frequency":
                row["role"] = "Frequency / spectral branch (stub unless spectral head trained)"
                row["notes"] = "Stub until F3Net/SPSL or spectral_head.npz is available"
            if row["kind"] == "siglip":
                row["role"] = "Complementary semantic embeddings (not a deepfake classifier alone)"
            if row["kind"] == "probe":
                row["role"] = "Primary recommended Tier-1 (frozen encoder + linear probe)"
                row["notes"] = "Requires artifacts/probe.npz trained on your data"
            rows.append(row)
    except Exception as e:
        rows.append({
            "name": "image_backends", "implemented": True, "loaded": False,
            "error": f"{type(e).__name__}: {e}", "heuristic": False,
        })
    return rows


def _artifact_status(artifact_dir: str) -> Dict[str, Any]:
    checks = {
        "probe.npz": "image linear probe (DINOv2/3 features)",
        "spectral_head.npz": "frequency / spectral expert head",
        "temporal_head.pt": "video temporal head",
        "audio_probe.npz": "audio linear probe (WavLM features)",
        "calibration.json": "Platt + conformal calibration bundle",
        "fusion_weights.json": "learned expert fusion weights",
        "ood_image.npz": "image OOD / Mahalanobis model",
        "ood_video.npz": "video OOD model",
        "ood_audio.npz": "audio OOD model",
        "audio_finetuned/": "fine-tuned HF audio classifier directory",
    }
    out = {}
    for name, desc in checks.items():
        path = os.path.join(artifact_dir, name)
        present = os.path.isdir(path) if name.endswith("/") else _exists(path)
        meta = {"present": present, "description": desc, "path": path}
        if present and name.endswith(".npz"):
            try:
                import numpy as np
                z = np.load(path, allow_pickle=True)
                meta["keys"] = list(z.files)
                if "val_auc" in z.files:
                    meta["val_auc"] = float(z["val_auc"])
            except Exception as e:
                meta["read_error"] = str(e)
        if present and name.endswith(".json"):
            try:
                with open(path) as f:
                    j = json.load(f)
                meta["keys"] = list(j.keys()) if isinstance(j, dict) else type(j).__name__
            except Exception as e:
                meta["read_error"] = str(e)
        out[name] = meta
    return out


def _static_experts(artifact_dir: str) -> List[Dict[str, Any]]:
    """Catalog every expert the cascade can invoke."""
    arts = _artifact_status(artifact_dir)
    def head(name):
        return arts.get(name, {}).get("present", False)

    return [
        {
            "name": "tier1_classifier",
            "implemented": True,
            "trained_head": False,
            "pretrained_weights": True,
            "heuristic": False,
            "role": "Fast image/video frame screen (HF classifier)",
            "checkpoint_env": "IMAGE_MODEL_ID",
            "skips_when": "never (always available if model downloads)",
        },
        {
            "name": "dinov2_probe",
            "implemented": True,
            "trained_head": head("probe.npz"),
            "pretrained_weights": True,
            "heuristic": False,
            "role": "Recommended primary visual evidence (frozen DINOv2 + linear probe)",
            "checkpoint_env": "EMBED_MODEL_ID + artifacts/probe.npz",
            "skips_when": "probe.npz missing → falls back to classifier only",
        },
        {
            "name": "siglip_semantic",
            "implemented": True,
            "trained_head": False,
            "pretrained_weights": True,
            "heuristic": False,
            "role": "Complementary semantic embeddings",
            "checkpoint_env": "SIGLIP_MODEL_ID",
            "skips_when": "backend load fails",
        },
        {
            "name": "spectral",
            "implemented": True,
            "trained_head": head("spectral_head.npz"),
            "pretrained_weights": False,
            "heuristic": False,
            "role": "Frequency / upsampling fingerprint",
            "checkpoint_env": "artifacts/spectral_head.npz",
            "skips_when": "no trained head or image too small (never guessed)",
        },
        {
            "name": "tamper_localization",
            "implemented": True,
            "trained_head": False,
            "pretrained_weights": False,
            "heuristic": True,
            "role": "Face X-ray style blending / patch suspicion map",
            "checkpoint_env": None,
            "skips_when": "image too small",
        },
        {
            "name": "temporal_head",
            "implemented": True,
            "trained_head": head("temporal_head.pt"),
            "pretrained_weights": False,
            "heuristic": not head("temporal_head.pt"),
            "role": "Video frame-to-frame inconsistency (statistical aggregator or learned head)",
            "checkpoint_env": "artifacts/temporal_head.pt",
            "skips_when": "no video / too few frames",
        },
        {
            "name": "audio_hf",
            "implemented": True,
            "trained_head": head("audio_finetuned/"),
            "pretrained_weights": True,
            "heuristic": False,
            "role": "Speech spoofing / deepfake audio classifier",
            "checkpoint_env": "AUDIO_MODEL_ID or artifacts/audio_finetuned",
            "skips_when": "no audio track",
        },
        {
            "name": "audio_probe",
            "implemented": True,
            "trained_head": head("audio_probe.npz"),
            "pretrained_weights": True,
            "heuristic": False,
            "role": "WavLM frozen features + linear probe",
            "checkpoint_env": "WAVLM_MODEL_ID + artifacts/audio_probe.npz",
            "skips_when": "probe missing (HF audio still runs)",
        },
        {
            "name": "av_sync",
            "implemented": True,
            "trained_head": False,
            "pretrained_weights": False,
            "heuristic": True,
            "role": "Lip-sync / audio-visual temporal alignment (heuristic)",
            "checkpoint_env": None,
            "skips_when": "missing face track or audio",
        },
        {
            "name": "rppg_facesignals",
            "implemented": True,
            "trained_head": False,
            "pretrained_weights": False,
            "heuristic": True,
            "role": "Remote photoplethysmography + geometry (quality-gated heuristic)",
            "checkpoint_env": None,
            "skips_when": "short / unstable / poorly lit face track — missing pulse ≠ deepfake",
        },
        {
            "name": "ood_mahalanobis",
            "implemented": True,
            "trained_head": head("ood_image.npz") or head("ood_video.npz") or head("ood_audio.npz"),
            "pretrained_weights": False,
            "heuristic": False,
            "role": "Out-of-distribution / generator-shift detector",
            "checkpoint_env": "artifacts/ood_*.npz from calibration",
            "skips_when": "no OOD model fitted",
        },
        {
            "name": "calibration_conformal",
            "implemented": True,
            "trained_head": head("calibration.json"),
            "pretrained_weights": False,
            "heuristic": False,
            "role": "Platt scaling + class-conditional conformal sets → Real/Fake/Inconclusive",
            "checkpoint_env": "artifacts/calibration.json",
            "skips_when": "unfitted → conservative margin around 0.5, reports uncalibrated",
        },
        {
            "name": "voi_controller",
            "implemented": True,
            "trained_head": False,
            "pretrained_weights": False,
            "heuristic": False,
            "role": "RAAFC adaptive evidence selection (value-of-information)",
            "checkpoint_env": "DEEPFAKE_CONTROLLER=1",
            "skips_when": "controller disabled",
        },
        {
            "name": "gls_fusion",
            "implemented": True,
            "trained_head": head("fusion_weights.json"),
            "pretrained_weights": False,
            "heuristic": False,
            "role": "Correlation-aware evidence fusion (generalised least squares)",
            "checkpoint_env": "DEEPFAKE_DEP_FUSION=1",
            "skips_when": "dependent_fusion off → weighted mean",
        },
    ]


def run_audit(artifact_dir: Optional[str] = None, try_load: bool = True) -> Dict[str, Any]:
    from detector.config import get_settings
    s = get_settings()
    adir = artifact_dir or s.artifact_dir
    report: Dict[str, Any] = {
        "artifact_dir": os.path.abspath(adir),
        "device": s.device,
        "settings": {
            "controller": s.controller,
            "dependent_fusion": s.dependent_fusion,
            "robustness_gate": s.robustness_gate,
            "localization": getattr(s, "localization", None),
            "image_model_id": s.image_model_id,
            "embed_model_id": s.embed_model_id,
            "audio_model_id": s.audio_model_id,
        },
        "artifacts": _artifact_status(adir),
        "experts": _static_experts(adir),
        "honest_status": (
            "This audit reports implementation and checkpoint presence only. "
            "It does NOT measure real-world deepfake detection accuracy. "
            "Run train.evaluate on held-out, unseen-generator media after training "
            "and calibration to obtain performance numbers."
        ),
    }
    if try_load:
        report["live_image_backends"] = _try_load_image_backends()
    # Summary counts
    experts = report["experts"]
    report["summary"] = {
        "n_experts": len(experts),
        "n_with_trained_head": sum(1 for e in experts if e.get("trained_head")),
        "n_heuristic": sum(1 for e in experts if e.get("heuristic")),
        "n_pretrained_only": sum(
            1 for e in experts
            if e.get("pretrained_weights") and not e.get("trained_head") and not e.get("heuristic")
        ),
        "calibration_fitted": report["artifacts"].get("calibration.json", {}).get("present", False),
        "probe_fitted": report["artifacts"].get("probe.npz", {}).get("present", False),
    }
    return report


def to_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Expert readiness audit",
        "",
        report["honest_status"],
        "",
        f"**Artifact dir:** `{report['artifact_dir']}`",
        f"**Device:** `{report['device']}`",
        "",
        "## Summary",
        "",
        f"- Experts catalogued: **{report['summary']['n_experts']}**",
        f"- With trained heads present: **{report['summary']['n_with_trained_head']}**",
        f"- Heuristic-only: **{report['summary']['n_heuristic']}**",
        f"- Pretrained weights only (no task head): **{report['summary']['n_pretrained_only']}**",
        f"- Calibration fitted: **{report['summary']['calibration_fitted']}**",
        f"- Image probe fitted: **{report['summary']['probe_fitted']}**",
        "",
        "## Experts",
        "",
        "| Name | Implemented | Trained head | Pretrained | Heuristic | Role | Skips when |",
        "|------|-------------|--------------|------------|-----------|------|------------|",
    ]
    for e in report["experts"]:
        lines.append(
            f"| `{e['name']}` | {e.get('implemented')} | {e.get('trained_head')} | "
            f"{e.get('pretrained_weights')} | {e.get('heuristic')} | {e.get('role','')} | "
            f"{e.get('skips_when','')} |"
        )
    lines += ["", "## Artifacts", ""]
    for name, meta in report["artifacts"].items():
        flag = "✓" if meta.get("present") else "✗"
        lines.append(f"- {flag} `{name}` — {meta.get('description','')}")
    lines += ["", "## Next steps for real evaluation (P0)", "",
              "1. Train probe / spectral / temporal / audio heads on real data.",
              "2. Fit calibration on a **disjoint** calib split.",
              "3. Evaluate on **unseen generators**: `python -m train.evaluate ...`",
              "4. Run fusion baselines: `python -m train.evaluate_ablation ...`",
              "5. Report AUC / ECE / abstention coverage — never claim accuracy from unit tests alone.",
              ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/expert_audit.json")
    ap.add_argument("--artifact-dir", default=None)
    ap.add_argument("--no-load", action="store_true",
                    help="skip live model loading (faster, offline)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if a.artifact_dir:
        os.environ["DEEPFAKE_ARTIFACT_DIR"] = a.artifact_dir
    report = run_audit(try_load=not a.no_load)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    md_path = os.path.splitext(a.out)[0] + ".md"
    with open(md_path, "w") as f:
        f.write(to_markdown(report))
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {a.out} and {md_path}")
    return report


if __name__ == "__main__":
    main()
