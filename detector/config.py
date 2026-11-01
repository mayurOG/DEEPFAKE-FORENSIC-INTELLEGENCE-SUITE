"""Central configuration — v4.0 (Next-Generation).

Everything is overridable through environment variables (or a .env file) so
the same code runs on a laptop CPU and on a GPU box without edits.

New in v4.0:
    * DINOv2-Large encoder (307M vs 86M base)
    * SigLIP-SO400M-patch14-384 semantic branch (400M, 384px)
    * Real NPR frequency expert (replaces the stub that returned zeros)
    * Multi-encoder audio ensemble (WavLM-Large + Wav2Vec2-XLSR)
    * VideoMAE-v2 temporal backbone
    * Generator attribution module (GAN / Diffusion / XTTS / RVC)
    * Optical flow inconsistency expert
    * LLM forensic narrative (optional OpenAI / Gemini)
    * GradCAM++ saliency maps
    * Tightened conformal alpha 0.10 → 0.05
    * Dual rPPG (CHROM + POS)
"""
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value


def _bool(name: str, default: bool = False) -> bool:
    return _env(name, "1" if default else "0") == "1"


@dataclass(frozen=True)
class Settings:
    # -------------------------------------------------------------------------
    # Image models
    # -------------------------------------------------------------------------
    # Primary Tier-1 image deepfake classifier.
    # UPGRADED: dima806/deepfake_vs_real_image_detection (better calibrated).
    image_model_id: str = field(default_factory=lambda: _env(
        "IMAGE_MODEL_ID", "dima806/deepfake_vs_real_image_detection"
    ))
    # Diffusion-specific detector (SDXL / DALL-E / Flux / MidJourney).
    # NEW: catches modern diffusion outputs that GAN-trained classifiers miss.
    diffusion_model_id: str = field(default_factory=lambda: _env(
        "DIFFUSION_MODEL_ID", "umm-maybe/AI-image-detector"
    ))
    # Foundation encoder for features + OOD.
    # UPGRADED: dinov2-LARGE (307M) vs old dinov2-base (86M).
    embed_model_id: str = field(default_factory=lambda: _env(
        "EMBED_MODEL_ID", "facebook/dinov2-large"
    ))
    # Semantic branch.
    # UPGRADED: siglip-so400m-patch14-384 (400M, 384px) vs old base/224px.
    siglip_model_id: str = field(default_factory=lambda: _env(
        "SIGLIP_MODEL_ID", "google/siglip-so400m-patch14-384"
    ))
    # Frequency domain backbone (empty = use internal NPR expert).
    frequency_model_id: str = field(default_factory=lambda: _env(
        "FREQUENCY_MODEL_ID", ""
    ))

    # -------------------------------------------------------------------------
    # Audio models
    # -------------------------------------------------------------------------
    # Primary audio deepfake detector.
    # UPGRADED: Better coverage of voice cloning and TTS systems.
    audio_model_id: str = field(default_factory=lambda: _env(
        "AUDIO_MODEL_ID", "MelodyFlow/audio-deepfake-detection"
    ))
    # Fallback if primary fails to load.
    audio_model_id_fallback: str = field(default_factory=lambda: _env(
        "AUDIO_MODEL_ID_FALLBACK", "mo-thecreator/Deepfake-audio-detection"
    ))
    # WavLM-Large speech encoder (SOTA speech representation).
    wavlm_model_id: str = field(default_factory=lambda: _env(
        "WAVLM_MODEL_ID", "microsoft/wavlm-large"
    ))
    # NEW: Secondary speech encoder for multi-encoder audio ensemble.
    # Wav2Vec2-XLSR-53 = multilingual, catches voice clones across languages.
    wav2vec2_model_id: str = field(default_factory=lambda: _env(
        "WAV2VEC2_MODEL_ID", "facebook/wav2vec2-large-xlsr-53"
    ))
    # NEW: Enable multi-encoder audio ensemble.
    audio_ensemble: bool = field(default_factory=lambda: _bool("AUDIO_ENSEMBLE", True))

    # -------------------------------------------------------------------------
    # Video models
    # -------------------------------------------------------------------------
    # UPGRADED: VideoMAE-v2 temporal backbone.
    video_temporal_model_id: str = field(default_factory=lambda: _env(
        "VIDEO_TEMPORAL_MODEL_ID", "MCG-NJU/videomae-base-ssv2"
    ))
    # NEW: RAFT-based optical flow inconsistency expert.
    optical_flow_expert: bool = field(default_factory=lambda: _bool("OPTICAL_FLOW_EXPERT", True))
    # NEW: Enhanced lip-sync scoring.
    enhanced_lip_sync: bool = field(default_factory=lambda: _bool("ENHANCED_LIP_SYNC", True))

    # -------------------------------------------------------------------------
    # New expert modules
    # -------------------------------------------------------------------------
    # NEW: Generator attribution (GAN | Diffusion | XTTS | RVC).
    generator_attribution: bool = field(default_factory=lambda: _bool("GENERATOR_ATTRIBUTION", True))
    # NEW: GradCAM++ saliency (faster + more accurate than occlusion).
    gradcam_explain: bool = field(default_factory=lambda: _bool("GRADCAM_EXPLAIN", True))
    # NEW: Real NPR frequency expert (NOT a stub; catches upsampling grids).
    npr_expert: bool = field(default_factory=lambda: _bool("NPR_EXPERT", True))
    # NEW: Dual rPPG (CHROM + POS hybrid).
    dual_rppg: bool = field(default_factory=lambda: _bool("DEEPFAKE_DUAL_RPPG", True))

    # -------------------------------------------------------------------------
    # LLM narrative generation
    # -------------------------------------------------------------------------
    # "openai" | "gemini" | "none" (rule-based fallback).
    llm_narrative_provider: str = field(default_factory=lambda: _env(
        "LLM_NARRATIVE_PROVIDER", "none"
    ))
    llm_narrative_model: str = field(default_factory=lambda: _env(
        "LLM_NARRATIVE_MODEL", "gpt-4o-mini"
    ))
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY", ""))

    # -------------------------------------------------------------------------
    # Authentication
    # -------------------------------------------------------------------------
    hf_token: str = field(default_factory=lambda: _env("HF_TOKEN", ""))

    # -------------------------------------------------------------------------
    # Runtime
    # -------------------------------------------------------------------------
    artifact_dir: str = field(default_factory=lambda: _env("DEEPFAKE_ARTIFACT_DIR", "artifacts"))
    # "auto" | "cpu" | "cuda" | "mps"
    device: str = field(default_factory=lambda: _env("DEEPFAKE_DEVICE", "auto"))
    # Half-precision inference (GPU only; set 1 to enable).
    fp16: bool = field(default_factory=lambda: _bool("DEEPFAKE_FP16", False))

    # -------------------------------------------------------------------------
    # Statistical guarantees
    # -------------------------------------------------------------------------
    # Conformal risk level. TIGHTENED: 0.05 (95%) vs old 0.10.
    alpha: float = field(default_factory=lambda: float(_env("DEEPFAKE_ALPHA", "0.05")))
    tier1_frames: int = field(default_factory=lambda: int(_env("DEEPFAKE_TIER1_FRAMES", "16")))
    tta_views: int = field(default_factory=lambda: int(_env("DEEPFAKE_TTA_VIEWS", "5")))

    # -------------------------------------------------------------------------
    # EGAFR v2 research modules
    # -------------------------------------------------------------------------
    controller: bool = field(default_factory=lambda: _env("DEEPFAKE_CONTROLLER", "1") == "1")
    risk_target: float = field(default_factory=lambda: float(_env("DEEPFAKE_RISK_TARGET", "0.04")))
    budget: float = field(default_factory=lambda: float(_env("DEEPFAKE_BUDGET", "2.5")))
    controller_policy: str = field(default_factory=lambda: _env("DEEPFAKE_CONTROLLER_POLICY", "voi"))
    independent_experts: bool = field(default_factory=lambda: _env("DEEPFAKE_INDEPENDENT", "0") == "1")
    dependent_fusion: bool = field(default_factory=lambda: _env("DEEPFAKE_DEP_FUSION", "1") == "1")
    conditional: bool = field(default_factory=lambda: _env("DEEPFAKE_CONDITIONAL", "0") == "1")
    robustness_gate: bool = field(default_factory=lambda: _env("DEEPFAKE_ROBUSTNESS_GATE", "1") == "1")
    min_sufficiency: float = field(default_factory=lambda: float(_env("DEEPFAKE_MIN_SUFFICIENCY", "0")))
    face_signal_seconds: float = field(default_factory=lambda: float(_env("DEEPFAKE_FACE_SECONDS", "12")))
    localization: bool = field(default_factory=lambda: _env("DEEPFAKE_LOCALIZATION", "1") == "1")
    provenance_analysis: bool = field(default_factory=lambda: _env("DEEPFAKE_PROVENANCE", "1") == "1")
    # NEW: SHAP-based expert importance attribution.
    shap_attribution: bool = field(default_factory=lambda: _bool("DEEPFAKE_SHAP_ATTRIBUTION", False))

    # -------------------------------------------------------------------------
    # Media ingestion security & resource limits
    # -------------------------------------------------------------------------
    max_upload_bytes: int = field(default_factory=lambda: int(_env(
        "DEEPFAKE_MAX_UPLOAD_BYTES", str(150 * 1024 * 1024)
    )))
    max_image_pixels: int = field(default_factory=lambda: int(_env(
        "DEEPFAKE_MAX_IMAGE_PIXELS", str(25_000_000)
    )))
    max_video_seconds: float = field(default_factory=lambda: float(_env("DEEPFAKE_MAX_VIDEO_SECONDS", "180")))
    max_audio_seconds: float = field(default_factory=lambda: float(_env("DEEPFAKE_MAX_AUDIO_SECONDS", "600")))
    max_analysis_seconds: float = field(default_factory=lambda: float(_env("DEEPFAKE_MAX_ANALYSIS_SECONDS", "240")))
    max_video_frames: int = field(default_factory=lambda: int(_env("DEEPFAKE_MAX_VIDEO_FRAMES", "96")))

    # -------------------------------------------------------------------------
    # UI
    # -------------------------------------------------------------------------
    enable_pdf_export: bool = field(default_factory=lambda: _bool("ENABLE_PDF_EXPORT", True))
    show_model_status: bool = field(default_factory=lambda: _bool("SHOW_MODEL_STATUS", True))


def get_settings() -> Settings:
    return Settings()


def artifact_path(name: str) -> str:
    return os.path.join(get_settings().artifact_dir, name)
