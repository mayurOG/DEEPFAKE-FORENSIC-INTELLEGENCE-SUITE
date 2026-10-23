"""Central configuration.

Everything is overridable through environment variables (or a .env file) so
the same code runs on a laptop CPU and on a GPU box without edits.
"""
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value


@dataclass(frozen=True)
class Settings:
    # Pretrained image classifier used as the default Tier-1 backbone.
    # Recommended stack: keep a strong semantic detector (e.g. SigLIP-derived
    # or a dedicated deepfake classifier) and pair with DINOv2/DINOv3 embeddings.
    image_model_id: str = field(default_factory=lambda: _env(
        "IMAGE_MODEL_ID", "prithivMLmods/deepfake-detector-model-v1"
    ))
    # Primary foundation encoder for features + OOD (DINOv2 is the practical
    # stand-in; set to a DINOv3 checkpoint when available). Used with a linear
    # probe when trained. Empty = rely on the classifier above.
    embed_model_id: str = field(default_factory=lambda: _env(
        "EMBED_MODEL_ID", "facebook/dinov2-base"
    ))
    # Optional complementary semantic branch (SigLIP / CLIP-style).
    siglip_model_id: str = field(default_factory=lambda: _env(
        "SIGLIP_MODEL_ID", "google/siglip-base-patch16-224"
    ))
    # Frequency-domain expert backbone preference (spectral head is always
    # available; F3Net/SPSL-style heads can be pointed here when trained).
    frequency_model_id: str = field(default_factory=lambda: _env(
        "FREQUENCY_MODEL_ID", ""
    ))
    # Audio deepfake detector (AASIST-style or XLS-R + head). Default is a
    # public HF checkpoint; replace with a fine-tuned AASIST / WavLM head.
    audio_model_id: str = field(default_factory=lambda: _env(
        "AUDIO_MODEL_ID", "mo-thecreator/Deepfake-audio-detection"
    ))
    # Optional speech foundation encoder for complementary audio features.
    wavlm_model_id: str = field(default_factory=lambda: _env(
        "WAVLM_MODEL_ID", "microsoft/wavlm-large"
    ))
    # Video temporal backbone preference. "ftcn" | "videomae" | "timesformer"
    # or a concrete HF model id. Empty uses the existing temporal head + frame
    # aggregation.
    video_temporal_model_id: str = field(default_factory=lambda: _env(
        "VIDEO_TEMPORAL_MODEL_ID", ""
    ))
    hf_token: str = field(default_factory=lambda: _env("HF_TOKEN", ""))
    # Directory holding trained/fitted artifacts (probe, calibration, ...).
    artifact_dir: str = field(default_factory=lambda: _env("DEEPFAKE_ARTIFACT_DIR", "artifacts"))
    # "auto" | "cpu" | "cuda" | "mps"
    device: str = field(default_factory=lambda: _env("DEEPFAKE_DEVICE", "auto"))
    # Conformal risk level: the true label is inside the predicted set with
    # probability >= 1 - alpha (given exchangeable calibration data).
    alpha: float = field(default_factory=lambda: float(_env("DEEPFAKE_ALPHA", "0.1")))
    # Number of frames in the cheap Tier-1 video screen.
    tier1_frames: int = field(default_factory=lambda: int(_env("DEEPFAKE_TIER1_FRAMES", "12")))
    # Test-time-augmentation views for the image Tier-1 screen (1 = off).
    tta_views: int = field(default_factory=lambda: int(_env("DEEPFAKE_TTA_VIEWS", "3")))
    # --- EGAFR / recommended-architecture research modules ---
    # Run only the Tier-2 tests with the best expected value per cost
    # (Feature 1: Evidence-driven adaptive inference). Default ON for the
    # recommended stack.
    controller: bool = field(default_factory=lambda: _env("DEEPFAKE_CONTROLLER", "1") == "1")
    # Stop investigating once the estimated decision risk is below this.
    risk_target: float = field(default_factory=lambda: float(_env("DEEPFAKE_RISK_TARGET", "0.05")))
    # Compute budget for Tier-2 tests, in the controller's relative cost units.
    budget: float = field(default_factory=lambda: float(_env("DEEPFAKE_BUDGET", "1.5")))
    # Test-selection policy: voi | fixed | random | quality (voi is the proposal;
    # the others are the baselines it must beat).
    controller_policy: str = field(default_factory=lambda: _env("DEEPFAKE_CONTROLLER_POLICY", "voi"))
    # Ablation switch: treat experts as independent (rho = 0) in fusion and the controller.
    independent_experts: bool = field(default_factory=lambda: _env("DEEPFAKE_INDEPENDENT", "0") == "1")
    # Fuse with the correlation-aware GLS estimator instead of the weighted mean
    # (Feature 2: Correlation-aware forensic evidence fusion). Default ON.
    dependent_fusion: bool = field(default_factory=lambda: _env("DEEPFAKE_DEP_FUSION", "1") == "1")
    # Use context-conditional conformal sets / learned expert reliabilities
    # (needs conditional_<media>.json from train.fit_calibration --conditional).
    conditional: bool = field(default_factory=lambda: _env("DEEPFAKE_CONDITIONAL", "0") == "1")
    # Abstain when the verdict flips under plausible, mild transformations
    # (Feature 3: Counterfactual robustness fingerprint). Default ON.
    robustness_gate: bool = field(default_factory=lambda: _env("DEEPFAKE_ROBUSTNESS_GATE", "1") == "1")
    # Abstain when the evidence-sufficiency statistic is below this (0 = off).
    min_sufficiency: float = field(default_factory=lambda: float(_env("DEEPFAKE_MIN_SUFFICIENCY", "0")))
    # Maximum seconds of video used by the Tier-2 face-signal experts.
    face_signal_seconds: float = field(default_factory=lambda: float(_env("DEEPFAKE_FACE_SECONDS", "8")))
    # Enable fine-grained localization (heatmaps, timestamps, audio intervals)
    # and link findings to originating detectors (Feature 4).
    localization: bool = field(default_factory=lambda: _env("DEEPFAKE_LOCALIZATION", "1") == "1")
    # Enable generator-shift / OOD + provenance analysis (Feature 5). Always
    # runs the embedding OOD detector; this flag controls extra provenance
    # reporting.
    provenance_analysis: bool = field(default_factory=lambda: _env("DEEPFAKE_PROVENANCE", "1") == "1")



    # --- Media ingestion security & resource limits (P1) ---
    # Reject uploads larger than this (bytes). Default 100 MB.
    max_upload_bytes: int = field(default_factory=lambda: int(_env("DEEPFAKE_MAX_UPLOAD_BYTES", str(100 * 1024 * 1024))))
    # Max decoded image pixels (width * height). Default ~25 MP.
    max_image_pixels: int = field(default_factory=lambda: int(_env("DEEPFAKE_MAX_IMAGE_PIXELS", str(25_000_000))))
    # Max video duration in seconds. Default 120 s.
    max_video_seconds: float = field(default_factory=lambda: float(_env("DEEPFAKE_MAX_VIDEO_SECONDS", "120")))
    # Max audio duration in seconds. Default 300 s.
    max_audio_seconds: float = field(default_factory=lambda: float(_env("DEEPFAKE_MAX_AUDIO_SECONDS", "300")))
    # Soft wall-clock budget for a single analysis (seconds). Soft: recorded, not hard-kill.
    max_analysis_seconds: float = field(default_factory=lambda: float(_env("DEEPFAKE_MAX_ANALYSIS_SECONDS", "180")))
    # Max frames decoded from a video for any expert. Default 64.
    max_video_frames: int = field(default_factory=lambda: int(_env("DEEPFAKE_MAX_VIDEO_FRAMES", "64")))


def get_settings() -> Settings:
    return Settings()


def artifact_path(name: str) -> str:
    return os.path.join(get_settings().artifact_dir, name)
