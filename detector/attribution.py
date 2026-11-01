"""Generator attribution classifier — v4.0 (NEW).

Identifies WHICH type of generator created the suspect media.
This is a novel contribution: most detectors only say "fake" — this one says
*how* it was faked.

Generator families covered:
    GAN         — GAN-based face swap / image synthesis (StyleGAN, VQGAN, etc.)
    Diffusion   — Diffusion model output (Stable Diffusion, DALL-E, Flux, Midjourney)
    VAE         — VAE / VQ-VAE decoder (older generators)
    Face_Swap   — Neural face swap (SimSwap, FaceSwap, DeepFaceLab)
    Inpaint     — Inpainting / partial edit (diffusion inpaint, DALL-E edit)
    TTS         — Text-to-speech clone (audio; ElevenLabs, Bark, XTTS)
    Voice_Conv  — Voice conversion / RVC / So-VITS-SVC
    Unknown     — Cannot confidently attribute

Attribution is based on the SPECTRAL FINGERPRINT of the generator:
  * GANs leave characteristic checkerboard patterns (transpose conv).
  * Diffusion models leave smooth spectral residuals but with characteristic
    UNet decoder frequency roll-offs.
  * VAE decoders have characteristic block-structure in the NPR features.
  * For audio: codec artifacts, F0 smoothness, formant transitions differ.

The attribution head is a softmax-output logistic regression trained on the
combined spectral+NPR feature vectors from known-source datasets.

Training:
    python -m train.train_attribution  (image)
    python -m train.train_attribution --media audio  (audio)

Important:
    This output is SUPPORTING CONTEXT ONLY.
    It never changes the verdict (Real / Fake / Inconclusive).
    It is displayed as an informational annotation.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

log = logging.getLogger("deepfake.attribution")

# Generator family labels (image).
IMAGE_FAMILIES: Tuple[str, ...] = (
    "GAN", "Diffusion", "VAE", "Face_Swap", "Inpaint", "Unknown"
)

# Generator family labels (audio).
AUDIO_FAMILIES: Tuple[str, ...] = (
    "TTS", "Voice_Conv", "GAN_Voice", "Unknown"
)

AUDIO_FAMILY_DESCRIPTIONS: Dict[str, str] = {
    "TTS": "Text-to-speech synthesis (ElevenLabs / Bark / XTTS / Tortoise)",
    "Voice_Conv": "Voice conversion / RVC / So-VITS-SVC",
    "GAN_Voice": "GAN-based speech synthesis (WaveGAN / MelGAN variants)",
    "Unknown": "Generator type could not be identified",
}

IMAGE_FAMILY_DESCRIPTIONS: Dict[str, str] = {
    "GAN": "GAN-based synthesis or face swap (StyleGAN / VQGAN / SimSwap)",
    "Diffusion": "Diffusion model output (Stable Diffusion / DALL-E / Flux / Midjourney)",
    "VAE": "VAE / VQ-VAE decoder (older generator architecture)",
    "Face_Swap": "Neural face swap (DeepFaceLab / FaceSwap / Ghost)",
    "Inpaint": "Partial edit / inpainting (diffusion inpaint / DALL-E edit)",
    "Unknown": "Generator type could not be identified",
}


class AttributionHead:
    """Softmax logistic head for multi-class generator attribution.

    Architecture:
        Input: combined spectral+NPR feature vector (FEATURE_DIM = 93 dims).
        Output: softmax probabilities over generator families.

    Saved as .npz with keys:
        W: (n_classes, n_features) weight matrix
        b: (n_classes,) bias
        mean, scale: normalisation vectors
        families: list of family names (as bytes array)
        media: "image" or "audio"
    """

    def __init__(self, W, b, mean, scale, families, media="image"):
        self.W = np.asarray(W, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.families: List[str] = list(families)
        self.media = media

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Softmax probabilities over generator families."""
        x = (np.asarray(features, dtype=np.float64) - self.mean) / np.maximum(self.scale, 1e-6)
        logits = self.W @ x + self.b
        # Numerically stable softmax.
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum() + 1e-12
        return probs

    def top(self, probs: np.ndarray) -> Tuple[str, float]:
        idx = int(np.argmax(probs))
        return self.families[idx], float(probs[idx])

def fit_attributor(
    X: np.ndarray, families: Sequence[str],
    unknown_below: float = 0.5, C: float = 1.0,
) -> Dict:
    """Train open-set generator-family attributor on feature vectors."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=np.float64)
    fam = np.asarray(list(families))
    classes = sorted(set(fam.tolist()))
    if len(classes) < 2:
        raise ValueError("need at least two families")
    sc = StandardScaler().fit(X)
    Z = sc.transform(X)
    clf = LogisticRegression(C=C, max_iter=2000).fit(Z, fam)
    cent = np.stack([Z[fam == c].mean(0) for c in classes])
    d = np.array([np.linalg.norm(Z[i] - cent[classes.index(fam[i])]) for i in range(len(Z))])
    return {
        "classes": classes, "mean": sc.mean_, "scale": sc.scale_,
        "coef": clf.coef_, "intercept": clf.intercept_, "centroids": cent,
        "dist_limit": float(np.quantile(d, 0.99) * 1.25),
        "unknown_below": float(unknown_below),
    }


def predict_family(head: Any, feats: np.ndarray) -> Dict:
    """Predict generator family with distance/probability rejection for unknown."""
    if isinstance(head, dict):
        z = (np.asarray(feats, dtype=np.float64) - head["mean"]) / head["scale"]
        logits = head["coef"] @ z + head["intercept"]
        if len(head["classes"]) == 2:
            logits = np.array([-logits[0], logits[0]]) if logits.shape[0] == 1 else logits
        e = np.exp(logits - logits.max())
        p = e / (e.sum() + 1e-12)
        dist = float(np.min(np.linalg.norm(head["centroids"] - z, axis=1)))
        i = int(np.argmax(p))
        unknown = bool(p[i] < head["unknown_below"] or dist > head["dist_limit"])
        top_fam = head["classes"][i]
        return {
            "family": "unknown" if unknown else top_fam,
            "top": top_fam,
            "probability": float(p[i]),
            "distance": dist,
            "scores": {c: float(v) for c, v in zip(head["classes"], p)},
        }
    elif hasattr(head, "predict_proba"):
        probs = head.predict_proba(feats)
        family, prob = head.top(probs)
        return {
            "family": family,
            "top": family,
            "probability": float(prob),
            "scores": {f: float(p) for f, p in zip(head.families, probs)},
        }
    return {"family": "unknown", "top": "unknown", "probability": 0.0, "scores": {}}


def fit_attribution_head(
    X: np.ndarray, y: np.ndarray, families: List[str], media: str = "image",
    C: float = 1.0, seed: int = 0,
) -> "AttributionHead":
    """Train a multi-class logistic regression attribution head.

    X: (n, features) — combined spectral+NPR vectors.
    y: (n,) integer class labels aligned with families.
    """
    from sklearn.linear_model import LogisticRegression

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    mean, scale = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(
        C=C, class_weight="balanced", max_iter=5000,
        multi_class="multinomial", solver="lbfgs",
        random_state=seed,
    )
    clf.fit((X - mean) / scale, y)
    return AttributionHead(clf.coef_, clf.intercept_, mean, scale, families, media)


def save_attributor(head: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if isinstance(head, dict):
        np.savez(
            path, classes=np.array(head["classes"]), mean=head["mean"],
            scale=head["scale"], coef=head["coef"], intercept=head["intercept"],
            centroids=head["centroids"], dist_limit=head["dist_limit"],
            unknown_below=head["unknown_below"],
        )
    elif isinstance(head, AttributionHead):
        np.savez(
            path,
            W=head.W, b=head.b, mean=head.mean, scale=head.scale,
            families=np.array([f.encode() for f in head.families]),
            media=np.array(head.media.encode()),
        )


def load_attributor(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    try:
        d = np.load(path, allow_pickle=False)
        if "classes" in d:
            return {
                "classes": [str(c) for c in d["classes"]],
                "mean": d["mean"], "scale": d["scale"],
                "coef": d["coef"], "intercept": d["intercept"],
                "centroids": d["centroids"],
                "dist_limit": float(d["dist_limit"]),
                "unknown_below": float(d["unknown_below"]),
            }
        families = [f.decode() if isinstance(f, bytes) else str(f) for f in d["families"]]
        media_raw = d["media"]
        media = media_raw.item().decode() if isinstance(media_raw.item(), bytes) else str(media_raw.item())
        return AttributionHead(d["W"], d["b"], d["mean"], d["scale"], families, media)
    except Exception as e:
        log.warning("Could not load attribution head at %s: %s", path, e)
        return None


def attribute_image(
    img: Image.Image, head: Optional[Any],
) -> Optional[Dict]:
    """Return attribution dict or None if unavailable.

    Dict keys:
        family     — top-1 generator family name
        top        — same as family
        probability — confidence in [0, 1]
        all_probs  — dict of {family: probability}
        description — human-readable description
        supporting_only — always True (never changes verdict)
    """
    if head is None:
        return None
    if min(img.size) < 96:
        return None
    if isinstance(head, dict):
        from detector.experts.spectral import spectral_features
        f = spectral_features(img)
        if not np.any(f):
            return None
        res = predict_family(head, f)
        res["supporting_only"] = True
        return res
    try:
        from detector.experts.spectral import combined_features
        feats = combined_features(img)
    except Exception as e:
        log.warning("Attribution feature extraction failed: %s", e)
        return None

    if len(feats) != head.W.shape[1]:
        # Feature dim mismatch (old vs new head).
        log.debug(
            "Attribution head feature dim mismatch: %d vs %d",
            head.W.shape[1], len(feats),
        )
        return None

    probs = head.predict_proba(feats)
    family, prob = head.top(probs)
    descriptions = IMAGE_FAMILY_DESCRIPTIONS if head.media == "image" else AUDIO_FAMILY_DESCRIPTIONS

    return {
        "family": family,
        "top": family,
        "probability": round(prob, 3),
        "all_probs": {f: round(float(p), 3) for f, p in zip(head.families, probs)},
        "description": descriptions.get(family, "Unknown generator type"),
        "supporting_only": True,  # NEVER changes the verdict
    }


def attribute_audio(
    features: np.ndarray, head: Optional["AttributionHead"],
) -> Optional[Dict]:
    """Attribution for audio features (LFCC or WavLM embeddings).

    Same interface as attribute_image but accepts pre-computed feature vectors.
    """
    if head is None or features is None:
        return None
    feats = np.asarray(features, dtype=np.float64).flatten()
    if len(feats) != head.W.shape[1]:
        return None
    probs = head.predict_proba(feats)
    family, prob = head.top(probs)
    descriptions = AUDIO_FAMILY_DESCRIPTIONS

    return {
        "family": family,
        "probability": round(prob, 3),
        "all_probs": {f: round(float(p), 3) for f, p in zip(head.families, probs)},
        "description": descriptions.get(family, "Unknown generator type"),
        "supporting_only": True,
    }
