"""Score candidate relationships between media assets (CALM-FORGE).

Relationship hypotheses (not hard labels):

* ``identical`` — same content hash
* ``near_duplicate`` — very close perceptual hashes
* ``resized`` / ``recompressed`` — similar content, different geometry/quality
* ``temporal_subset`` — video segment / frame set contained in another
* ``audio_match`` — similar audio peaks
* ``unrelated`` — below threshold
* ``unknown`` — insufficient fingerprint evidence

Scores are in [0, 1] with an explicit uncertainty term.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from detector.media_fingerprints import MediaFingerprint, hamming


@dataclass
class RelationshipScore:
    source_id: str
    target_id: str
    relation: str
    confidence: float
    uncertainty: float
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _hist_sim(a: Optional[List[float]], b: Optional[List[float]]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    aa, bb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = (np.linalg.norm(aa) * np.linalg.norm(bb)) or 1.0
    return float(np.clip(np.dot(aa, bb) / denom, 0.0, 1.0))


def _hash_sim(a: Optional[str], b: Optional[str]) -> float:
    """Perceptual-hash similarity calibrated so random pairs score near 0.

    For independent 64-bit hashes the expected Hamming distance is ~32, so
    ``1 - d/64`` sits near 0.5 for unrelated media. That inflated the
    weak_similarity rate. We instead map distance relative to a tolerance
    band: perfect match → 1.0, ~16-bit disagreement → ~0, worse → 0.
    """
    if not a or not b:
        return 0.0
    d = hamming(a, b)
    # 16 bits of agreement beyond random (~32) is a soft near-duplicate band.
    return float(max(0.0, 1.0 - d / 16.0))


def score_pair(
    fp_a: MediaFingerprint,
    fp_b: MediaFingerprint,
    id_a: str,
    id_b: str,
    same_sha: bool = False,
) -> RelationshipScore:
    if same_sha:
        return RelationshipScore(
            id_a, id_b, "identical", 1.0, 0.0,
            evidence={"reason": "identical_sha256"},
        )

    if fp_a.media_type != fp_b.media_type and not (
        {fp_a.media_type, fp_b.media_type} <= {"image", "video"}
    ):
        return RelationshipScore(
            id_a, id_b, "unrelated", 0.05, 0.5,
            evidence={"reason": "media_type_mismatch",
                      "types": [fp_a.media_type, fp_b.media_type]},
        )

    # Image / video frame hashes
    sims = {
        "ahash": _hash_sim(fp_a.ahash, fp_b.ahash),
        "dhash": _hash_sim(fp_a.dhash, fp_b.dhash),
        "phash": _hash_sim(fp_a.phash, fp_b.phash),
        "hist": _hist_sim(fp_a.hist_rgb, fp_b.hist_rgb),
    }

    # Video: compare frame-hash sequences (simple average pairwise max)
    if fp_a.frame_hashes and fp_b.frame_hashes:
        fa, fb = fp_a.frame_hashes, fp_b.frame_hashes
        pair_sims = []
        for ha in fa:
            pair_sims.append(max(_hash_sim(ha, hb) for hb in fb))
        sims["frame_seq"] = float(np.mean(pair_sims)) if pair_sims else 0.0
        # temporal subset heuristic: shorter roughly matches a subset of longer
        if len(fa) != len(fb) and sims["frame_seq"] > 0.85:
            relation_hint = "temporal_subset"
        else:
            relation_hint = None
    else:
        relation_hint = None

    # Audio peaks
    if fp_a.audio_peaks and fp_b.audio_peaks:
        sims["audio"] = _hist_sim(fp_a.audio_peaks, fp_b.audio_peaks)

    # Aggregate
    weights = {
        "ahash": 0.25, "dhash": 0.2, "phash": 0.25, "hist": 0.15,
        "frame_seq": 0.3, "audio": 0.4,
    }
    num = den = 0.0
    for k, v in sims.items():
        w = weights.get(k, 0.1)
        num += w * v
        den += w
    score = num / den if den else 0.0

    # Geometry hints for resized / recompressed
    sa = fp_a.meta.get("size") if fp_a.meta else None
    sb = fp_b.meta.get("size") if fp_b.meta else None
    if sa and sb and sa != sb and score > 0.75:
        relation = "resized"
    elif relation_hint:
        relation = relation_hint
    elif score >= 0.90:
        relation = "near_duplicate"
    elif score >= 0.70:
        relation = "likely_derivative"
    elif score >= 0.40:
        relation = "weak_similarity"
    else:
        relation = "unrelated"

    # Uncertainty: higher when few modalities agree
    vals = list(sims.values())
    spread = float(np.std(vals)) if len(vals) > 1 else 0.3
    uncertainty = float(np.clip(0.15 + spread * 0.5 + (0.2 if len(vals) < 2 else 0), 0, 0.9))

    return RelationshipScore(
        id_a, id_b, relation, float(np.clip(score, 0, 1)), uncertainty,
        evidence={"similarities": {k: round(v, 4) for k, v in sims.items()}},
    )


def match_all(
    fingerprints: Dict[str, MediaFingerprint],
    sha_by_id: Optional[Dict[str, str]] = None,
    min_score: float = 0.40,
) -> List[RelationshipScore]:
    """All-pairs matching (O(n^2)); fine for session-scale asset sets."""
    ids = list(fingerprints.keys())
    sha_by_id = sha_by_id or {}
    out: List[RelationshipScore] = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            same = bool(sha_by_id.get(a) and sha_by_id.get(a) == sha_by_id.get(b))
            rel = score_pair(fingerprints[a], fingerprints[b], a, b, same_sha=same)
            if rel.relation != "unrelated" or rel.confidence >= min_score:
                out.append(rel)
    out.sort(key=lambda r: -r.confidence)
    return out
