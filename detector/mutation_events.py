"""Candidate mutation / transformation events between assets (CALM-FORGE).

Represents *hypotheses* about what changed between related files: resize,
re-encode, crop, temporal trim, audio mix, or unknown edit intervals.
These link lineage edges to forensic evidence intervals.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from detector.lineage_matcher import RelationshipScore
from detector.media_fingerprints import MediaFingerprint


@dataclass
class MutationEvent:
    event_id: str
    source_id: str
    target_id: str
    kind: str  # resize | recompress | crop | temporal_trim | audio_filter | unknown_edit
    confidence: float
    uncertainty: float
    parameters: Dict[str, Any] = field(default_factory=dict)
    suspicious_intervals: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _infer_kind(rel: RelationshipScore, fp_a: MediaFingerprint, fp_b: MediaFingerprint) -> str:
    if rel.relation == "resized":
        return "resize"
    if rel.relation == "temporal_subset":
        return "temporal_trim"
    if rel.relation in ("near_duplicate", "identical"):
        # same content, possible recompress if hist differs more than hashes
        sims = (rel.evidence or {}).get("similarities", {})
        if sims.get("hist", 1.0) < sims.get("ahash", 1.0) - 0.1:
            return "recompress"
        return "copy_or_reencode"
    if rel.relation == "likely_derivative":
        return "unknown_edit"
    if fp_a.media_type == "audio" or fp_b.media_type == "audio":
        return "audio_filter"
    return "unknown_edit"


def events_from_relationships(
    rels: List[RelationshipScore],
    fingerprints: Dict[str, MediaFingerprint],
) -> List[MutationEvent]:
    events: List[MutationEvent] = []
    for i, rel in enumerate(rels):
        if rel.relation in ("unrelated", "weak_similarity") and rel.confidence < 0.7:
            continue
        fp_a = fingerprints.get(rel.source_id)
        fp_b = fingerprints.get(rel.target_id)
        if not fp_a or not fp_b:
            continue
        kind = _infer_kind(rel, fp_a, fp_b)
        params: Dict[str, Any] = {}
        if fp_a.meta.get("size") and fp_b.meta.get("size"):
            params["size_from"] = fp_a.meta["size"]
            params["size_to"] = fp_b.meta["size"]
        if fp_a.duration_s is not None and fp_b.duration_s is not None:
            params["duration_from"] = fp_a.duration_s
            params["duration_to"] = fp_b.duration_s
        events.append(MutationEvent(
            event_id=f"mut_{i}_{rel.source_id[:6]}_{rel.target_id[:6]}",
            source_id=rel.source_id,
            target_id=rel.target_id,
            kind=kind,
            confidence=rel.confidence,
            uncertainty=rel.uncertainty,
            parameters=params,
            notes=f"Derived from relationship '{rel.relation}'",
        ))
    return events
