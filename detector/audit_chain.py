"""Provenance-aware chain of custody.

A hash-linked log of everything that produced a verdict: the source file,
what is known (and not known) about its history, each analysis step, the
detector and calibration versions, and the decision. Every entry carries a
**provenance status** so claims of different strength are never mixed:

* ``verified``  - computed or checked by this system now (e.g. the file's
  SHA-256, the SHA-256 of the calibration file used);
* ``declared``  - asserted by the file or its metadata and NOT validated here
  (a C2PA manifest is present but its signature is not checked; an
  AI-generator tag in EXIF);
* ``inferred``  - deduced from signal evidence (JPEG quantisation suggests a
  recompression);
* ``unknown``   - nothing is known.

Entry hash = SHA-256 over (index, kind, status, payload, previous hash), so
editing, deleting, inserting or reordering any entry breaks every later
hash. The head hash goes into the signed report, which binds the whole
history to the report signature.

This complements, and does not replace, standards-based C2PA validation: it
records *that* a manifest exists, not that it is valid.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

STATUSES = ("verified", "declared", "inferred", "unknown")
GENESIS = "0" * 64


def _canon(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      default=str).encode("utf-8")


def _entry_hash(index: int, kind: str, status: str, payload: Dict[str, Any], prev: str) -> str:
    return hashlib.sha256(_canon({"i": index, "k": kind, "s": status,
                                  "p": payload, "prev": prev})).hexdigest()


def file_sha256(path: str) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class AuditChain:
    def __init__(self, entries: Optional[List[Dict[str, Any]]] = None):
        self.entries: List[Dict[str, Any]] = list(entries or [])

    @property
    def head(self) -> str:
        return self.entries[-1]["hash"] if self.entries else GENESIS

    def add(self, kind: str, status: str, /, **payload: Any) -> Dict[str, Any]:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        i = len(self.entries)
        # round-trip through JSON so the stored payload is exactly what is hashed
        payload = json.loads(_canon(payload))
        e = {"index": i, "kind": kind, "status": status, "payload": payload,
             "prev": self.head, "hash": _entry_hash(i, kind, status, payload, self.head)}
        self.entries.append(e)
        return e

    def verify(self) -> Dict[str, Any]:
        prev = GENESIS
        for i, e in enumerate(self.entries):
            if e.get("index") != i:
                return {"ok": False, "bad_index": i, "reason": "entry index out of order (inserted/removed/reordered)"}
            if e.get("prev") != prev:
                return {"ok": False, "bad_index": i, "reason": "broken link to previous entry"}
            if e.get("status") not in STATUSES:
                return {"ok": False, "bad_index": i, "reason": "invalid provenance status"}
            if _entry_hash(i, e["kind"], e["status"], e["payload"], prev) != e.get("hash"):
                return {"ok": False, "bad_index": i, "reason": "entry content does not match its hash"}
            prev = e["hash"]
        return {"ok": True, "bad_index": None, "reason": None, "length": len(self.entries),
                "head": prev}

    def summary(self) -> Dict[str, int]:
        out = {s: 0 for s in STATUSES}
        for e in self.entries:
            out[e["status"]] += 1
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"entries": self.entries, "head": self.head, "summary": self.summary()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AuditChain":
        return cls(d.get("entries", []))


def verify_chain_dict(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Verify a serialised chain, including that its stated head matches."""
    if not d or "entries" not in d:
        return {"ok": False, "reason": "no audit chain present", "bad_index": None}
    chain = AuditChain.from_dict(d)
    r = chain.verify()
    if r["ok"] and d.get("head") != chain.head:
        return {"ok": False, "bad_index": None, "reason": "stated head does not match the entries (tail truncated?)"}
    return r


def _pkg_versions() -> Dict[str, str]:
    from importlib import metadata
    out = {}
    for p in ("torch", "transformers", "numpy", "scikit-learn", "Pillow", "opencv-python"):
        try:
            out[p] = metadata.version(p)
        except Exception:
            pass
    return out


def build_chain(result: Dict[str, Any], artifact_dir: Optional[str] = None,
                source_size: Optional[int] = None) -> AuditChain:
    """Chain of custody for one ``AnalysisResult.to_dict()``."""
    c = AuditChain()
    c.add("source", "verified", filename=result.get("filename"),
          media_type=result.get("media_type"), sha256=result.get("sha256"),
          size_bytes=source_size)
    prov = result.get("provenance") or {}
    if prov.get("c2pa_manifest_detected"):
        c.add("content_credentials", "declared", manifest_present=True,
              signature_validated=False,
              note="C2PA manifest found; signature NOT validated by this system")
    else:
        c.add("content_credentials", "unknown", manifest_present=False)
    meta = result.get("metadata") or {}
    gen = meta.get("ai_generator_signature")
    if gen:
        c.add("metadata_claim", "declared", ai_generator_tag=gen,
              note="asserted by file metadata; metadata can be edited or stripped")
    q = result.get("quality") or {}
    if q.get("jpeg_quality") is not None:
        c.add("transformation_clue", "inferred", kind="jpeg_recompression",
              estimated_quality=q.get("jpeg_quality"),
              note="estimated from quantisation tables; original history unknown")
    elif not meta:
        c.add("transformation_clue", "unknown", note="no metadata or quantisation evidence")
    versions = _pkg_versions()
    for b in result.get("backends") or []:
        c.add("detector", "verified", name=b, packages=versions)
    if artifact_dir:
        files = {n: file_sha256(os.path.join(artifact_dir, n)) for n in
                 ("calibration.json", "conditional_calibration.json", "probe.npz",
                  "spectral_head.npz", "fusion_weights.json", "ood_image.npz",
                  "ood_video.npz", "ood_audio.npz", "temporal_head.pt")}
        c.add("calibration_version", "verified",
              files={k: v for k, v in files.items() if v},
              calibrated=bool(result.get("calibrated")))
    inv = result.get("investigation") or {}
    for s in inv.get("steps", []):
        c.add("expert_run", "verified", test=s.get("test"), why=s.get("reason"),
              utility=s.get("utility"), cost=s.get("cost"))
    for nr in inv.get("not_run", []):
        c.add("expert_not_run", "verified", test=nr.get("test"), why=nr.get("reason"))
    c.add("decision", "verified", verdict=result.get("verdict"),
          p_fake=result.get("p_fake"), label_set=result.get("label_set"),
          tier=result.get("tier"), mode=result.get("mode"),
          evidence=[{"name": e.get("name"), "status": e.get("status"),
                     "logit": e.get("logit"), "reliability": e.get("reliability")}
                    for e in result.get("evidence", [])],
          graph_fusion=(result.get("graph") or {}).get("fusion"))
    return c
