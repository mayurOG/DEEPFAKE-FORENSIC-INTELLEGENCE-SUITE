"""Asset registry for multi-file forensic sessions (CALM-FORGE).

Registers each uploaded or discovered media file with content hashes,
lightweight metadata and optional analysis IDs so lineage matching and
reports can refer to stable asset identities.

This does not replace single-file analysis; it is the intake layer for
cross-asset investigation.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from detector.triage import sha256_bytes, sha256_file


@dataclass
class AssetRecord:
    asset_id: str
    path: Optional[str] = None
    filename: Optional[str] = None
    media_type: Optional[str] = None  # image | video | audio
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    duration_s: Optional[float] = None
    sample_rate: Optional[int] = None
    registered_at: float = field(default_factory=time.time)
    analysis_ids: List[str] = field(default_factory=list)
    fingerprint_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AssetRegistry:
    """In-memory registry with optional JSON persistence."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self._assets: Dict[str, AssetRecord] = {}

    def register(
        self,
        source,
        media_type: Optional[str] = None,
        filename: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> AssetRecord:
        """Register a path, bytes, or open file-like object."""
        path = None
        raw: Optional[bytes] = None
        size = None
        if hasattr(source, "read"):
            filename = filename or getattr(source, "name", None)
            raw = source.read()
            size = len(raw)
        elif isinstance(source, (bytes, bytearray)):
            raw = bytes(source)
            size = len(raw)
        elif isinstance(source, str):
            path = source
            filename = filename or os.path.basename(path)
            if os.path.isfile(path):
                size = os.path.getsize(path)
                digest = sha256_file(path)
            else:
                raise FileNotFoundError(path)
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        if raw is not None:
            digest = sha256_bytes(raw)
        else:
            digest = sha256_file(path)  # type: ignore[arg-type]

        # Deduplicate by content hash within the session
        for existing in self._assets.values():
            if existing.sha256 == digest:
                if extra:
                    existing.metadata.update(extra)
                return existing

        asset_id = f"a_{digest[:12]}"
        rec = AssetRecord(
            asset_id=asset_id,
            path=path,
            filename=filename,
            media_type=media_type,
            sha256=digest,
            size_bytes=size,
            metadata=dict(extra or {}),
        )
        # Best-effort image geometry
        if media_type == "image" or (filename and filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp", ".bmp")
        )):
            try:
                from PIL import Image
                import io
                if raw is not None:
                    im = Image.open(io.BytesIO(raw))
                elif path:
                    im = Image.open(path)
                else:
                    im = None
                if im is not None:
                    rec.width, rec.height = im.size
                    if not rec.media_type:
                        rec.media_type = "image"
            except Exception:
                pass
        self._assets[asset_id] = rec
        return rec

    def get(self, asset_id: str) -> Optional[AssetRecord]:
        return self._assets.get(asset_id)

    def by_sha256(self, digest: str) -> Optional[AssetRecord]:
        for a in self._assets.values():
            if a.sha256 == digest:
                return a
        return None

    def list(self) -> List[AssetRecord]:
        return list(self._assets.values())

    def attach_analysis(self, asset_id: str, analysis_id: str) -> None:
        rec = self._assets.get(asset_id)
        if rec and analysis_id not in rec.analysis_ids:
            rec.analysis_ids.append(analysis_id)

    def set_fingerprint(self, asset_id: str, fingerprint_id: str) -> None:
        rec = self._assets.get(asset_id)
        if rec:
            rec.fingerprint_id = fingerprint_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "n_assets": len(self._assets),
            "assets": [a.to_dict() for a in self._assets.values()],
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "AssetRegistry":
        with open(path) as f:
            data = json.load(f)
        reg = cls(session_id=data.get("session_id"))
        for ad in data.get("assets", []):
            rec = AssetRecord(**{k: ad.get(k) for k in AssetRecord.__dataclass_fields__})
            reg._assets[rec.asset_id] = rec
        return reg
