"""Multi-asset investigation orchestration (CALM-FORGE).

Preserves the existing single-file Pipeline; this module adds an optional
session mode that registers assets, fingerprints them, builds a lineage graph,
runs optional shortcut audits and produces a cross-asset report.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence

from detector.asset_registry import AssetRegistry
from detector.lineage_graph import LineageGraph, LineageNode
from detector.lineage_matcher import match_all
from detector.lineage_report import build_lineage_report, save_report, to_markdown
from detector.media_fingerprints import MediaFingerprint, fingerprint_asset
from detector.mutation_events import events_from_relationships
from detector.pipeline import infer_media_type
from detector.shortcut_audit import ShortcutAuditReport, audit_with_pipeline

log = logging.getLogger("deepfake.cross_asset")


class CrossAssetSession:
    def __init__(self, session_id: Optional[str] = None, run_forensics: bool = True):
        self.registry = AssetRegistry(session_id=session_id)
        self.fingerprints: Dict[str, MediaFingerprint] = {}
        self.run_forensics = run_forensics
        self._verdicts: Dict[str, Dict[str, Any]] = {}
        self._shortcut: Dict[str, ShortcutAuditReport] = {}
        self._pipe = None

    def _pipeline(self):
        if self._pipe is None:
            from detector.pipeline import get_pipeline
            self._pipe = get_pipeline()
        return self._pipe

    def add_file(self, path: str, media_type: Optional[str] = None) -> str:
        mt = media_type or infer_media_type(path)
        rec = self.registry.register(path, media_type=mt, filename=os.path.basename(path))
        fp = fingerprint_asset(path, media_type=mt)
        self.fingerprints[rec.asset_id] = fp
        self.registry.set_fingerprint(rec.asset_id, fp.fingerprint_id)
        # Enrich node attributes for role inference
        if fp.meta.get("size"):
            rec.metadata["size"] = fp.meta["size"]
        if fp.duration_s is not None:
            rec.metadata["duration_s"] = fp.duration_s
        if rec.width:
            rec.metadata["width"] = rec.width

        if self.run_forensics and os.path.isfile(path):
            try:
                res = self._pipeline().analyze(
                    path, media_type=mt, explain=False, stability=False,
                )
                self._verdicts[rec.asset_id] = {
                    "verdict": res.verdict,
                    "p_fake": res.p_fake,
                    "confidence": res.confidence,
                    "tier": res.tier,
                }
                self.registry.attach_analysis(rec.asset_id, f"analysis_{rec.asset_id}")
            except Exception as e:
                log.warning("forensics failed for %s: %s", path, e)
                self._verdicts[rec.asset_id] = {"error": str(e)}
        return rec.asset_id

    def add_files(self, paths: Sequence[str]) -> List[str]:
        return [self.add_file(p) for p in paths]

    def build_graph(self) -> LineageGraph:
        sha = {a.asset_id: a.sha256 for a in self.registry.list() if a.sha256}
        rels = match_all(self.fingerprints, sha_by_id=sha)
        graph = LineageGraph(session_id=self.registry.session_id)
        for a in self.registry.list():
            graph.add_node(LineageNode(
                asset_id=a.asset_id,
                filename=a.filename,
                media_type=a.media_type,
                sha256=a.sha256,
                attributes=dict(a.metadata),
            ))
        graph.add_relationships(rels)
        graph.infer_roles()
        self._rels = rels
        self._graph = graph
        return graph

    def run_shortcut_audits(self, max_assets: int = 5) -> Dict[str, ShortcutAuditReport]:
        pipe = self._pipeline()
        n = 0
        for a in self.registry.list():
            if a.media_type != "image" or not a.path:
                continue
            if n >= max_assets:
                break
            try:
                self._shortcut[a.asset_id] = audit_with_pipeline(
                    pipe, a.path, asset_id=a.asset_id,
                )
                n += 1
            except Exception as e:
                log.warning("shortcut audit failed for %s: %s", a.asset_id, e)
        return self._shortcut

    def investigate(
        self,
        paths: Sequence[str],
        shortcut_audit: bool = True,
    ) -> Dict[str, Any]:
        self.add_files(paths)
        graph = self.build_graph()
        mutations = events_from_relationships(
            getattr(self, "_rels", []), self.fingerprints,
        )
        if shortcut_audit:
            self.run_shortcut_audits()
        report = build_lineage_report(
            self.registry,
            graph,
            mutations=mutations,
            shortcut_audits=self._shortcut,
            per_file_verdicts=self._verdicts,
        )
        return report
