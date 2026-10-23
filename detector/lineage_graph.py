"""Lineage graph: assets as nodes, scored relationships as edges (CALM-FORGE).

Supports multiple plausible parents and an explicit unknown outcome rather than
forcing every file into a single tree.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set

from detector.lineage_matcher import RelationshipScore


@dataclass
class LineageNode:
    asset_id: str
    filename: Optional[str] = None
    media_type: Optional[str] = None
    sha256: Optional[str] = None
    role_hypothesis: str = "unknown"  # original | derivative | unrelated | unknown
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LineageEdge:
    source_id: str
    target_id: str
    relation: str
    confidence: float
    uncertainty: float
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LineageGraph:
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id
        self.nodes: Dict[str, LineageNode] = {}
        self.edges: List[LineageEdge] = []

    def add_node(self, node: LineageNode) -> None:
        self.nodes[node.asset_id] = node

    def add_relationship(self, rel: RelationshipScore) -> None:
        if rel.source_id not in self.nodes or rel.target_id not in self.nodes:
            return
        if rel.relation in ("unrelated",) and rel.confidence < 0.55:
            return
        self.edges.append(LineageEdge(
            source_id=rel.source_id,
            target_id=rel.target_id,
            relation=rel.relation,
            confidence=rel.confidence,
            uncertainty=rel.uncertainty,
            evidence=rel.evidence,
        ))

    def add_relationships(self, rels: List[RelationshipScore]) -> None:
        for r in rels:
            self.add_relationship(r)

    def infer_roles(self) -> None:
        """Heuristic role assignment: highest-resolution / longest / no parents → original."""
        children: Set[str] = set()
        parents: Set[str] = set()
        for e in self.edges:
            if e.relation in ("resized", "recompressed", "likely_derivative",
                              "temporal_subset", "near_duplicate") and e.confidence >= 0.75:
                # Prefer treating lower-resolution as derivative when sizes known
                a, b = self.nodes[e.source_id], self.nodes[e.target_id]
                sa = (a.attributes or {}).get("size") or (a.attributes or {}).get("width")
                sb = (b.attributes or {}).get("size") or (b.attributes or {}).get("width")
                # Without clear direction, mark both as related; don't force parent
                if e.relation == "temporal_subset":
                    # shorter video → child
                    da = (a.attributes or {}).get("duration_s")
                    db = (b.attributes or {}).get("duration_s")
                    if da is not None and db is not None:
                        if da < db:
                            children.add(e.source_id)
                            parents.add(e.target_id)
                        elif db < da:
                            children.add(e.target_id)
                            parents.add(e.source_id)
                elif e.relation in ("resized", "likely_derivative", "recompressed",
                                      "near_duplicate"):
                    # Prefer lower-resolution / smaller area as the derivative
                    def _area(sz):
                        if sz is None:
                            return None
                        if isinstance(sz, (list, tuple)) and len(sz) >= 2:
                            try:
                                return float(sz[0]) * float(sz[1])
                            except (TypeError, ValueError):
                                return None
                        try:
                            return float(sz)
                        except (TypeError, ValueError):
                            return None
                    aa, bb = _area(sa), _area(sb)
                    if aa is not None and bb is not None and aa != bb:
                        if aa > bb:
                            parents.add(e.source_id)
                            children.add(e.target_id)
                        else:
                            parents.add(e.target_id)
                            children.add(e.source_id)
                    else:
                        # No size evidence: do not invent direction from upload order
                        pass

        for aid, node in self.nodes.items():
            if aid in parents and aid not in children:
                node.role_hypothesis = "original_candidate"
            elif aid in children:
                node.role_hypothesis = "derivative_candidate"
            elif any(e.confidence >= 0.9 and e.relation == "identical"
                     for e in self.edges if aid in (e.source_id, e.target_id)):
                node.role_hypothesis = "duplicate"
            elif not self.edges:
                node.role_hypothesis = "unknown"
            else:
                # connected weakly or not at all
                connected = any(
                    aid in (e.source_id, e.target_id) and e.confidence >= 0.55
                    for e in self.edges
                )
                node.role_hypothesis = "related" if connected else "unknown"

    def components(self) -> List[Set[str]]:
        parent = {n: n for n in self.nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for e in self.edges:
            if e.confidence >= 0.55 and e.relation != "unrelated":
                union(e.source_id, e.target_id)
        groups: Dict[str, Set[str]] = {}
        for n in self.nodes:
            groups.setdefault(find(n), set()).add(n)
        return list(groups.values())

    def to_dict(self) -> Dict[str, Any]:
        self.infer_roles()
        return {
            "session_id": self.session_id,
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "components": [sorted(list(c)) for c in self.components()],
            "note": (
                "Roles are hypotheses with uncertainty, not proven provenance. "
                "Unknown is a valid outcome."
            ),
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
