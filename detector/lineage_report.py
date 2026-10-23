"""Cross-asset evidence summaries and relationship explanations (CALM-FORGE)."""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from detector.asset_registry import AssetRegistry
from detector.lineage_graph import LineageGraph
from detector.mutation_events import MutationEvent
from detector.shortcut_audit import ShortcutAuditReport


def build_lineage_report(
    registry: AssetRegistry,
    graph: LineageGraph,
    mutations: Optional[List[MutationEvent]] = None,
    shortcut_audits: Optional[Dict[str, ShortcutAuditReport]] = None,
    per_file_verdicts: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Assemble a single JSON-serialisable cross-asset forensic report."""
    g = graph.to_dict()
    mutations = mutations or []
    shortcut_audits = shortcut_audits or {}
    per_file_verdicts = per_file_verdicts or {}

    # Narrative bullets (grounded in graph facts only)
    bullets: List[str] = []
    bullets.append(
        f"Session {registry.session_id}: {len(registry.list())} asset(s) registered."
    )
    comps = g.get("components") or []
    if len(comps) == 1 and len(registry.list()) > 1:
        bullets.append(
            "All assets fall in one connected component under current thresholds "
            "(possible shared origin or heavy near-duplication)."
        )
    elif len(comps) > 1:
        bullets.append(
            f"Found {len(comps)} disconnected component(s) — some files may be unrelated."
        )
    for node in g.get("nodes", []):
        role = node.get("role_hypothesis", "unknown")
        if role != "unknown":
            bullets.append(
                f"Asset {node.get('filename') or node['asset_id']}: role hypothesis = {role}."
            )
    for e in sorted(g.get("edges", []), key=lambda x: -x.get("confidence", 0))[:8]:
        bullets.append(
            f"Relationship {e['source_id'][:10]}… ↔ {e['target_id'][:10]}…: "
            f"{e['relation']} (confidence={e['confidence']:.2f}, "
            f"uncertainty={e['uncertainty']:.2f})."
        )
    for aid, audit in shortcut_audits.items():
        if audit.likely_shortcut_reliance:
            bullets.append(
                f"Shortcut audit on {aid}: high nuisance sensitivity "
                f"(fragility={audit.fragility:.2f}) — interpret with caution."
            )

    return {
        "report_type": "calm_forge_lineage",
        "version": "1.0",
        "generated_at": time.time(),
        "session_id": registry.session_id,
        "registry": registry.to_dict(),
        "lineage_graph": g,
        "mutation_events": [m.to_dict() for m in mutations],
        "shortcut_audits": {k: v.to_dict() for k, v in shortcut_audits.items()},
        "per_file_verdicts": per_file_verdicts,
        "narrative": bullets,
        "limitations": [
            "Relationship roles are hypotheses, not proven provenance.",
            "Unknown is a valid outcome when evidence is weak.",
            "Shortcut audits measure detector behaviour under benign transforms, "
            "not ground-truth manipulation.",
            "No accuracy claim is implied without held-out evaluation on real corpora.",
        ],
    }


def save_report(report: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)


def to_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# CALM-FORGE cross-asset lineage report",
        "",
        f"Session: `{report.get('session_id')}`",
        "",
        "## Narrative",
        "",
    ]
    for b in report.get("narrative", []):
        lines.append(f"- {b}")
    lines += ["", "## Assets", ""]
    for a in report.get("registry", {}).get("assets", []):
        lines.append(
            f"- `{a.get('asset_id')}` {a.get('filename')} "
            f"({a.get('media_type')}) sha256={str(a.get('sha256'))[:12]}…"
        )
    lines += ["", "## Relationships", ""]
    for e in report.get("lineage_graph", {}).get("edges", []):
        lines.append(
            f"- {e['relation']}: `{e['source_id']}` ↔ `{e['target_id']}` "
            f"(conf={e['confidence']:.2f}, unc={e['uncertainty']:.2f})"
        )
    lines += ["", "## Limitations", ""]
    for lim in report.get("limitations", []):
        lines.append(f"- {lim}")
    lines.append("")
    return "\n".join(lines)
