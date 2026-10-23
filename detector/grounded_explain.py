"""Evidence-constrained explanations.

Free-text explanations (from a VLM or a template) can sound confident while
saying things no evidence supports. This module turns the result into a list
of *claims*, each bound to the evidence that backs it, and filters any
free-text rationale down to the sentences it can ground. Unsupported
sentences are dropped, and the drop count is reported.

A sentence is supported when it mentions an expert or evidence name present in
the result and its stated direction (fake/real) does not contradict that
expert's sign. Numbers quoted in a sentence must match a claim within a
tolerance. This is a deliberately strict, deterministic check.
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence

_FAKE_WORDS = re.compile(r"\b(fake|synthetic|manipulated|generated|forged|tampered)\b", re.I)
_REAL_WORDS = re.compile(r"\b(real|authentic|genuine|natural|unaltered)\b", re.I)
_NUM = re.compile(r"(\d+(?:\.\d+)?)\s*(%?)")


def build_claims(result: Dict) -> List[Dict]:
    claims: List[Dict] = []
    for e in result.get("evidence", []) or []:
        if e.get("status") != "used" or e.get("logit") is None:
            continue
        z = float(e["logit"])
        claims.append({"id": f"ev:{e['name']}", "evidence": e["name"],
                       "direction": "fake" if z > 0 else "real",
                       "logit": round(z, 3),
                       "reliability": e.get("reliability"),
                       "text": f"{e['name']} points to {'fake' if z > 0 else 'real'} "
                               f"(log-odds {z:+.2f}, reliability "
                               f"{float(e.get('reliability') or 0):.2f})."})
    v = result.get("verdict")
    if v:
        claims.append({"id": "verdict", "evidence": "verdict", "direction": v.lower(),
                       "text": f"Calibrated verdict: {v}."})
    for c in (result.get("contradiction_map") or {}).get("intervals", []) or []:
        claims.append({"id": f"contra:{c.get('start')}", "evidence": "sync",
                       "direction": "fake",
                       "text": f"Mouth and audio disagree from {c.get('start')}s to {c.get('end')}s."})
    return claims


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def ground_text(text: str, result: Dict, tol: float = 0.06) -> Dict:
    claims = build_claims(result)
    names = {c["evidence"].lower(): c for c in claims if c["evidence"] != "verdict"}
    kept, dropped = [], []
    for s in _sentences(text):
        low = s.lower()
        hit = [c for n, c in names.items() if n in low]
        if not hit:
            dropped.append({"sentence": s, "reason": "mentions no recorded evidence"})
            continue
        says_fake, says_real = bool(_FAKE_WORDS.search(s)), bool(_REAL_WORDS.search(s))
        bad = [c for c in hit if (says_fake and not says_real and c["direction"] == "real")
               or (says_real and not says_fake and c["direction"] == "fake")]
        if bad:
            dropped.append({"sentence": s, "reason": f"contradicts {bad[0]['evidence']}"})
            continue
        pool = [abs(float(c["logit"])) for c in hit if "logit" in c]
        bad_num = False
        for m in _NUM.finditer(s):
            if m.group(2) == "%":
                continue
            v = float(m.group(1))
            if pool and not any(abs(v - p) <= max(tol, 0.05 * p) for p in pool):
                bad_num = True
        if bad_num:
            dropped.append({"sentence": s, "reason": "quoted number matches no evidence"})
            continue
        kept.append({"sentence": s, "supported_by": [c["id"] for c in hit]})
    return {"claims": claims, "kept": kept, "dropped": dropped,
            "grounded_fraction": (len(kept) / max(1, len(kept) + len(dropped)))}
