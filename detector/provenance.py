"""Provenance lane: C2PA / Content Credentials and generator markers.

A cheap first lane. It reads the *bytes* for a C2PA (JUMBF) manifest and for
IPTC digital-source-type markers that say an asset was made by a generative
model. It does **not** validate the cryptographic signature chain: a present
manifest is reported as "present, unverified". Only an explicit
"trainedAlgorithmicMedia"-style declaration is treated as evidence; a missing
or stripped manifest is neutral (platforms strip them from genuine media).
"""
from __future__ import annotations

import re
from typing import Dict

AI_SOURCE_TYPES = (b"trainedalgorithmicmedia", b"compositewithtrainedalgorithmicmedia",
                   b"algorithmicmedia")
_CLAIM_GEN = re.compile(rb'claim_generator(?:_info)?["\x00\s:\[\{]{1,12}(?:name["\x00\s:]{1,6})?([A-Za-z0-9 ._/\-]{3,60})')
_MAX_SCAN = 8 * 1024 * 1024


def inspect_provenance(data: bytes) -> Dict:
    blob = bytes(data[:_MAX_SCAN])
    low = blob.lower()
    present = b"c2pa" in low and (b"jumb" in low or b"jumd" in low)
    ai = next((t.decode() for t in AI_SOURCE_TYPES if t in low), None)
    gen = None
    m = _CLAIM_GEN.search(blob)
    if m:
        gen = m.group(1).decode("utf-8", "ignore").strip()
    return {
        "c2pa_present": bool(present),
        "signature_validated": False,   # we never claim validation here
        "declared_ai_source": ai,
        "claim_generator": gen,
        "note": ("C2PA manifest present (signature not validated)." if present
                 else "No C2PA manifest (neutral: platforms strip them)."),
    }
