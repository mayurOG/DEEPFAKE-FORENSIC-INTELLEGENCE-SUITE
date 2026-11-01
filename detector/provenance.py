"""Provenance lane: C2PA / Content Credentials and generator markers.

Reads and cryptographically verifies C2PA manifests using c2pa-python when available,
and audits IPTC digital-source-type markers that declare AI synthesis.

Categorization:
  * "valid": Manifest present, signature cryptographically valid, trusted certificate
  * "invalid": Manifest corrupted or signature check failed
  * "unverifiable": Manifest present but certificate/signature chain unverified
  * "absent": No manifest present (neutral: platforms frequently strip them)
"""
from __future__ import annotations

import io
import json
import logging
import re
from typing import Any, Dict, Optional

log = logging.getLogger("deepfake.provenance")

AI_SOURCE_TYPES = (
    b"trainedalgorithmicmedia",
    b"compositewithtrainedalgorithmicmedia",
    b"algorithmicmedia",
)
_CLAIM_GEN = re.compile(
    rb'claim_generator(?:_info)?["\x00\s:\[\{]{1,12}(?:name["\x00\s:]{1,6})?([A-Za-z0-9 ._/\-]{3,60})'
)
_MAX_SCAN = 8 * 1024 * 1024


def inspect_provenance(data: bytes) -> Dict[str, Any]:
    """Inspect and cryptographically verify C2PA provenance and AI declarations."""
    blob = bytes(data[:_MAX_SCAN])
    low = blob.lower()
    raw_present = b"c2pa" in low and (b"jumb" in low or b"jumd" in low)

    ai = next((t.decode() for t in AI_SOURCE_TYPES if t in low), None)
    gen = None
    m = _CLAIM_GEN.search(blob)
    if m:
        gen = m.group(1).decode("utf-8", "ignore").strip()

    c2pa_present = bool(raw_present)
    signature_validated = False
    validation_state = "absent" if not raw_present else "unverifiable"
    validation_details: Optional[Dict[str, Any]] = None
    certificate_info: Optional[Dict[str, Any]] = None

    # Cryptographic verification via c2pa-python
    try:
        import c2pa
        reader = c2pa.Reader.try_create(stream=io.BytesIO(blob))
        if reader is not None:
            c2pa_present = True
            state = reader.get_validation_state()
            validation_state = str(state) if state is not None else "unverifiable"
            results = reader.get_validation_results()
            validation_details = results if isinstance(results, dict) else None

            # Check if signature passed cryptographic verification
            state_low = validation_state.lower()
            if "valid" in state_low and "invalid" not in state_low:
                signature_validated = True
                validation_state = "valid"
            elif "invalid" in state_low or "failure" in state_low:
                signature_validated = False
                validation_state = "invalid"

            # Parse active manifest for generator info
            try:
                manifest_json = json.loads(reader.json())
                active_manifest = manifest_json.get("active_manifest") or {}
                if not gen and "claim_generator" in active_manifest:
                    gen = active_manifest["claim_generator"]
                cert = active_manifest.get("signature_info", {})
                if cert:
                    certificate_info = {
                        "issuer": cert.get("issuer"),
                        "cert_serial_number": cert.get("cert_serial_number"),
                        "time": cert.get("time"),
                    }
            except Exception:
                pass
    except Exception as e:
        log.debug("C2PA cryptographic reader check skipped or failed: %s", e)

    if signature_validated:
        note = "Cryptographically verified C2PA manifest (signature and certificate valid)."
    elif c2pa_present:
        note = f"C2PA manifest present (status: {validation_state}). Presence alone does not prove authenticity."
    else:
        note = "No C2PA manifest (neutral: platforms strip them from genuine media)."

    return {
        "c2pa_present": c2pa_present,
        "signature_validated": signature_validated,
        "validation_state": validation_state,
        "declared_ai_source": ai,
        "claim_generator": gen,
        "certificate_info": certificate_info,
        "validation_details": validation_details,
        "note": note,
    }
