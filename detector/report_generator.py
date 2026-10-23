"""Forensic evidence reports (HTML + JSON) with a tamper-evident signature.

The JSON report contains the evidence the verdict was based on, the file's
SHA-256, and an Ed25519 signature over the canonical payload, so a recipient
can check the report was not edited after it was produced
(``python -m scripts.verify_report report.json``).

The HTML template is rendered with autoescaping: filenames and EXIF fields
are attacker-controlled text and must never be injected as raw HTML (the
previous template did not escape them).
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, Optional, Tuple

from jinja2 import Environment

from detector.config import get_settings
from detector.fusion import _jsonable

SYSTEM_NAME = "Deepfake Forensic Intelligence Suite v3.0"

HTML_REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Forensic Report - {{ r.filename }}</title>
<style>
 body{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;padding:32px}
 .c{max-width:960px;margin:0 auto;background:#1e293b;border-radius:12px;padding:28px}
 h1{color:#38bdf8;margin:0 0 4px 0} h3{margin-top:26px;color:#94a3b8}
 .v{text-align:center;padding:18px;border-radius:10px;margin:18px 0;border:2px solid}
 .Fake{background:rgba(239,68,68,.15);border-color:#ef4444;color:#f87171}
 .Real{background:rgba(34,197,94,.15);border-color:#22c55e;color:#4ade80}
 .Inconclusive{background:rgba(234,179,8,.13);border-color:#eab308;color:#facc15}
 table{width:100%;border-collapse:collapse} th,td{padding:8px;text-align:left;border-bottom:1px solid #334155;font-size:14px}
 th{background:#0f172a;color:#94a3b8}
 .w{background:#7f1d1d;color:#fca5a5;padding:4px 8px;border-radius:4px;display:inline-block;margin:2px}
 .n{color:#94a3b8;font-size:13px} code{color:#7dd3fc;word-break:break-all}
 img{max-width:100%;border-radius:8px;margin:6px 0} pre{white-space:pre-wrap;color:#cbd5e1}
</style></head><body><div class="c">
<h1>Deepfake Forensic Report</h1>
<div class="n">{{ system }} | generated {{ timestamp }}</div>
<div class="n">File: <b>{{ r.filename }}</b> ({{ r.media_type }}) | SHA-256: <code>{{ r.sha256 }}</code></div>

<div class="v {{ r.verdict }}">
 <div style="font-size:26px;font-weight:700">Verdict: {{ r.verdict|upper }}</div>
 <div>Estimated chance of manipulation: {{ "%.1f"|format(r.p_fake*100) }}%
  {% if r.confidence is not none %} | Confidence in verdict: {{ "%.1f"|format(r.confidence*100) }}%{% endif %}</div>
 <div class="n">Label set: {{ r.label_set|join(", ") or "none" }} | mode: {{ r.mode }} | path: {{ r.tier }}
  | calibrated: {{ "yes (n=" ~ r.calibrator.n_cal ~ ", alpha=" ~ r.calibrator.alpha ~ ")" if r.calibrated else "NO" }}</div>
</div>

{% if r.warnings %}<div>{% for w in r.warnings %}<span class="w">{{ w }}</span>{% endfor %}</div>{% endif %}

<h3>Summary</h3><pre>{{ r.narrative }}</pre>

<h3>Evidence</h3>
<table><tr><th>Source</th><th>Status</th><th>Fake log-odds</th><th>Reliability</th><th>Notes</th></tr>
{% for e in r.evidence %}
<tr><td>{{ e.name }}{% if e.heuristic %} <span class="n">(heuristic)</span>{% endif %}</td>
<td>{{ e.status }}</td><td>{{ e.logit if e.logit is not none else "-" }}</td>
<td>{{ e.reliability }}</td><td class="n">{{ e.reason }}{% if e.detail %} {{ e.detail }}{% endif %}</td></tr>
{% endfor %}</table>

{% if r.stability %}<h3>Verdict stability</h3>
<div>{{ "%.0f"|format(r.stability.stability_score*100) }}% of degraded copies kept the same call
 (p range {{ "%.2f"|format(r.stability.p_min) }} - {{ "%.2f"|format(r.stability.p_max) }}).
 {% if r.stability.flips %}Flipped under: {{ r.stability.flips|join(", ") }}.{% endif %}</div>{% endif %}

{% if images %}<h3>Visual evidence</h3>
{% for title, b64 in images %}<div class="n">{{ title }}</div><img src="data:image/png;base64,{{ b64 }}">{% endfor %}{% endif %}

<h3>Media quality and provenance</h3>
<table>{% for k, v in r.quality.items() if v is not none and k != "notes" %}<tr><td>{{ k }}</td><td>{{ v }}</td></tr>{% endfor %}
<tr><td>provenance</td><td>{{ r.provenance.note if r.provenance else "n/a" }}</td></tr></table>
{% if r.metadata %}<h3>File metadata</h3><table>
<tr><td>Dimensions</td><td>{{ r.metadata.dimensions }}</td></tr>
<tr><td>Camera</td><td>{{ r.metadata.camera_make }} / {{ r.metadata.camera_model }}</td></tr>
<tr><td>Software</td><td>{{ r.metadata.software }}</td></tr>
<tr><td>Date</td><td>{{ r.metadata.date_taken }}</td></tr></table>
{% for w in r.metadata.warnings %}<span class="w">{{ w }}</span>{% endfor %}{% endif %}

<h3>Integrity</h3>
<div class="n">Report payload SHA-256: <code>{{ sig.payload_sha256 }}</code><br>
Signature ({{ sig.algorithm }}): <code>{{ sig.signature or "unsigned" }}</code><br>
Public key: <code>{{ sig.public_key or "n/a" }}</code></div>
<p class="n">This report supports a human review; it is not proof. Detectors can be wrong,
especially on generators they were not trained or calibrated on.</p>
</div></body></html>
"""


# --------------------------------------------------------------------------
# signing
# --------------------------------------------------------------------------
def _canonical(payload: Dict[str, Any]) -> bytes:
    return json.dumps(_jsonable(payload), sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _key_path() -> str:
    return os.path.join(get_settings().artifact_dir, "report_signing_key.pem")


def _load_or_create_key():
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except Exception:
        return None
    path = _key_path()
    if os.path.exists(path):
        with open(path, "rb") as f:
            return serialization.load_pem_private_key(f.read(), password=None)
    key = Ed25519PrivateKey.generate()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pem = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(pem)
    return key


def sign_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    body = _canonical(payload)
    digest = hashlib.sha256(body).hexdigest()
    key = _load_or_create_key()
    if key is None:
        return {"algorithm": "none", "payload_sha256": digest,
                "signature": None, "public_key": None}
    from cryptography.hazmat.primitives import serialization

    sig = key.sign(body)
    pub = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return {
        "algorithm": "Ed25519", "payload_sha256": digest,
        "signature": base64.b64encode(sig).decode(),
        "public_key": base64.b64encode(pub).decode(),
    }


def verify_report(
    report: Dict[str, Any],
    trusted_public_key_b64: Optional[str] = None,
) -> Tuple[bool, str]:
    """Check payload hash and signature of a JSON report dict.

    If ``trusted_public_key_b64`` is provided (or the env
    ``DEEPFAKE_REPORT_PUBKEY`` is set), the signature must verify against
    *that* key — not the key embedded in the report. Embedding the public
    key in the report only proves the report is self-consistent, not that
    it came from a trusted issuer.
    """
    import os
    try:
        payload, sig = report["payload"], report["signature"]
    except (KeyError, TypeError):
        return False, "Not a signed report (missing payload/signature)."
    body = _canonical(payload)
    if hashlib.sha256(body).hexdigest() != sig.get("payload_sha256"):
        return False, "Payload hash mismatch: the report was modified."
    if sig.get("algorithm") != "Ed25519" or not sig.get("signature"):
        return True, "Hash matches; report is unsigned (no signature to check)."
    pinned = trusted_public_key_b64 or os.getenv("DEEPFAKE_REPORT_PUBKEY") or ""
    pinned = pinned.strip()
    embedded = sig.get("public_key") or ""
    if pinned and embedded and pinned != embedded:
        return False, (
            "Report public key does not match the trusted key "
            "(set DEEPFAKE_REPORT_PUBKEY). Self-signed reports are not accepted."
        )
    key_b64 = pinned or embedded
    if not key_b64:
        return False, "No public key available to verify the signature."
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        pub = Ed25519PublicKey.from_public_bytes(base64.b64decode(key_b64))
        pub.verify(base64.b64decode(sig["signature"]), body)
    except InvalidSignature:
        return False, "Signature invalid: the report was modified or forged."
    except Exception as e:
        return False, f"Could not verify signature: {e}"
    chain = payload.get("audit")
    if chain:
        from detector.audit_chain import verify_chain_dict
        cr = verify_chain_dict(chain)
        if not cr["ok"]:
            return False, f"Audit chain invalid: {cr['reason']} (entry {cr['bad_index']})."
        return True, ("Signature valid and audit chain intact "
                      f"({len(chain['entries'])} entries): unchanged since signed.")
    return True, "Signature valid: the report is unchanged since it was signed."


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------
def build_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    """Select the report-worthy fields of ``AnalysisResult.to_dict()``."""
    keep = ("media_type", "filename", "sha256", "verdict", "label_set", "p_fake",
            "confidence", "calibrated", "calibrator", "ood", "ood_ratio", "tier",
            "mode", "evidence", "fusion", "quality", "provenance", "stability", "uncertainty",
            "graph", "investigation", "robustness", "abstention_reason", "attribution", "grounded", "audit",
            "narrative", "warnings", "backends", "timings")
    payload = {k: result.get(k) for k in keep}
    payload["metadata"] = _slim_metadata(result.get("metadata"))
    cm = result.get("contradiction_map")
    if cm:
        payload["contradiction_map"] = {k: cm.get(k) for k in (
            "status", "global_lag_s", "reference_corr", "benign_latency", "intervals",
            "coverage", "n_contradiction_windows")}
    expl = result.get("explanation") or {}
    payload["explanation"] = {k: expl.get(k) for k in ("method", "faithfulness") if k in expl}
    cf = expl.get("counterfactual") or {}
    if cf:
        payload["explanation"]["counterfactual"] = {
            k: cf.get(k) for k in ("applicable", "flip_fraction", "sufficient_fraction")}
    payload["system"] = SYSTEM_NAME
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    return _jsonable(payload)


def _slim_metadata(meta):
    if not meta:
        return None
    return {k: v for k, v in meta.items() if k != "exif_data"}


def generate_json_report(result: Dict[str, Any]) -> str:
    payload = build_payload(result)
    return json.dumps({"payload": payload, "signature": sign_payload(payload)},
                      indent=2, ensure_ascii=False)


def _b64_png(img) -> str:
    buf = BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def generate_html_report(
    result: Dict[str, Any], artifacts: Optional[Dict[str, Any]] = None,
    signed: Optional[Dict[str, Any]] = None,
) -> str:
    payload = signed["payload"] if signed else build_payload(result)
    sig = signed["signature"] if signed else sign_payload(payload)
    images = []
    for key, title in (("annotated", "Detected faces"),
                       ("heatmap", "Occlusion heatmap (regions driving the score)"),
                       ("tamper_map", "Noise-inconsistency map")):
        if artifacts and artifacts.get(key) is not None:
            try:
                images.append((title, _b64_png(artifacts[key])))
            except Exception:
                pass
    env = Environment(autoescape=True)
    tpl = env.from_string(HTML_REPORT_TEMPLATE)
    return tpl.render(
        r=_Obj(payload), system=SYSTEM_NAME, sig=_Obj(sig), images=images,
        timestamp=payload.get("generated_at", ""),
    )


class _Obj(dict):
    """dict with attribute access for the Jinja template."""

    def __getattr__(self, k):
        v = self.get(k)
        return _Obj(v) if isinstance(v, dict) else v
