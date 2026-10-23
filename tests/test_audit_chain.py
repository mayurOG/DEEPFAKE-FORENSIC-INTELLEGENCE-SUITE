import copy
import json

import pytest

from detector.audit_chain import (AuditChain, GENESIS, build_chain, file_sha256,
                                  verify_chain_dict)


def make():
    c = AuditChain()
    c.add("source", "verified", sha256="ab" * 32)
    c.add("content_credentials", "declared", manifest_present=True)
    c.add("decision", "verified", verdict="Fake")
    return c


def test_chain_links_and_verifies():
    c = make()
    assert c.entries[0]["prev"] == GENESIS and c.entries[1]["prev"] == c.entries[0]["hash"]
    assert c.verify()["ok"] and c.summary() == {"verified": 2, "declared": 1, "inferred": 0, "unknown": 0}
    assert verify_chain_dict(json.loads(json.dumps(c.to_dict())))["ok"]


def test_every_kind_of_tampering_is_detected():
    d = make().to_dict()
    e = copy.deepcopy(d); e["entries"][1]["payload"]["manifest_present"] = False
    r = verify_chain_dict(e); assert not r["ok"] and r["bad_index"] == 1
    e = copy.deepcopy(d); e["entries"][0]["status"] = "unknown"
    assert not verify_chain_dict(e)["ok"]
    e = copy.deepcopy(d); del e["entries"][1]
    assert not verify_chain_dict(e)["ok"]                        # removed
    e = copy.deepcopy(d); e["entries"].insert(1, copy.deepcopy(e["entries"][0]))
    assert not verify_chain_dict(e)["ok"]                        # inserted
    e = copy.deepcopy(d); e["entries"][0], e["entries"][1] = e["entries"][1], e["entries"][0]
    assert not verify_chain_dict(e)["ok"]                        # reordered
    e = copy.deepcopy(d); e["entries"].pop()                     # truncated tail, head left alone
    r = verify_chain_dict(e); assert not r["ok"] and "head" in r["reason"]
    e = copy.deepcopy(d); e["entries"][2]["status"] = "bogus"
    assert not verify_chain_dict(e)["ok"]
    assert not verify_chain_dict(None)["ok"] and not verify_chain_dict({})["ok"]


def test_invalid_status_rejected():
    with pytest.raises(ValueError):
        AuditChain().add("x", "trusted")


def test_build_chain_separates_claim_strengths(tmp_path):
    (tmp_path / "calibration.json").write_text("{}")
    res = {"filename": "a.jpg", "media_type": "image", "sha256": "cd" * 32,
           "provenance": {"c2pa_manifest_detected": True},
           "metadata": {"ai_generator_signature": "Midjourney"},
           "quality": {"jpeg_quality": 61.0}, "backends": ["probe:x", "clf"],
           "calibrated": True, "verdict": "Fake", "p_fake": 0.97, "label_set": ["Fake"],
           "tier": "tier2", "mode": "balanced",
           "evidence": [{"name": "neural", "status": "used", "logit": 3.0, "reliability": 0.9}],
           "investigation": {"steps": [{"test": "tamper", "reason": "voi", "utility": 0.3, "cost": 0.4}],
                             "not_run": [{"test": "spectral", "reason": "no trained head"}]}}
    c = build_chain(res, str(tmp_path))
    kinds = [(e["kind"], e["status"]) for e in c.entries]
    assert ("content_credentials", "declared") in kinds       # present but NOT verified
    assert ("metadata_claim", "declared") in kinds
    assert ("transformation_clue", "inferred") in kinds
    assert ("calibration_version", "verified") in kinds
    assert ("expert_run", "verified") in kinds and ("expert_not_run", "verified") in kinds
    cc = [e for e in c.entries if e["kind"] == "content_credentials"][0]
    assert cc["payload"]["signature_validated"] is False
    cal = [e for e in c.entries if e["kind"] == "calibration_version"][0]
    assert cal["payload"]["files"]["calibration.json"] == file_sha256(str(tmp_path / "calibration.json"))
    assert c.verify()["ok"] and c.entries[-1]["kind"] == "decision"
    unknown = build_chain({"filename": "b", "provenance": {}, "metadata": None, "quality": {}})
    assert ("content_credentials", "unknown") in [(e["kind"], e["status"]) for e in unknown.entries]
