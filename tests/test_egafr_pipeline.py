import dataclasses
import json

import numpy as np
import pytest

from detector import models as M
from detector import report_generator as RG
from detector.audit_chain import verify_chain_dict
from detector.pipeline import Pipeline
from tests.conftest import make_img, stub_image_fn


def mk(settings, **flags):
    return Pipeline(settings=dataclasses.replace(settings, **flags))


def run(p, img=None, **kw):
    img = img or make_img(10, 0, -2)   # ambiguous for the stub
    kw.setdefault("explain", False)
    kw.setdefault("stability", False)
    return p.analyze(img, media_type="image", filename="x.png", **kw)


def test_default_outputs_graph_and_valid_audit(pipe):
    r = pipe.analyze(make_img(60, 0, -60), media_type="image", filename="f.png")
    assert r.graph["nodes"] and r.graph["fusion"]["n_used"] >= 1
    assert verify_chain_dict(r.audit)["ok"]
    kinds = [e["kind"] for e in r.audit["entries"]]
    assert kinds[0] == "source" and kinds[-1] == "decision" and "calibration_version" in kinds
    assert r.audit["entries"][0]["payload"]["sha256"] == r.sha256
    assert r.robustness and r.robustness["rows"] and "gate" in r.robustness
    assert "Evidence sufficiency" in r.narrative and "Transformation robustness" in r.narrative
    assert r.investigation == {} and r.fusion.get("method", "weighted_mean") == "weighted_mean"


def test_flags_off_is_legacy_and_flags_on_keeps_the_verdict(settings):
    """Flags off = the legacy weighted mean. Turning dependency-aware fusion
    on may move the probability (GLS is a different estimator, on by default
    since v5) but must not flip the verdict or its sign on clear inputs."""
    M.set_backends(image=M.StubBackend(stub_image_fn), face=M.StubFaceDetector())
    on = mk(settings, controller=False, dependent_fusion=True)
    off = mk(settings, controller=False, dependent_fusion=False)
    for img in (make_img(60, 0, -60), make_img(-60, 0, 60)):
        a = run(on, img, force_full=True)
        b = run(off, img, force_full=True)
        assert b.fusion["method"] == "weighted_mean"
        assert a.fusion["method"] == "dependent_gls"
        assert a.verdict == b.verdict
        assert (a.p_fake - 0.5) * (b.p_fake - 0.5) > 0
    # legacy path is deterministic
    img = make_img(10, 0, -2)
    assert round(run(off, img, force_full=True).p_fake, 9) == round(run(off, img, force_full=True).p_fake, 9)
    M.reset_backends()


def test_controller_records_why_and_skips_unneeded_tests(settings):
    M.set_backends(image=M.StubBackend(stub_image_fn), face=M.StubFaceDetector())
    # loose target + ambiguous image -> investigate; tight budget stops it
    fixed = run(mk(settings), force_full=False)
    ctrl = run(mk(settings, controller=True, risk_target=1e-9, budget=0.2))
    if ctrl.tier == "tier2":
        inv = ctrl.investigation
        assert inv["policy"] == "voi" and inv["spent"] <= 0.2 + 1e-9
        assert inv["stop_reason"] in ("no_eligible_test", "no_test_worth_its_cost", "all_tests_run")
        ran = {s["test"] for s in inv["steps"]}
        names = {e["name"]: e for e in ctrl.evidence}
        for nr in inv["not_run"]:
            assert names[nr["test"]]["status"] == "skipped" and nr["reason"]
        assert ran.isdisjoint({nr["test"] for nr in inv["not_run"]})
        kinds = [e["kind"] for e in ctrl.audit["entries"]]
        assert "expert_run" in kinds or "expert_not_run" in kinds
        assert "Investigation:" in ctrl.narrative
    # an already-certain image exits before the controller is needed
    sure = run(mk(settings, controller=True), make_img(80, 0, -80))
    assert sure.tier == "tier1_early_exit" and sure.investigation == {}
    assert fixed.verdict in ("Real", "Fake", "Inconclusive")
    M.reset_backends()


def test_force_full_bypasses_controller_for_calibration_fitting(settings):
    M.set_backends(image=M.StubBackend(stub_image_fn), face=M.StubFaceDetector())
    r = run(mk(settings, controller=True, risk_target=0.9), force_full=True)
    assert r.investigation == {}
    assert {"tamper", "forensic_prior"} <= {e["name"] for e in r.evidence}
    assert "expert_logits" in r.raw and "ctx" in r.raw
    M.reset_backends()


def test_dependent_fusion_flag(settings):
    M.set_backends(image=M.StubBackend(stub_image_fn), face=M.StubFaceDetector())
    r = run(mk(settings, dependent_fusion=True), force_full=True)
    assert r.fusion["method"] == "dependent_gls" and "sufficiency" in r.fusion
    M.reset_backends()


def test_sufficiency_gate_abstains(settings):
    M.set_backends(image=M.StubBackend(stub_image_fn), face=M.StubFaceDetector())
    r = run(mk(settings, min_sufficiency=1.01), make_img(80, 0, -80))
    assert r.verdict == "Inconclusive" and r.abstention_reason.startswith("insufficient_evidence")
    assert "Abstention reason" in r.narrative
    assert r.audit["entries"][-1]["payload"]["verdict"] == "Inconclusive"
    M.reset_backends()


def test_robustness_gate_downgrades_fragile_calls(settings):
    def fragile(im):
        a = np.asarray(im.convert("L"), dtype=np.float64)
        hf = np.abs(np.diff(a, axis=1)).mean()
        z = 4.0 if hf > 8.0 else -4.0                            # lives in high-frequency detail
        return z, np.array([z])
    M.set_backends(image=M.StubBackend(fragile), face=M.StubFaceDetector())
    rng = np.random.default_rng(1)
    from PIL import Image
    img = Image.fromarray(rng.integers(100, 170, (160, 160, 3), dtype=np.uint8))
    on = mk(settings, robustness_gate=True).analyze(img, media_type="image", explain=False, stability=True)
    off = mk(settings, robustness_gate=False).analyze(img, media_type="image", explain=False, stability=True)
    assert off.verdict == "Fake"                      # confident without the gate ...
    assert on.robustness["fragile_to"] and not on.robustness["gate"]["ok"]
    assert on.verdict == "Inconclusive" and on.abstention_reason.startswith("unstable")  # ... gated with it
    M.reset_backends()


def test_signed_report_binds_audit_chain(pipe):
    r = pipe.analyze(make_img(60, 0, -60), media_type="image", filename="f.png")
    rep = json.loads(RG.generate_json_report(r.to_dict()))
    ok, msg = RG.verify_report(rep)
    assert ok and "audit chain intact" in msg
    assert rep["payload"]["audit"]["head"] == r.audit["head"]
    # editing a history entry breaks the signature; re-signing cannot hide a broken chain
    bad = json.loads(json.dumps(rep))
    bad["payload"]["audit"]["entries"][0]["payload"]["filename"] = "other.png"
    assert not RG.verify_report(bad)[0]
    forged = json.loads(json.dumps(rep))
    forged["payload"]["audit"]["entries"][0]["payload"]["filename"] = "other.png"
    forged["signature"] = RG.sign_payload(forged["payload"])
    ok2, msg2 = RG.verify_report(forged)
    assert not ok2 and "Audit chain invalid" in msg2


def test_conditional_flag_uses_group_thresholds(settings, tmp_path):
    from detector.conditional_reliability import fit_conditional, save_conditional, group_key
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 800)
    z = (2 * y - 1) * 2.0 + rng.normal(0, 1.2, 800)
    ctxs = [{"compression": "unknown", "n_used": 3, "min_side": 256}] * 800
    cc = fit_conditional(z, y, ctxs, min_group_n=50)
    save_conditional(f"{settings.artifact_dir}/conditional_image.json", cc, {})
    M.set_backends(image=M.StubBackend(stub_image_fn), face=M.StubFaceDetector())
    p = mk(settings, conditional=True)
    assert "image" in p.conditional and p.status()["conditional_calibration"] == ["image"]
    r = run(p, force_full=True)
    assert r.calibrated is True and r.verdict in ("Real", "Fake", "Inconclusive")
    M.reset_backends()


def test_video_path_builds_graph_audit_and_contradiction_map(settings, tmp_path):
    import shutil
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg needed to mux the synthetic audio track")
    from tests.conftest import stub_audio_fn
    from tests.test_pipeline import _write_video
    M.set_backends(image=M.StubBackend(stub_image_fn), face=M.StubFaceDetector(),
                   audio=M.StubAudioBackend(stub_audio_fn))
    path = _write_video(tmp_path / "v.mp4", frames=150, fps=30, with_audio=True)
    r = mk(settings).analyze(path, media_type="video", explain=False, stability=False,
                             force_full=True)
    assert r.contradiction_map is not None and "status" in r.contradiction_map
    assert r.graph["fusion"]["n_used"] >= 2 and verify_chain_dict(r.audit)["ok"]
    rep = json.loads(RG.generate_json_report(r.to_dict()))
    assert RG.verify_report(rep)[0] and "contradiction_map" in rep["payload"]
    M.reset_backends()
