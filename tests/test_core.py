import json

import numpy as np
import pytest
from PIL import Image

from detector import metrics
from detector.calibration import (CalibrationBundle, OODModel, conformal_quantile,
                                  fit_calibrator, sigmoid)
from detector.fusion import Evidence, fit_fusion_weights, fuse, skipped
from detector.models import find_fake_index, fake_logit_from_logits
from detector.triage import detect_c2pa, estimate_jpeg_quality, triage_image


def _data(n=600, sep=2.0, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    z = rng.normal(0, 1.5, n) + (y * 2 - 1) * sep
    return z * 3.0, y  # deliberately over-confident logits


def test_metrics():
    y = [0, 0, 1, 1]
    assert metrics.auc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert metrics.eer(y, [0.1, 0.2, 0.8, 0.9]) == 0.0
    assert metrics.tpr_at_fpr(y, [0.1, 0.2, 0.8, 0.9], 0.01) == 1.0
    assert np.isnan(metrics.auc([1, 1], [0.1, 0.2]))
    assert 0 <= metrics.brier(y, [0.1, 0.2, 0.8, 0.9]) < 0.1


def test_platt_reduces_ece():
    z, y = _data(2000)
    raw = metrics.expected_calibration_error(y, sigmoid(z))
    cal = fit_calibrator(z, y)
    zt, yt = _data(2000, seed=1)
    after = metrics.expected_calibration_error(yt, cal.prob_fake(zt))
    assert after < raw


def test_conformal_coverage_and_abstain():
    z, y = _data(3000, sep=1.0)
    cal = fit_calibrator(z, y, alpha=0.1)
    zt, yt = _data(3000, sep=1.0, seed=5)
    p = cal.prob_fake(zt)
    sets = [cal.label_set(float(pi), "balanced") for pi in p]
    cover = np.mean([("Fake" if t else "Real") in s for s, t in zip(sets, yt)])
    assert cover >= 0.85
    assert any(len(s) != 1 for s in sets)  # it does abstain somewhere
    assert cal.verdict(0.999, "balanced") == "Fake"
    assert cal.verdict(0.001, "balanced") == "Real"


def test_conformal_infinite_with_few_samples():
    assert conformal_quantile(np.array([0.1, 0.2]), 0.01) == float("inf")


def test_calibration_roundtrip(tmp_path):
    z, y = _data(400)
    b = CalibrationBundle()
    b.set("image.final", fit_calibrator(z, y))
    p = str(tmp_path / "c.json")
    b.save(p)
    b2 = CalibrationBundle.load(p)
    assert abs(b.get("image.final").prob_fake(1.0) - b2.get("image.final").prob_fake(1.0)) < 1e-9
    assert not CalibrationBundle().get("nothing").fitted if hasattr(
        CalibrationBundle().get("nothing"), "fitted") else True


def test_ood(tmp_path):
    rng = np.random.default_rng(0)
    emb = rng.normal(0, 1, (500, 8))
    m = OODModel().fit(emb)
    assert m.is_ood(rng.normal(0, 1, 8))[0] is False
    assert m.is_ood(np.full(8, 15.0))[0] is True
    p = str(tmp_path / "o.npz")
    m.save(p)
    assert OODModel.load(p).is_ood(np.full(8, 15.0))[0] is True
    assert OODModel().is_ood(np.zeros(8))[0] is None


def test_fusion():
    assert fuse([]).n_used == 0
    a = Evidence("neural", 3.0)
    b = Evidence("audio", 3.0)
    agree = fuse([a, b])
    c = Evidence("audio", -3.0)
    clash = fuse([a, c])
    assert agree.logit > 2.0 > abs(clash.logit) + 1.0 or abs(clash.logit) < 0.5
    assert clash.disagreement > agree.disagreement
    sk = fuse([a, skipped("sync", "no audio")])
    assert sk.n_used == 1
    # heuristic with low weight barely moves a strong neural score
    h = fuse([a, Evidence("rppg", -3.0, heuristic=True)])
    assert h.logit > 1.0  # still Fake-leaning (shrunk by disagreement)
    # low reliability reduces influence
    lo = fuse([Evidence("neural", 3.0, reliability=0.1), Evidence("audio", -3.0)])
    assert lo.logit < 0


def test_fit_fusion_weights():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    good = (y * 2 - 1) * 2 + rng.normal(0, 1, 500)
    noise = rng.normal(0, 1, 500)
    w = fit_fusion_weights(np.c_[good, noise], np.ones((500, 2)), y, ["neural", "sync"])
    assert w["neural"] > w["sync"]


def test_label_index_helpers():
    assert find_fake_index({0: "Realism", 1: "Deepfake"})[0] == 1
    assert find_fake_index({0: "fake", 1: "real"})[0] == 0
    z = fake_logit_from_logits(np.array([[0.0, 2.0], [3.0, 0.0]]), 1)
    assert z[0] > 0 > z[1]


def test_triage_jpeg_and_c2pa(tmp_path):
    import io
    rng = np.random.default_rng(0)
    im = Image.fromarray(rng.integers(0, 255, (128, 128, 3), dtype=np.uint8))
    b1, b2 = io.BytesIO(), io.BytesIO()
    im.save(b1, "JPEG", quality=95)
    im.save(b2, "JPEG", quality=40)
    q1 = estimate_jpeg_quality(Image.open(io.BytesIO(b1.getvalue())))
    q2 = estimate_jpeg_quality(Image.open(io.BytesIO(b2.getvalue())))
    assert q1 is not None and q2 is not None and q1 > q2
    assert detect_c2pa(b"xx jumb c2pa xx")["c2pa_manifest_detected"] is True
    assert detect_c2pa(b"xx jumb c2pa xx")["signature_verified"] is False
    assert detect_c2pa(b"nothing here")["c2pa_manifest_detected"] is False
    t = triage_image(b1.getvalue())
    assert "quality" in t or "sha256" in t
