import numpy as np
from PIL import Image

from detector.adversarial import evasion_attack, robustness_curve
from detector.attribution import fit_attributor, predict_family, save_attributor, load_attributor
from detector.experts.spectral import spectral_features
from detector.grounded_explain import build_claims, ground_text
from detector.provenance import inspect_provenance
from detector.shift_conformal import (AdaptiveConformal, estimate_density_ratio,
                                      prediction_set, weighted_class_quantiles,
                                      weighted_conformal_quantile)
from detector.transformation_lab import audio_transforms


# ---- shift-aware conformal -------------------------------------------------
def test_weighted_quantile_reduces_to_split_with_unit_weights():
    rng = np.random.default_rng(0)
    s = rng.uniform(size=999)
    q = weighted_conformal_quantile(s, np.ones_like(s), 1.0, 0.1)
    assert abs(q - np.quantile(s, 0.9)) < 0.01


def test_weighted_conformal_covers_better_under_covariate_shift():
    rng = np.random.default_rng(1)
    # features: cal ~ N(0,1), target ~ N(1.2,1). Score grows with x, so the
    # target is harder and plain split conformal undercovers.
    def draw(mu, n):
        x = rng.normal(mu, 1, (n, 1))
        return x, np.abs(x[:, 0] + rng.normal(0, 0.5, n))
    xc, sc = draw(0.0, 4000)
    xt, st = draw(1.2, 4000)
    dr = estimate_density_ratio(xc, xt)
    plain = np.quantile(sc, 0.9)
    cover_plain = float(np.mean(st <= plain))
    covs = []
    for x, s in zip(xt[:400], st[:400]):
        q = weighted_conformal_quantile(sc, dr(xc), float(dr(x[None])[0]), 0.1)
        covs.append(s <= q)
    assert np.mean(covs) > cover_plain + 0.03
    assert np.mean(covs) > 0.85


def test_adaptive_conformal_tracks_target_under_drift():
    """After a shift the realised error is 3x the working alpha. A fixed
    alpha=0.1 would then miscover ~30%; ACI should pull it back near 10%."""
    ac = AdaptiveConformal(alpha=0.1, gamma=0.02)
    rng = np.random.default_rng(2)
    late = []
    for t in range(8000):
        err_p = min(1.0, 3.0 * ac.effective_alpha())
        covered = rng.uniform() > err_p
        ac.update(bool(covered))
        if t >= 4000:
            late.append(not covered)
    assert 0.07 < float(np.mean(late)) < 0.13


def test_prediction_set_and_class_quantiles():
    qr, qf = weighted_class_quantiles([0.1, 0.2, 0.9, 0.8], [0, 0, 1, 1], [1, 1, 1, 1], 1.0, 0.5)
    assert prediction_set(0.05, 0.5, 0.5) == ["Real"]
    assert prediction_set(0.5, 0.5, 0.5) == ["Real", "Fake"]


# ---- provenance --------------------------------------------------------------
def test_provenance_detects_declared_ai_and_stays_neutral_otherwise():
    p = inspect_provenance(b"\xff\xd8 jumb c2pa claim_generator\x00ExampleGen/1.0 "
                           b"http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia")
    assert p["c2pa_present"] and p["declared_ai_source"] == "trainedalgorithmicmedia"
    assert p["signature_validated"] is False
    q = inspect_provenance(b"\xff\xd8 plain jpeg bytes")
    assert not q["c2pa_present"] and q["declared_ai_source"] is None


# ---- attribution -----------------------------------------------------------
def _fam(kind, seed):
    rng = np.random.default_rng(seed)
    base = rng.normal(128, 40, (32, 32, 3)).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(base)
    mode = {"nearest": Image.NEAREST, "bilinear": Image.BILINEAR, "lanczos": Image.LANCZOS}[kind]
    return im.resize((256, 256), mode)


def test_attributor_separates_upsampling_families_and_roundtrips(tmp_path):
    kinds = ["nearest", "bilinear", "lanczos"]
    X, fam = [], []
    for k in kinds:
        for s in range(40):
            X.append(spectral_features(_fam(k, s)))
            fam.append(k)
    head = fit_attributor(np.array(X), fam, unknown_below=0.0)
    save_attributor(head, str(tmp_path / "a.npz"))
    h2 = load_attributor(str(tmp_path / "a.npz"))
    ok = sum(predict_family(h2, spectral_features(_fam(k, 1000 + s)))["top"] == k
             for k in kinds for s in range(10))
    assert ok >= 24      # >= 80% on fresh seeds of synthetic upsampling families


def test_attributor_says_unknown_for_far_samples():
    X = np.vstack([np.random.default_rng(0).normal(0, 1, (40, 6)),
                   np.random.default_rng(1).normal(4, 1, (40, 6))])
    head = fit_attributor(X, ["a"] * 40 + ["b"] * 40)
    far = predict_family(head, np.full(6, 60.0))
    assert far["family"] == "unknown"


# ---- adversarial -------------------------------------------------------------
def test_evasion_attack_flips_a_fragile_scorer_and_not_a_robust_one():
    img = Image.fromarray(np.full((64, 64, 3), 100, np.uint8))
    fragile = lambda im: 0.2 - float(np.asarray(im, float).std()) * 0.5   # noise => lower score
    r = evasion_attack(fragile, img, eps=8, queries=80, block=16)
    assert r["flipped"] and r["queries"] <= 81
    robust = lambda im: 5.0
    assert not evasion_attack(robust, img, eps=8, queries=10)["flipped"]
    curve = robustness_curve(robust, [img], (1, 4), queries=5)
    assert all(row["flip_rate"] == 0 for row in curve["rows"])


# ---- grounded explanations ------------------------------------------------
RES = {"verdict": "Fake", "evidence": [
    {"name": "neural", "status": "used", "logit": 2.4, "reliability": 0.8},
    {"name": "spectral", "status": "used", "logit": -1.1, "reliability": 0.5},
    {"name": "audio", "status": "skipped", "logit": None}]}


def test_grounded_text_drops_unsupported_and_contradicting_sentences():
    txt = ("The neural expert finds synthetic texture (log-odds 2.4). "
           "The spectral expert says the image looks fake. "
           "The skin looks plastic and the lighting is odd. "
           "The neural expert finds log-odds 9.9.")
    g = ground_text(txt, RES)
    kept = [k["sentence"] for k in g["kept"]]
    reasons = {d["sentence"]: d["reason"] for d in g["dropped"]}
    assert kept == ["The neural expert finds synthetic texture (log-odds 2.4)."]
    assert "contradicts spectral" in reasons["The spectral expert says the image looks fake."]
    assert "no recorded evidence" in reasons["The skin looks plastic and the lighting is odd."]
    assert "number" in reasons["The neural expert finds log-odds 9.9."]
    assert 0 < g["grounded_fraction"] < 1
    assert {c["id"] for c in build_claims(RES)} >= {"ev:neural", "ev:spectral", "verdict"}


# ---- audio laundering --------------------------------------------------------
def test_audio_laundering_transforms_keep_shape_and_change_signal():
    sr = 16000
    w = (0.3 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)).astype(np.float32)
    tr = {t.name: t for t in audio_transforms(sr)}
    for name in ("codec_mulaw", "reverb_rt60_04"):
        out = tr[name].fn(w) if hasattr(tr[name], "fn") else tr[name](w)
        assert out.shape == w.shape and np.isfinite(out).all() and not np.allclose(out, w)


# ---- pipeline integration ----------------------------------------------------
def test_pipeline_emits_grounded_claims_and_attribution_and_provenance(settings, tmp_path):
    import dataclasses, io
    from detector import models as M
    from detector.pipeline import Pipeline
    from tests.conftest import make_img, stub_image_fn

    M.set_backends(image=M.StubBackend(stub_image_fn), face=M.StubFaceDetector())
    kinds = ["nearest", "bilinear"]
    X = np.array([spectral_features(_fam(k, s)) for k in kinds for s in range(30)])
    head = fit_attributor(X, [k for k in kinds for _ in range(30)], unknown_below=0.0)
    save_attributor(head, str(tmp_path / "attribution_head.npz"))
    p = Pipeline(settings=dataclasses.replace(settings, artifact_dir=str(tmp_path)))
    p.attribution_head = load_attributor(str(tmp_path / "attribution_head.npz"))
    buf = io.BytesIO()
    make_img(60, 0, -60).save(buf, "PNG", pnginfo=None)
    data = buf.getvalue() + b"c2pa jumb trainedAlgorithmicMedia"
    r = p.analyze(data, media_type="image", filename="x.png", force_full=True,
                  explain=False, stability=False)
    assert r.grounded["claims"], "claims are always emitted"
    assert any(c["id"] == "verdict" for c in r.grounded["claims"])
    if r.verdict == "Fake":
        assert r.attribution and r.attribution["top"] in kinds
    prov = (r.metadata or {}).get("provenance")
    assert prov and prov["declared_ai_source"] == "trainedalgorithmicmedia"
    names = [e["name"] for e in r.evidence if e.get("status") == "used"]
    assert "metadata_ai" in names
    M.reset_backends()
