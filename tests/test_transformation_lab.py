import numpy as np
from PIL import Image

from detector.transformation_lab import (IMAGE_TRANSFORMS, Transform, audio_transforms,
                                         robustness_gate, robustness_profile)
from tests.conftest import make_img, stub_image_fn


def score_imgs(imgs):
    return [stub_image_fn(i)[0] for i in imgs]


def test_stable_input_has_low_fragility_and_passes_gate():
    p = robustness_profile(score_imgs, make_img(80, 0, -80))
    assert p["base_call"] == "Fake" and p["fragile_to"] == []
    assert p["fragility"] < 0.2 and p["mean_retained"] > 0.7
    assert robustness_gate(p)["ok"]
    assert {r["transform"] for r in p["rows"]} == {t.name for t in IMAGE_TRANSFORMS}


def test_fragile_input_flips_and_is_gated():
    # decision hinges on high-frequency detail that JPEG/blur/downscale remove
    def hf_score(imgs):
        out = []
        for im in imgs:
            a = np.asarray(im.convert("L"), dtype=np.float64)
            out.append((np.abs(np.diff(a, axis=1)).mean() - 12.0) / 2.0)
        return out
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(100, 160, (128, 128, 3), dtype=np.uint8))
    p = robustness_profile(hf_score, img)
    assert p["base_call"] == "Fake" and p["fragile_to"]
    assert p["fragility"] > 0.3
    g = robustness_gate(p)
    assert not g["ok"] and g["reason"]


def test_plausibility_ties_gate_to_quality():
    heavy = Transform("jpeg_q35", 0.7, lambda x: x, ("high",))
    assert not heavy.plausible_given({"compression": "low"})   # clean original: hypothetical
    assert heavy.plausible_given({"compression": "high"})      # already crushed: realistic
    # scalar "media": a mild transform keeps the score, severe ones flip it
    ts = [Transform("mild", 0.2, lambda x: x), Transform("severe_a", 0.7, lambda x: -x, ("high",)),
          Transform("severe_b", 0.8, lambda x: -x, ("high",))]
    score = lambda xs: list(xs)
    clean = robustness_profile(score, 3.0, ts, quality={"compression": "low"})
    crushed = robustness_profile(score, 3.0, ts, quality={"compression": "high"})
    assert clean["fragile_to"] == [] and crushed["fragile_to"] == ["severe_a", "severe_b"]
    assert crushed["fragility"] > clean["fragility"]


def test_verdict_fn_and_audio_transforms():
    p = robustness_profile(score_imgs, make_img(80, 0, -80),
                           verdict_fn=lambda z: "Fake" if z > 5.0 else "Inconclusive")
    assert p["base_call"] in ("Fake", "Inconclusive")
    sr = 16000
    w = (0.4 * np.sin(2 * np.pi * 220 * np.arange(sr * 2) / sr)).astype(np.float32)
    tr = audio_transforms(sr)
    outs = [t.fn(w) for t in tr]
    assert all(o.shape == w.shape and np.isfinite(o).all() for o in outs)
    pa = robustness_profile(lambda ws: [float(np.std(x)) * 10 - 1 for x in ws], w, tr)
    assert len(pa["rows"]) == len(tr)
