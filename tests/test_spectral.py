import numpy as np
from PIL import Image

from detector.experts.spectral import (N_BINS, fit_head, head_logit, load_head,
                                       save_head, spectral_evidence, spectral_features)
from detector import metrics as MT
from train import train_spectral


def natural(seed, n=256):
    rng = np.random.default_rng(seed)
    f = np.fft.fft2(rng.normal(size=(n, n)))
    fy, fx = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij")
    amp = 1.0 / np.maximum(np.hypot(fy, fx), 1.0 / n) ** 1.0
    a = np.real(np.fft.ifft2(f * amp))
    a = (a - a.min()) / (a.max() - a.min()) * 255
    return Image.fromarray(a.astype(np.uint8)).convert("RGB")


def synthetic_fake(seed, n=256):
    """Nearest-neighbour 2x upsample + faint periodic grid: the kind of
    trace transposed convolutions leave."""
    small = natural(seed, n // 2).resize((n, n), Image.NEAREST)
    a = np.asarray(small, dtype=np.float64)
    a[::4, :, :] += 6
    a[:, ::4, :] += 6
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _data(k=30):
    imgs = [natural(i) for i in range(k)] + [synthetic_fake(1000 + i) for i in range(k)]
    y = np.array([0] * k + [1] * k)
    return np.array([spectral_features(i) for i in imgs]), y


def test_feature_shape_and_determinism():
    f1, f2 = spectral_features(natural(1)), spectral_features(natural(1))
    assert f1.shape == (N_BINS + 5,) and np.allclose(f1, f2)
    assert np.isfinite(f1).all()
    assert (spectral_features(Image.new("RGB", (40, 40))) == 0).all()


def test_head_separates_synthetic_traces_and_roundtrips(tmp_path):
    X, y = _data()
    head = fit_head(X, y, groups=np.arange(len(y)))
    assert head["val_auc"] > 0.9
    p = str(tmp_path / "h.npz")
    save_head(head, p)
    h2 = load_head(p)
    z = [head_logit(h2, f) for f in X]
    assert MT.auc(y, z) > 0.95
    assert load_head(str(tmp_path / "missing.npz")) is None


def test_evidence_skipped_without_head_and_votes_with_head():
    ev = spectral_evidence(natural(3), None)
    assert ev.status == "skipped"
    X, y = _data()
    head = fit_head(X, y)
    fake = spectral_evidence(synthetic_fake(5000), head)
    real = spectral_evidence(natural(5001), head)
    assert fake.logit > real.logit
    useless = dict(head, val_auc=np.array(0.5))
    assert spectral_evidence(natural(1), useless).reliability == 0.0


def test_train_script(tmp_path):
    root = tmp_path / "d"
    for lab, fn in (("real", natural), ("fake", synthetic_fake)):
        (root / lab).mkdir(parents=True)
        for i in range(12):
            fn(i + (500 if lab == "fake" else 0)).save(root / lab / f"{lab}_{i}.png")
    head = train_spectral.main(["--data", str(root), "--out", str(tmp_path / "art")])
    assert (tmp_path / "art" / "spectral_head.npz").exists() and head["val_auc"] > 0.8


# ---- TTA / uncertainty -----------------------------------------------------
def test_tta_views_and_spread():
    from detector.tta import make_views, reliability_from_spread, tta_logit
    img = natural(1)
    assert len(make_views(img, 3)) == 3 and len(make_views(img, 1)) == 1
    assert all(v.size == img.size for v in make_views(img, 3))
    agg = lambda z: float(np.mean(z))
    stable = tta_logit(lambda ims: (np.array([2.0] * len(ims)), None), [img], agg, 3)
    assert stable["spread"] == 0 and not stable["sign_flip"]
    calls = iter([1.0, -1.0, 1.0])
    flip = tta_logit(lambda ims: (np.array([next(calls)] * len(ims)), None), [img], agg, 3)
    assert flip["sign_flip"] and flip["spread"] > 0.5
    assert reliability_from_spread(0) == 1.0 > reliability_from_spread(3.0)


def test_unstable_view_blocks_early_exit(pipe):
    from detector import models as M
    from tests.conftest import make_img, stub_image_fn
    state = {"n": 0}

    def flaky(im):
        state["n"] += 1
        z, emb = stub_image_fn(im)
        return (z if state["n"] % 3 != 2 else -z), emb  # mirror view disagrees
    M.set_backends(image=M.StubBackend(flaky), face=M.StubFaceDetector())
    res = pipe.analyze(make_img(60, 0, -60), media_type="image", explain=False, stability=False)
    assert res.tier == "tier2" and res.uncertainty["sign_flip"]
    assert any("unstable" in w for w in res.warnings)
