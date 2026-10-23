import json
import os
import subprocess
import shutil

import cv2
import numpy as np
import pytest
from PIL import Image

from detector import report_generator as RG
from detector.experts.facesignals import chrom_pulse, pulse_snr
from detector.experts.tamper import tamper_map
from detector.experts.temporal import aggregate_frame_logits, temporal_features
from detector.explain import deletion_faithfulness, occlusion_map
from detector.stability import stability_report
from tests.conftest import make_img, stub_image_fn


def test_image_verdicts(pipe):
    fake = pipe.analyze(make_img(60, 0, -60), media_type="image", filename="f.png")
    real = pipe.analyze(make_img(-60, 0, 60), media_type="image", filename="r.png")
    assert fake.verdict == "Fake" and real.verdict == "Real"
    assert fake.calibrated is False or fake.calibrated is True
    assert any(e["name"].startswith("neural") for e in fake.evidence)
    d = fake.to_dict()
    json.dumps(d)  # serialisable


def test_invalid_mode(pipe):
    with pytest.raises(ValueError):
        pipe.analyze(make_img(0, 0, 0), media_type="image", mode="bogus")


def test_modes_monotone(pipe):
    img = make_img(8, 0, 0)
    v = {m: pipe.analyze(img, media_type="image", mode=m, explain=False,
                         stability=False).verdict
         for m in ("strict", "balanced", "sensitive")}
    assert set(v.values()) <= {"Real", "Fake", "Inconclusive"}


def test_expert_pulse():
    fps, n = 30.0, 300
    t = np.arange(n) / fps
    rng = np.random.default_rng(0)
    base = np.array([150.0, 110.0, 90.0])
    pulse = np.sin(2 * np.pi * 1.2 * t)
    rgb = base + np.c_[0.5 * pulse, 1.5 * pulse, 0.3 * pulse] + rng.normal(0, 0.05, (n, 3))
    hr, snr_live = pulse_snr(chrom_pulse(rgb, fps), fps)
    assert abs(hr - 72) < 6 and snr_live > 3
    noise = base + rng.normal(0, 1.0, (n, 3))
    _, snr_noise = pulse_snr(chrom_pulse(noise, fps), fps)
    assert snr_live > snr_noise


def test_tamper_map_flags_pasted_region():
    rng = np.random.default_rng(0)
    a = rng.normal(128, 4, (256, 256, 3))
    a[64:160, 64:160] += rng.normal(0, 30, (96, 96, 3))  # different noise level
    tm = tamper_map(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)),
                    face_box=(64, 64, 160, 160))
    assert "face_bg_log_ratio" in tm and abs(tm["face_bg_log_ratio"]) > 0.3


def test_temporal_aggregate():
    assert aggregate_frame_logits([]) == 0.0
    assert aggregate_frame_logits([-5, -5, -5, 4]) > np.mean([-5, -5, -5, 4])
    f = temporal_features([0, 1, 5, 6], [0, 1, 2, 3])
    assert isinstance(f, dict)


def test_explain_and_faithfulness():
    img = make_img(0, 0, 0)
    arr = np.asarray(img).copy()
    arr[:128, :128] = [250, 0, 0]  # red patch the stub finds "fake"
    img = Image.fromarray(arr)

    def score(imgs):
        return np.array([stub_image_fn(i)[0] for i in imgs])

    om = occlusion_map(score, img, grid=4)
    imp = np.asarray(om["importance"])
    assert imp.shape == (4, 4)
    assert imp[:2, :2].mean() > imp[2:, 2:].mean()
    f = deletion_faithfulness(score, img, imp)
    assert "faithfulness" in f


def test_stability():
    imgs = [make_img(60, 0, -60)]
    r = stability_report(imgs, lambda ims: float(
        1 / (1 + np.exp(-stub_image_fn(ims[0])[0]))))
    assert 0.0 <= r["stability_score"] <= 1.0


def test_report_sign_verify_tamper(pipe, art_dir):
    res = pipe.analyze(make_img(60, 0, -60), media_type="image", filename="f.png")
    js = RG.generate_json_report(res.to_dict())
    rep = json.loads(js)
    ok, msg = RG.verify_report(rep)
    assert ok, msg
    rep["payload"]["verdict"] = "Real" if rep["payload"].get("verdict") != "Real" else "Fake"
    ok2, _ = RG.verify_report(rep)
    assert not ok2
    key = os.path.join(art_dir, "report_signing_key.pem")
    assert os.path.exists(key) and (os.stat(key).st_mode & 0o077) == 0


def test_html_report_escapes(pipe):
    res = pipe.analyze(make_img(60, 0, -60), media_type="image",
                       filename="<script>alert(1)</script>.png")
    html = RG.generate_html_report(res.to_dict(), res.artifacts)
    assert "<script>alert(1)</script>" not in html
    assert "Fake" in html


def _write_video(path, frames=60, fps=30, size=160, with_audio=False, tmp=None):
    raw = str(path) + ".raw.mp4"
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    rng = np.random.default_rng(0)
    for i in range(frames):
        a = rng.normal(128, 10, (size, size, 3)); a[..., 2] += 70  # reddish in BGR
        vw.write(np.clip(a, 0, 255).astype(np.uint8))
    vw.release()
    if with_audio and shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-f", "lavfi",
                        "-i", "sine=frequency=300:duration=2", "-shortest", "-c:v", "copy",
                        "-c:a", "aac", str(path)], check=True)
        return str(path)
    os.replace(raw, path)
    return str(path)


def test_video_pipeline(pipe, tmp_path):
    p = _write_video(tmp_path / "v.mp4", with_audio=True)
    res = pipe.analyze(p, media_type="video", filename="v.mp4", explain=False,
                       stability=False)
    assert res.verdict in ("Real", "Fake", "Inconclusive")
    assert res.media_type == "video"
    names = [e["name"] for e in res.evidence]
    assert "temporal" in names
    json.dumps(res.to_dict())


def test_audio_pipeline(pipe, tmp_path):
    import soundfile as sf
    sr = 16000
    t = np.arange(sr * 6) / sr
    sf.write(str(tmp_path / "a.wav"), (0.5 * np.sin(2 * np.pi * 220 * t)).astype("float32"), sr)
    res = pipe.analyze(str(tmp_path / "a.wav"), media_type="audio", filename="a.wav")
    assert res.verdict in ("Real", "Fake", "Inconclusive")
    json.dumps(res.to_dict())


def test_ood_downgrades_to_inconclusive(pipe, settings):
    from detector.calibration import OODModel
    rng = np.random.default_rng(0)
    m = OODModel().fit(rng.normal(0, 1, (300, 4)))
    os.makedirs(settings.artifact_dir, exist_ok=True)
    m.save(os.path.join(settings.artifact_dir, "ood_image.npz"))
    pipe.reload()
    res = pipe.analyze(make_img(60, 0, -60), media_type="image", explain=False,
                       stability=False)
    assert res.verdict == "Inconclusive"


def test_counterfactual_and_multiscale():
    from detector.explain import counterfactual_evidence, multiscale_occlusion
    arr = np.asarray(make_img(0, 0, 0)).copy()
    arr[:128, :128] = [250, 0, 0]  # compact fake evidence in one quadrant
    img = Image.fromarray(arr)
    score = lambda imgs: np.array([stub_image_fn(i)[0] for i in imgs])
    ms = multiscale_occlusion(score, img, grids=(3, 6))
    imp = np.asarray(ms["importance"])
    assert imp.shape == (18, 18) and imp[:9, :9].mean() > imp[9:, 9:].mean()
    cf = counterfactual_evidence(score, img, imp, threshold=2.0, grid=6)
    assert cf["applicable"] and cf["flip_fraction"] is not None
    assert cf["flip_fraction"] <= 0.5          # removing the quadrant flips it
    assert cf["sufficient_fraction"] is not None
    # a clearly real image: not applicable
    real = counterfactual_evidence(score, make_img(-60, 0, 60), np.zeros((6, 6)))
    assert real["applicable"] is False


def test_narrative_mentions_new_explanations(pipe):
    res = pipe.analyze(make_img(60, 0, -60), media_type="image", filename="f.png")
    ex = res.explanation
    assert ex["method"] == "multiscale-occlusion" and "counterfactual" in ex
    assert "Consistency under mirror/crop" in res.narrative
    assert "Counterfactual" in res.narrative
