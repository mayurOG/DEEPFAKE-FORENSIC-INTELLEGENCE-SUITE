import json
import os

import numpy as np
import pytest

from detector import models as M
from detector.calibration import CalibrationBundle
from detector.pipeline import Pipeline
from tests.conftest import stub_image_fn
from train import evaluate, fit_calibration, train_probe, features, finetune_ln, train_temporal
from train.data import scan_folder, group_id, group_split


def test_scan_and_group_split(dataset):
    items = scan_folder(dataset, "image")
    assert len(items) == 48 and {i.label for i in items} == {0, 1}
    tr, va = group_split(items, 0.25)
    assert not ({i.group for i in tr} & {i.group for i in va})  # no leakage


def test_celebrity_source_video_ids_share_split_group():
    root = r"C:\datasets\celebdf"
    real = group_id(r"C:\datasets\celebdf\real\id0_0001.mp4", root)
    fake = group_id(r"C:\datasets\celebdf\fake\id0_id16_0001.mp4", root)
    assert real == fake == "id0"


def test_numbered_real_fake_images_have_independent_groups():
    root = r"C:\datasets\images"
    real_0 = group_id(r"C:\datasets\images\real\real_0.jpg", root)
    real_1 = group_id(r"C:\datasets\images\real\real_1.jpg", root)
    fake_0 = group_id(r"C:\datasets\images\fake\fake_0.jpg", root)
    assert real_0 != real_1
    assert real_0 != fake_0


def test_image_backend_uses_probe_encoder_metadata(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    probe = artifact_dir / "probe.npz"
    monkeypatch.setenv("DEEPFAKE_ARTIFACT_DIR", str(artifact_dir))
    monkeypatch.setenv("EMBED_MODEL_ID", "configured-encoder")
    np.savez(probe, encoder="artifact-matched-encoder")
    M.reset_backends()
    try:
        backends = M.get_image_backends()
        assert isinstance(backends[0], M.ProbeBackend)
        assert backends[0].embed_model_id == "artifact-matched-encoder"
    finally:
        M.reset_backends()


def test_image_backend_legacy_probe_uses_configured_encoder(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    np.savez(artifact_dir / "probe.npz", w=np.ones(4))
    monkeypatch.setenv("DEEPFAKE_ARTIFACT_DIR", str(artifact_dir))
    monkeypatch.setenv("EMBED_MODEL_ID", "configured-encoder")
    M.reset_backends()
    try:
        backends = M.get_image_backends()
        assert isinstance(backends[0], M.ProbeBackend)
        assert backends[0].embed_model_id == "configured-encoder"
    finally:
        M.reset_backends()


def test_features_probe_and_probe_backend(dataset, tiny_encoder, tmp_path, settings):
    out = str(tmp_path / "f.npz")
    features.main(["--data", dataset, "--encoder", tiny_encoder, "--out", out, "--no-faces"])
    d = np.load(out, allow_pickle=True)
    assert d["emb"].shape == (48, 32)
    probe = str(tmp_path / "probe.npz")
    train_probe.main(["--train", out, "--out", probe])
    assert os.path.exists(probe)
    be = M.ProbeBackend(tiny_encoder, probe)
    from tests.conftest import make_img
    z, emb = be.predict([make_img(50, 0, -50, size=96), make_img(-50, 0, 50, size=96)])
    assert np.asarray(z).shape == (2,) and np.asarray(emb).shape[-1] == 32


def test_feature_extraction_batches_across_images(dataset):
    class BatchEncoder:
        batch_size = 7

        def __init__(self):
            self.batch_lengths = []

        def embed(self, images):
            self.batch_lengths.append(len(images))
            return np.asarray([
                np.asarray(image).mean(axis=(0, 1)) for image in images
            ], dtype=np.float32)

    items = scan_folder(dataset, "image")
    encoder = BatchEncoder()
    out = features.extract(items, encoder, detector=object(), use_faces=False)
    assert encoder.batch_lengths == [7, 7, 7, 7, 7, 7, 6]
    assert out["emb"].shape == (48, 3)
    assert out["y"].tolist() == [item.label for item in items]
    assert out["paths"].tolist() == [item.path for item in items]


def test_hf_classifier_backend_tiny(tmp_path):
    from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor
    cfg = ViTConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                    intermediate_size=64, image_size=32, patch_size=16,
                    id2label={0: "Realism", 1: "Deepfake"}, label2id={"Realism": 0, "Deepfake": 1})
    d = str(tmp_path / "clf")
    ViTForImageClassification(cfg).save_pretrained(d)
    ViTImageProcessor(size={"height": 32, "width": 32}).save_pretrained(d)
    be = M.HFClassifierBackend(d)
    from tests.conftest import make_img
    z, emb = be.predict([make_img(0, 0, 0, size=64)])
    assert np.isfinite(np.asarray(z)).all()


def test_calibrate_then_evaluate(dataset, pipe, settings, tmp_path, monkeypatch):
    items = scan_folder(dataset, "image")
    rows = fit_calibration.collect_scores(pipe, items, "image")
    assert len(rows) == 48
    rep, bundle = fit_calibration.fit_from_rows(rows, "image", 0.1, out_dir=settings.artifact_dir)
    bundle.save(os.path.join(settings.artifact_dir, "calibration.json"))
    assert rep["final"]["auc"] > 0.95 and rep["ood_fitted"]
    assert os.path.exists(os.path.join(settings.artifact_dir, "calibration.json"))
    pipe.reload()
    assert pipe.status()  # loads calibration
    res = evaluate.evaluate_items(pipe, items, "image", "balanced")
    assert res["cascade"]["auc"] > 0.95 and res["decisions"]["false_alarm_rate"] <= 0.1
    json.dumps(res)


def test_finetune_ln_smoke(dataset, tiny_encoder, tmp_path):
    out = str(tmp_path / "ft")
    finetune_ln.main(["--data", dataset, "--encoder", tiny_encoder, "--out", out,
                      "--epochs", "1", "--batch-size", "8", "--no-faces", "--max-items", "16"])
    assert os.path.exists(os.path.join(out, "probe.npz"))


def test_temporal_head_fit():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 80)
    X = rng.normal(0, 1, (80, 12, 8)) + y[:, None, None] * 0.8
    head = train_temporal.fit_head(X, y, epochs=30)
    assert head is not None


def test_run_all_end_to_end(dataset, tiny_encoder, tmp_path, monkeypatch):
    """Orchestration on synthetic folders with stub Tier-1 (offline)."""
    import shutil
    from train import run_all
    from tests.conftest import make_img
    monkeypatch.setenv("DEEPFAKE_RUN_ALL_KEEP_BACKENDS", "1")
    M.set_backends(image=M.StubBackend(stub_image_fn), face=M.StubFaceDetector())
    sets = {}
    for name in ("calib", "test"):
        root = tmp_path / name
        for lab, (r, b) in {"real": (-50, 50), "fake": (50, -50)}.items():
            (root / lab).mkdir(parents=True)
            for i in range(20):
                make_img(r, 0, b, seed=900 + i, size=96).save(root / lab / f"{lab}_{i}.png")
        sets[name] = str(root)
    out = str(tmp_path / "art")
    s = run_all.main(["--train", dataset, "--calib", sets["calib"], "--test", sets["test"],
                      "--encoder", tiny_encoder, "--out", out, "--no-faces"])
    M.reset_backends()
    failed = {k: v for k, v in s["steps"].items() if not v["ok"]}
    assert not failed, failed
    for f in ("probe.npz", "spectral_head.npz", "calibration.json", "evaluation.json",
              "run_summary.json"):
        assert os.path.exists(os.path.join(out, f)), f


def test_run_all_rejects_overlap(dataset, tiny_encoder, tmp_path):
    from train import run_all
    with pytest.raises(SystemExit):
        run_all.main(["--train", dataset, "--calib", dataset, "--encoder", tiny_encoder,
                      "--out", str(tmp_path / "a")])
