"""Unit tests for CALM-FORGE lineage modules (synthetic media only)."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from detector.asset_registry import AssetRegistry
from detector.media_fingerprints import (
    fingerprint_image, ahash64, hamming, fingerprint_image_path,
)
from detector.lineage_matcher import score_pair, match_all
from detector.lineage_graph import LineageGraph, LineageNode
from detector.mutation_events import events_from_relationships
from detector.shortcut_audit import audit_image
from detector.cross_asset import CrossAssetSession


def _img(color, size=(128, 128)):
    return Image.new("RGB", size, color)


def test_registry_dedup_by_hash():
    reg = AssetRegistry()
    img = _img((10, 20, 30))
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    a = reg.register(data, media_type="image", filename="a.png")
    b = reg.register(data, media_type="image", filename="b.png")
    assert a.asset_id == b.asset_id
    assert len(reg.list()) == 1


def test_perceptual_hash_similar_for_resize():
    im = _img((200, 100, 50), (256, 256))
    # draw a simple shape so hash isn't uniform
    for x in range(40, 200):
        for y in range(40, 200):
            im.putpixel((x, y), (20, 180, 90))
    small = im.resize((128, 128), Image.Resampling.BILINEAR).resize((256, 256))
    h1, h2 = ahash64(im), ahash64(small)
    assert hamming(h1, h2) < 12


def test_lineage_near_duplicate():
    im = _img((80, 80, 80), (200, 200))
    for x in range(20, 180):
        im.putpixel((x, 100), (255, 0, 0))
    fp1 = fingerprint_image(im)
    fp2 = fingerprint_image(im.resize((180, 180)).resize((200, 200)))
    rel = score_pair(fp1, fp2, "a1", "a2")
    assert rel.confidence > 0.5
    assert rel.relation in (
        "near_duplicate", "resized", "likely_derivative", "weak_similarity",
    )


def test_lineage_graph_components():
    g = LineageGraph(session_id="t")
    for i in range(3):
        g.add_node(LineageNode(asset_id=f"n{i}", filename=f"{i}.jpg"))
    from detector.lineage_matcher import RelationshipScore
    g.add_relationship(RelationshipScore("n0", "n1", "near_duplicate", 0.95, 0.1))
    comps = g.components()
    assert any(len(c) >= 2 for c in comps)


def test_shortcut_audit_stable_on_constant_scorer():
    im = _img((128, 128, 128))

    def score_fn(image):
        return 0.2, "Real"

    rep = audit_image(im, score_fn, asset_id="x")
    assert rep.fragility == 0.0
    assert rep.likely_shortcut_reliance is False
    assert len(rep.results) >= 3


def test_shortcut_audit_detects_flip():
    im = _img((128, 128, 128))
    state = {"n": 0}

    def flaky(image):
        state["n"] += 1
        # baseline Real, then flips on transforms
        if state["n"] == 1:
            return 0.2, "Real"
        return 0.9, "Fake"

    rep = audit_image(im, flaky, asset_id="y")
    assert rep.nuisance_flip_rate > 0 or rep.fragility > 0


def test_cross_asset_session_fingerprints_only():
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i, col in enumerate([(255, 0, 0), (255, 10, 10)]):
            p = os.path.join(td, f"im{i}.png")
            im = _img(col, (160, 160))
            for x in range(30, 130):
                for y in range(30, 130):
                    im.putpixel((x, y), (0, 0, 255))
            im.save(p)
            paths.append(p)
        session = CrossAssetSession(run_forensics=False)
        report = session.investigate(paths, shortcut_audit=False)
        assert report["registry"]["n_assets"] == 2
        assert "lineage_graph" in report
        assert report["report_type"] == "calm_forge_lineage"


def test_hash_sim_unrelated_near_zero():
    """Random 64-bit hashes must not score as weak_similarity after calibration."""
    from detector.lineage_matcher import _hash_sim
    import random
    random.seed(0)
    sims = []
    for _ in range(100):
        a = "".join(str(random.randint(0, 1)) for _ in range(64))
        b = "".join(str(random.randint(0, 1)) for _ in range(64))
        sims.append(_hash_sim(a, b))
    assert sum(sims) / len(sims) < 0.15
    assert max(sims) < 0.55


def test_infer_roles_prefers_higher_resolution_as_original():
    g = LineageGraph(session_id="roles")
    g.add_node(LineageNode(
        asset_id="hi", filename="hi.png",
        attributes={"size": [256, 256]},
    ))
    g.add_node(LineageNode(
        asset_id="lo", filename="lo.png",
        attributes={"size": [128, 128]},
    ))
    from detector.lineage_matcher import RelationshipScore
    # Edge direction deliberately opposite to resolution order
    g.add_relationship(RelationshipScore(
        "lo", "hi", "resized", 0.9, 0.1,
        evidence={"similarities": {"ahash": 0.9}},
    ))
    g.infer_roles()
    assert g.nodes["hi"].role_hypothesis == "original_candidate"
    assert g.nodes["lo"].role_hypothesis == "derivative_candidate"
