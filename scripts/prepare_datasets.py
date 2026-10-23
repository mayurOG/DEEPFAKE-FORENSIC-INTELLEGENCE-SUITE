"""Prepare real deepfake corpora into the layout the trainers expect.

Expected layout after preparation::

    data/<dataset>/real/**/*.{jpg,png,mp4,wav,...}
    data/<dataset>/fake/**/*.{jpg,png,mp4,wav,...}

This script does **not** download copyrighted benchmarks for you (licenses
vary). It:

1. Documents canonical sources and commands for FF++, Celeb-DF, DFDC-sample,
   ASVspoof-style audio, etc.
2. Validates an existing folder tree (counts, extensions, empty classes).
3. Builds **leakage-safe** train / calib / test splits by *group* (video id /
   speaker id) and refuses overlap.

    python -m scripts.prepare_datasets --validate data/ffpp
    python -m scripts.prepare_datasets --split data/ffpp --out data/splits/ffpp \\
        --train-frac 0.6 --calib-frac 0.2 --test-frac 0.2

For actual downloads, follow each dataset's official instructions, then point
this tool at the extracted tree.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from train.data import Item, group_id, scan_folder

# Canonical public sources (documentation only — user must accept each license)
DATASET_CATALOG = {
    "FaceForensics++": {
        "url": "https://github.com/ondyari/FaceForensics",
        "media": "video/image",
        "notes": "Academic license; request access. Prefer c23 compression for realism.",
    },
    "Celeb-DF v2": {
        "url": "https://github.com/yuezunli/celeb-deepfakeforensics",
        "media": "video",
        "notes": "High-quality celebrity deepfakes; strong cross-dataset test set.",
    },
    "DFDC": {
        "url": "https://www.kaggle.com/c/deepfake-detection-challenge",
        "media": "video",
        "notes": "Large; use a sample for development. Identity leakage risk — split by folder id.",
    },
    "ASVspoof 2019/2021 LA": {
        "url": "https://www.asvspoof.org/",
        "media": "audio",
        "notes": "Standard speech anti-spoofing. Map bonafide→real, spoof→fake.",
    },
    "FakeAVCeleb": {
        "url": "https://github.com/DASH-Lab/FakeAVCeleb",
        "media": "video+audio",
        "notes": "Audio-visual deepfakes; useful for AV-consistency experts.",
    },
}


def validate_tree(root: str, media: Optional[str] = None) -> Dict:
    items = scan_folder(root, media=media)
    n_real = sum(1 for it in items if it.label == 0)
    n_fake = sum(1 for it in items if it.label == 1)
    by_mt: Dict[str, int] = defaultdict(int)
    for it in items:
        by_mt[it.media_type] += 1
    groups = {it.group for it in items}
    return {
        "root": os.path.abspath(root),
        "n_items": len(items),
        "n_real": n_real,
        "n_fake": n_fake,
        "n_groups": len(groups),
        "by_media_type": dict(by_mt),
        "ok": n_real > 0 and n_fake > 0,
        "issues": (
            ([] if n_real > 0 else ["no real/ samples"])
            + ([] if n_fake > 0 else ["no fake/ samples"])
            + ([] if items else ["empty tree — expected data/<set>/real|fake/**"])
        ),
    }


def _split_groups(groups: List[str], train_frac: float, calib_frac: float,
                  test_frac: float, seed: int) -> Tuple[Set[str], Set[str], Set[str]]:
    s = train_frac + calib_frac + test_frac
    if abs(s - 1.0) > 1e-6:
        raise SystemExit(f"fractions must sum to 1.0, got {s}")
    rng = random.Random(seed)
    g = list(groups)
    rng.shuffle(g)
    n = len(g)
    n_tr = int(n * train_frac)
    n_ca = int(n * calib_frac)
    tr, ca, te = set(g[:n_tr]), set(g[n_tr:n_tr + n_ca]), set(g[n_tr + n_ca:])
    # guarantee non-empty when possible
    if n >= 3:
        if not tr and g:
            tr.add(g[0])
        if not ca and len(g) > 1:
            ca.add(g[1])
        if not te and len(g) > 2:
            te.add(g[2])
    assert tr.isdisjoint(ca) and tr.isdisjoint(te) and ca.isdisjoint(te)
    return tr, ca, te


def build_splits(root: str, out: str, train_frac: float = 0.6,
                 calib_frac: float = 0.2, test_frac: float = 0.2,
                 seed: int = 0, media: Optional[str] = None,
                 copy_files: bool = False) -> Dict:
    """Create train/calib/test manifests with zero group overlap."""
    items = scan_folder(root, media=media)
    if not items:
        raise SystemExit(f"No items under {root}")
    groups = sorted({it.group for it in items})
    tr_g, ca_g, te_g = _split_groups(groups, train_frac, calib_frac, test_frac, seed)

    def select(gs):
        return [it for it in items if it.group in gs]

    splits = {"train": select(tr_g), "calib": select(ca_g), "test": select(te_g)}
    os.makedirs(out, exist_ok=True)
    manifests = {}
    for name, subset in splits.items():
        man = {
            "root": os.path.abspath(root),
            "split": name,
            "n": len(subset),
            "n_real": sum(1 for it in subset if it.label == 0),
            "n_fake": sum(1 for it in subset if it.label == 1),
            "groups": sorted({it.group for it in subset}),
            "items": [
                {"path": it.path, "label": it.label, "group": it.group,
                 "media_type": it.media_type}
                for it in subset
            ],
        }
        path = os.path.join(out, f"{name}.json")
        with open(path, "w") as f:
            json.dump(man, f, indent=2)
        manifests[name] = path

        if copy_files:
            dest_root = os.path.join(out, name)
            for it in subset:
                label = "real" if it.label == 0 else "fake"
                dest = os.path.join(dest_root, label, os.path.basename(it.path))
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                if not os.path.exists(dest):
                    shutil.copy2(it.path, dest)

    # Overlap hard check
    overlap = (
        (set(manifests) and set())
    )
    gsets = {k: set(json.load(open(v))["groups"]) for k, v in manifests.items()}
    assert gsets["train"].isdisjoint(gsets["calib"])
    assert gsets["train"].isdisjoint(gsets["test"])
    assert gsets["calib"].isdisjoint(gsets["test"])

    summary = {
        "source": os.path.abspath(root),
        "out": os.path.abspath(out),
        "manifests": manifests,
        "counts": {k: json.load(open(v))["n"] for k, v in manifests.items()},
        "group_counts": {k: len(json.load(open(v))["groups"]) for k, v in manifests.items()},
        "leakage_free": True,
        "catalog_hint": DATASET_CATALOG,
    }
    with open(os.path.join(out, "split_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", metavar="DIR", help="validate data/<set>/real|fake tree")
    ap.add_argument("--split", metavar="DIR", help="build leakage-safe train/calib/test")
    ap.add_argument("--out", default="data/splits/out")
    ap.add_argument("--media", default=None, choices=["image", "video", "audio"])
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--calib-frac", type=float, default=0.2)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy", action="store_true", help="copy files into split folders")
    ap.add_argument("--catalog", action="store_true", help="print dataset catalog and exit")
    a = ap.parse_args(argv)

    if a.catalog or (not a.validate and not a.split):
        print(json.dumps(DATASET_CATALOG, indent=2))
        if not a.validate and not a.split:
            return

    if a.validate:
        r = validate_tree(a.validate, a.media)
        print(json.dumps(r, indent=2))
        if not r["ok"]:
            raise SystemExit(1)

    if a.split:
        r = build_splits(
            a.split, a.out, a.train_frac, a.calib_frac, a.test_frac,
            a.seed, a.media, copy_files=a.copy,
        )
        print(json.dumps({k: r[k] for k in ("source", "out", "counts", "group_counts", "leakage_free")}, indent=2))


if __name__ == "__main__":
    main()
