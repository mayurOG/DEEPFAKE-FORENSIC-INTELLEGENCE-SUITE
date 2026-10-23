"""CLI for CALM-FORGE cross-asset lineage investigation.

    python -m scripts.investigate_lineage file1.jpg file2.jpg file3.mp4 \\
        --out artifacts/lineage_report.json

Optional: --no-forensics (fingerprints + graph only)
          --no-shortcut-audit
"""
from __future__ import annotations

import argparse
import json
import logging
import os

from detector.cross_asset import CrossAssetSession
from detector.lineage_report import save_report, to_markdown


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="media files to investigate together")
    ap.add_argument("--out", default="artifacts/lineage_report.json")
    ap.add_argument("--no-forensics", action="store_true",
                    help="skip per-file deepfake pipeline (faster)")
    ap.add_argument("--no-shortcut-audit", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    for p in a.paths:
        if not os.path.isfile(p):
            raise SystemExit(f"Not a file: {p}")

    session = CrossAssetSession(run_forensics=not a.no_forensics)
    report = session.investigate(
        a.paths, shortcut_audit=not a.no_shortcut_audit,
    )
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    save_report(report, a.out)
    md = os.path.splitext(a.out)[0] + ".md"
    with open(md, "w") as f:
        f.write(to_markdown(report))
    print(json.dumps({
        "session_id": report["session_id"],
        "n_assets": report["registry"]["n_assets"],
        "n_edges": report["lineage_graph"]["n_edges"],
        "out": a.out,
        "markdown": md,
    }, indent=2))
    return report


if __name__ == "__main__":
    main()
