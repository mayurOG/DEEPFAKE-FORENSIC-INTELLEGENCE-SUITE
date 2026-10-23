#!/usr/bin/env python3
"""Command-line scanner for images, videos and audio.

    python cli_scan.py -i suspect.mp4 --mode strict
"""
import argparse
import os
import sys

from detector.pipeline import get_pipeline, infer_media_type
from detector.report_generator import (
    build_payload, generate_html_report, sign_payload,
)
import json


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Deepfake Forensic Intelligence CLI v3.0 (calibrated evidence cascade)"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Path to input media file (image, video or audio)")
    parser.add_argument("--output-dir", "-o", default="./forensic_reports",
                        help="Directory to save generated forensic reports")
    parser.add_argument("--mode", "-m", default="balanced",
                        choices=["strict", "balanced", "sensitive"],
                        help="strict = fewest false alarms, sensitive = catch more fakes")
    parser.add_argument("--no-explain", action="store_true",
                        help="skip heatmaps and stability test (faster)")
    args = parser.parse_args(argv)

    path = args.input
    if not os.path.exists(path):
        print(f"Error: file '{path}' does not exist.")
        return 1
    media = infer_media_type(path)
    if media is None:
        print(f"Unsupported file type: {path}")
        return 1
    os.makedirs(args.output_dir, exist_ok=True)
    filename = os.path.basename(path)
    print(f"Analyzing {filename} ({media})...")

    res = get_pipeline().analyze(
        path, media_type=media, mode=args.mode,
        explain=not args.no_explain, stability=not args.no_explain,
    )
    d = res.to_dict()

    print("\n==========================================")
    print(f"  VERDICT : {res.verdict.upper()}")
    print(f"  P(fake) : {res.p_fake * 100:.1f}%  (path: {res.tier}, mode: {res.mode})")
    if res.confidence is not None:
        print(f"  Confidence in verdict: {res.confidence * 100:.1f}%")
    print(f"  Calibrated: {'yes' if res.calibrated else 'NO (uncalibrated fallback)'}")
    print("==========================================\n")
    print(res.narrative)
    for w in res.warnings:
        print(f"! {w}")

    signed = {"payload": build_payload(d)}
    signed["signature"] = sign_payload(signed["payload"])
    base = os.path.join(args.output_dir, filename)
    with open(base + "_report.json", "w", encoding="utf-8") as f:
        json.dump(signed, f, indent=2, ensure_ascii=False)
    with open(base + "_report.html", "w", encoding="utf-8") as f:
        f.write(generate_html_report(d, res.artifacts, signed))
    print(f"\nReports written to '{args.output_dir}':")
    print(f"  {base}_report.html\n  {base}_report.json (signed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
