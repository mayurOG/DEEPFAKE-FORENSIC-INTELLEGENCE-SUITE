#!/usr/bin/env python3
"""
Sync fine-tuned artifacts from Google Colab back into the local project.

Usage:
    python scripts/sync_from_colab.py --zip ~/Downloads/deepfake_artifacts.zip
    python scripts/sync_from_colab.py --zip ~/Downloads/deepfake_artifacts.zip --out artifacts --env .env

This script:
1. Unzips the Colab artifact bundle
2. Merges all artifacts into the local artifacts/ directory
3. Updates .env with the new finetuned_encoder path
4. Prints a summary of what changed
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path
from datetime import datetime, timezone


def update_env(key: str, value: str, env_path: str = ".env"):
    """Update or insert KEY=VALUE in .env without touching other lines."""
    env_file = Path(env_path)
    if not env_file.exists():
        env_file.write_text(f"{key}={value}\n")
        print(f"  Created {env_path} → {key}={value}")
        return
    content = env_file.read_text(encoding="utf-8")
    pattern = rf"^{re.escape(key)}=.*$"
    if re.search(pattern, content, flags=re.MULTILINE):
        new_content = re.sub(pattern, f"{key}={value}", content, flags=re.MULTILINE)
    else:
        new_content = content.rstrip() + f"\n{key}={value}\n"
    env_file.write_text(new_content, encoding="utf-8")
    print(f"  Updated .env → {key}={value}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True, help="Path to deepfake_artifacts.zip downloaded from Colab")
    ap.add_argument("--out", default="artifacts", help="Local artifacts directory (default: artifacts/)")
    ap.add_argument("--env", default=".env", help="Path to .env file (default: .env)")
    ap.add_argument("--dry-run", action="store_true", help="Show what would happen without making changes")
    a = ap.parse_args()

    zip_path = Path(a.zip).expanduser().resolve()
    if not zip_path.exists():
        sys.exit(f"ERROR: ZIP file not found: {zip_path}")

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'[DRY RUN] ' if a.dry_run else ''}Syncing Colab artifacts")
    print(f"  ZIP source : {zip_path}  ({zip_path.stat().st_size/1e6:.1f} MB)")
    print(f"  Target dir : {out_dir.resolve()}")
    print()

    # List ZIP contents
    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = zf.namelist()
        print(f"  ZIP contains {len(all_names)} files")

        # Extract everything to a temp staging area
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            if not a.dry_run:
                zf.extractall(tmpdir)

            staged = Path(tmpdir)

            # Copy artifacts/ contents
            staged_art = staged / "artifacts"
            if staged_art.is_dir():
                for f in staged_art.rglob("*"):
                    if f.is_file():
                        rel = f.relative_to(staged_art)
                        dest = out_dir / rel
                        if a.dry_run:
                            print(f"  WOULD COPY: artifacts/{rel}")
                        else:
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(f, dest)
                            print(f"  Copied: {rel}  ({f.stat().st_size/1024:.1f} KB)")
            else:
                # ZIP may have files at root level
                for name in all_names:
                    if name.startswith("artifacts/"):
                        rel = Path(name).relative_to("artifacts")
                        if not a.dry_run:
                            src = staged / name
                            if src.exists():
                                dest = out_dir / rel
                                dest.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(src, dest)
                                print(f"  Copied: {rel}")

            # Update .env with new encoder path
            encoder_dir = out_dir / "finetuned_encoder"
            if encoder_dir.is_dir():
                if a.dry_run:
                    print(f"\n  WOULD UPDATE .env → EMBED_MODEL_ID={encoder_dir}")
                else:
                    update_env("EMBED_MODEL_ID", str(encoder_dir), a.env)
            else:
                print("  Note: finetuned_encoder/ not found in artifacts — EMBED_MODEL_ID unchanged")

            # Log summary
            ts = datetime.now(timezone.utc).isoformat()
            summary = {
                "synced_at": ts,
                "source_zip": str(zip_path),
                "dry_run": a.dry_run,
                "files_synced": [n for n in all_names if n.startswith("artifacts/")],
            }
            summary_path = out_dir / "colab_sync_log.json"
            if not a.dry_run:
                summary_path.write_text(json.dumps(summary, indent=2))
                print(f"\n  Sync log written to {summary_path}")

    # Print final artifact state
    print("\n📦 Artifacts now on disk:")
    key_files = [
        "probe.npz", "spectral_head.npz", "calibration.json", "evaluation.json",
        "audio_probe.npz", "temporal_head.pt", "run_summary.json", "finetuned_encoder",
    ]
    for fn in key_files:
        p = out_dir / fn
        if p.is_dir():
            n = sum(1 for _ in p.rglob("*") if _.is_file())
            print(f"  ✅ {fn}/  ({n} files)")
        elif p.exists():
            print(f"  ✅ {fn}  ({p.stat().st_size/1024:.1f} KB)")
        else:
            print(f"  ❌ {fn}  NOT FOUND")

    # Print key metrics if calibration.json present
    cal = out_dir / "calibration.json"
    if cal.exists():
        try:
            d = json.loads(cal.read_text())
            f = d.get("final", {})
            print(f"\n📈 Model metrics (calibration.json):")
            print(f"  AUC              : {f.get('auc', 'N/A')}")
            print(f"  EER              : {f.get('eer', 'N/A')}")
            print(f"  ECE (calibrated) : {f.get('ece_after', 'N/A')}")
            print(f"  TPR @ 1% FPR    : {f.get('tpr_at_1pct_fpr', 'N/A')}")
        except Exception as e:
            print(f"  Could not parse calibration.json: {e}")

    print("\n✅ Sync complete! Restart your Streamlit app to use the new models.")
    print("   streamlit run app.py")


if __name__ == "__main__":
    main()
