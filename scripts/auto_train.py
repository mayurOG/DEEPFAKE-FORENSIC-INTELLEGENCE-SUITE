"""Automated large-scale dataset preparation, LayerNorm fine-tuning & calibration pipeline.

This script:
1. Prepares a balanced, large-scale deepfake vs real face dataset into:
   - data/train/ (real & fake, for representation learning & probe training)
   - data/calib/ (real & fake, for conformal risk calibration & OOD envelope)
   - data/test/  (real & fake, for strict out-of-sample generalization evaluation)
2. Fine-tunes the foundation vision encoder (DINOv2) via parameter-efficient
   LayerNorm adaptation (GenD recipe, WACV 2026) + linear probe on hyperspherical embeddings.
3. Fits the frequency spectral expert (artifacts/spectral_head.npz).
4. Fits the Conformal Risk Calibration bundle & Mahalanobis OOD model (artifacts/calibration.json, artifacts/ood_image.npz).
5. Evaluates accuracy, AUC, EER, and calibration on holdout test data.
6. Persists the fine-tuned checkpoint to artifacts/finetuned_encoder/ and updates .env.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict

from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("deepfake.auto_train")


def scan_existing_count(folder: Path) -> int:
    """Return count of valid image files in a folder."""
    if not folder.exists():
        return 0
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    count = 0
    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() in valid_exts and f.stat().st_size > 0:
            count += 1
    return count


def download_data(
    dataset_name: str = "Hemg/deepfake-and-real-images",
    train_per_class: int = 500,
    calib_per_class: int = 150,
    test_per_class: int = 150,
    base_dir: str = "data",
) -> Dict:
    """Download and prepare a balanced dataset across train, calib, and test splits."""
    log.info("==================================================================")
    log.info("STEP 1: PREPARING LARGE-SCALE BALANCED FORENSIC DATASET")
    log.info("Dataset source: %s", dataset_name)
    log.info("Targets per class -> Train: %d | Calib: %d | Test: %d",
             train_per_class, calib_per_class, test_per_class)
    log.info("==================================================================")

    root = Path(base_dir)
    splits = {
        "train": {
            "real": root / "train" / "real",
            "fake": root / "train" / "fake",
            "target": train_per_class,
        },
        "calib": {
            "real": root / "calib" / "real",
            "fake": root / "calib" / "fake",
            "target": calib_per_class,
        },
        "test": {
            "real": root / "test" / "real",
            "fake": root / "test" / "fake",
            "target": test_per_class,
        },
    }

    for split_cfg in splits.values():
        split_cfg["real"].mkdir(parents=True, exist_ok=True)
        split_cfg["fake"].mkdir(parents=True, exist_ok=True)

    counts = {
        s: {
            "real": scan_existing_count(splits[s]["real"]),
            "fake": scan_existing_count(splits[s]["fake"]),
        }
        for s in splits
    }

    log.info("Existing valid samples on disk:")
    for s in splits:
        log.info("  - %s: %d real, %d fake (target: %d each)",
                 s, counts[s]["real"], counts[s]["fake"], splits[s]["target"])

    all_fulfilled = all(
        counts[s]["real"] >= splits[s]["target"] and counts[s]["fake"] >= splits[s]["target"]
        for s in splits
    )

    if all_fulfilled:
        log.info("All target splits are already satisfied. Skipping download.")
    else:
        log.info("Streaming missing samples from Hugging Face: %s ...", dataset_name)
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split="train", streaming=True)
        saved_new = 0

        for i, item in enumerate(ds):
            label_int = item.get("label")
            # In Hemg/deepfake-and-real-images: 0 = Fake, 1 = Real
            label_str = "fake" if label_int == 0 else "real"

            # Determine which split needs this class
            target_split = None
            for s_name in ("train", "calib", "test"):
                if counts[s_name][label_str] < splits[s_name]["target"]:
                    target_split = s_name
                    break

            if target_split is None:
                # Check if everything is full
                if all(
                    counts[s]["real"] >= splits[s]["target"] and counts[s]["fake"] >= splits[s]["target"]
                    for s in splits
                ):
                    break
                continue

            raw_img = item.get("image")
            if raw_img is None:
                continue

            try:
                img: Image.Image = raw_img.convert("RGB")
                if img.width < 64 or img.height < 64:
                    continue
            except Exception as ex:
                log.warning("Skipping unreadable image at index %d: %s", i, ex)
                continue

            counts[target_split][label_str] += 1
            idx = counts[target_split][label_str]
            out_path = splits[target_split][label_str] / f"{label_str}_{idx:05d}.jpg"

            # Save with high-quality JPEG
            img.save(out_path, format="JPEG", quality=95)
            saved_new += 1

            total_target = sum(splits[s]["target"] * 2 for s in splits)
            current_total = sum(counts[s]["real"] + counts[s]["fake"] for s in splits)
            if saved_new % 50 == 0 or current_total >= total_target:
                log.info("Progress: %d / %d total images collected (+%d new)",
                         current_total, total_target, saved_new)

            if all(
                counts[s]["real"] >= splits[s]["target"] and counts[s]["fake"] >= splits[s]["target"]
                for s in splits
            ):
                break

    manifest = {
        "dataset_name": dataset_name,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "splits": {
            s: {
                "real": counts[s]["real"],
                "fake": counts[s]["fake"],
                "total": counts[s]["real"] + counts[s]["fake"],
            }
            for s in splits
        },
        "total_images": sum(counts[s]["real"] + counts[s]["fake"] for s in splits),
    }

    manifest_path = root / "dataset_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Dataset manifest written to %s", manifest_path)
    return manifest


def update_env_file(key: str, value: str, env_path: str = ".env"):
    """Update or insert a key-value pair in .env without modifying other lines."""
    env_file = Path(env_path)
    if not env_file.exists():
        env_file.write_text(f"{key}={value}\n")
        log.info("Created %s with %s=%s", env_path, key, value)
        return

    content = env_file.read_text(encoding="utf-8")
    pattern = rf"^{re.escape(key)}=.*$"
    if re.search(pattern, content, flags=re.MULTILINE):
        new_content = re.sub(pattern, f"{key}={value}", content, flags=re.MULTILINE)
    else:
        new_content = content.rstrip() + f"\n{key}={value}\n"

    env_file.write_text(new_content, encoding="utf-8")
    log.info("Updated %s: %s=%s", env_path, key, value)


def run_pipeline(
    encoder: str = "facebook/dinov2-base",
    epochs: int = 4,
    batch_size: int = 8,
    out_dir: str = "artifacts",
    finetune: bool = True,
) -> Dict:
    """Execute complete training, LayerNorm adaptation, calibration and evaluation."""
    log.info("==================================================================")
    log.info("STEP 2: RUNNING TRAINING, FINETUNING & CONFORMAL CALIBRATION")
    log.info("Encoder backbone: %s", encoder)
    log.info("LayerNorm Fine-tuning: %s | Epochs: %d | Batch Size: %d", finetune, epochs, batch_size)
    log.info("Artifacts output directory: %s", out_dir)
    log.info("==================================================================")

    from train import run_all

    args = [
        "--train", "data/train",
        "--calib", "data/calib",
        "--test", "data/test",
        "--encoder", encoder,
        "--out", out_dir,
        "--media", "image",
        "--batch-size", str(batch_size),
        "--epochs", str(epochs),
    ]
    if finetune:
        args.append("--finetune")

    log.info("Executing run_all with args: %s", " ".join(args))
    summary = run_all.main(args)

    # Persist fine-tuned encoder reference in .env
    tuned_enc_path = os.path.join(out_dir, "finetuned_encoder")
    if finetune and os.path.isdir(tuned_enc_path):
        update_env_file("EMBED_MODEL_ID", tuned_enc_path)
        log.info("Set EMBED_MODEL_ID=%s in .env for active production inference.", tuned_enc_path)

    # Summarize final metrics
    eval_file = os.path.join(out_dir, "evaluation.json")
    if os.path.exists(eval_file):
        try:
            with open(eval_file) as f:
                ev = json.load(f)
            log.info("==================================================================")
            log.info("HOLDOUT TEST EVALUATION RESULTS:")
            for d_name, d_res in ev.get("datasets", {}).items():
                bb = d_res.get("backbone", {})
                cas = d_res.get("cascade", {})
                log.info("  Dataset '%s':", d_name)
                log.info("    Backbone AUC: %.4f | EER: %.4f", bb.get("auc", 0), bb.get("eer", 0))
                log.info("    Cascade  AUC: %.4f | EER: %.4f | ECE: %.4f",
                         cas.get("auc", 0), cas.get("eer", 0), cas.get("ece", 0))
            log.info("==================================================================")
        except Exception as e:
            log.warning("Could not parse evaluation.json: %s", e)

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-samples", type=int, default=500,
                        help="Number of real & fake samples for training (default: 500 each = 1000 total)")
    parser.add_argument("--calib-samples", type=int, default=150,
                        help="Number of real & fake samples for calibration (default: 150 each = 300 total)")
    parser.add_argument("--test-samples", type=int, default=150,
                        help="Number of real & fake samples for holdout testing (default: 150 each = 300 total)")
    parser.add_argument("--epochs", type=int, default=4,
                        help="Epochs for LayerNorm fine-tuning (default: 4)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--encoder", default="facebook/dinov2-base",
                        help="Vision encoder model ID")
    parser.add_argument("--dataset", default="Hemg/deepfake-and-real-images",
                        help="Hugging Face dataset identifier")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip dataset download and use existing files in data/")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training phase")
    parser.add_argument("--no-finetune", action="store_true",
                        help="Disable LayerNorm fine-tuning (run linear probe only)")
    parser.add_argument("--out", default="artifacts",
                        help="Output directory for fine-tuned weights and calibration")

    args = parser.parse_args()

    if not args.skip_download:
        download_data(
            dataset_name=args.dataset,
            train_per_class=args.train_samples,
            calib_per_class=args.calib_samples,
            test_per_class=args.test_samples,
        )

    if not args.skip_train:
        run_pipeline(
            encoder=args.encoder,
            epochs=args.epochs,
            batch_size=args.batch_size,
            out_dir=args.out,
            finetune=not args.no_finetune,
        )

    log.info("🎉 Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()
