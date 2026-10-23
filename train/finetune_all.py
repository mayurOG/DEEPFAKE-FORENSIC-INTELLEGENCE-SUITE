"""Fine-tune image, video and audio detectors in one command.

Orchestrates the existing modality-specific trainers:

* **Image**  – DINOv2/DINOv3 (or any HF vision encoder) feature extraction →
  linear probe → optional LayerNorm fine-tune → spectral head
* **Video**  – frame embeddings → temporal head
* **Audio**  – WavLM / HF-audio embeddings → linear probe → optional last-layer
  fine-tune of the audio classifier

Dataset layout (per modality, can share roots)::

    data/<name>/real/**.{jpg,png,mp4,wav,...}
    data/<name>/fake/**.{jpg,png,mp4,wav,...}

Recommended: keep train / calib / test disjoint (different generators when
possible). Calibration and evaluation still go through ``train.run_all`` or
``train.fit_calibration`` + ``train.evaluate``.

Examples::

    # Full stack (image + video + audio) with LayerNorm and audio fine-tune
    python -m train.finetune_all \\
        --train data/ffpp data/asvspoof \\
        --encoder facebook/dinov2-base \\
        --out artifacts \\
        --finetune-image --finetune-audio \\
        --modalities image video audio

    # Image only (fast)
    python -m train.finetune_all --train data/ffpp --encoder facebook/dinov2-base \\
        --out artifacts --modalities image

    # Audio only
    python -m train.finetune_all --train data/asvspoof --out artifacts \\
        --modalities audio --finetune-audio
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Dict, List

log = logging.getLogger("deepfake.train.finetune_all")


def _step(name: str, summary: Dict, fn):
    t = time.time()
    log.info("======== %s ========", name)
    try:
        out = fn()
        summary["steps"][name] = {
            "ok": True,
            "seconds": round(time.time() - t, 1),
            **({"result": out} if isinstance(out, dict) else {}),
        }
    except SystemExit as e:
        summary["steps"][name] = {"ok": False, "error": f"exit: {e}"}
        log.error("%s exited: %s", name, e)
    except Exception as e:
        log.exception("%s failed", name)
        summary["steps"][name] = {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "seconds": round(time.time() - t, 1),
        }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--train", required=True, nargs="+",
                    help="dataset root(s) with real/ and fake/ subfolders")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--encoder", default="facebook/dinov2-base",
                    help="vision foundation encoder (DINOv2/DINOv3/SigLIP/...)")
    ap.add_argument("--audio-encoder", default="",
                    help="WavLM / wav2vec2 id (default: microsoft/wavlm-large)")
    ap.add_argument("--audio-model", default="",
                    help="HF audio classifier for --finetune-audio")
    ap.add_argument("--modalities", nargs="+",
                    default=["image", "video", "audio"],
                    choices=["image", "video", "audio"],
                    help="which modalities to train")
    ap.add_argument("--finetune-image", action="store_true",
                    help="also run LayerNorm-only tuning of the vision encoder")
    ap.add_argument("--finetune-audio", action="store_true",
                    help="also light-fine-tune the HF audio classifier")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--epochs-audio", type=int, default=3)
    ap.add_argument("--epochs-image", type=int, default=8)
    ap.add_argument("--epochs-video", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="vision/audio batch size (use 1-2 on GPUs with <=4 GB VRAM)")
    ap.add_argument("--frames", type=int, default=4,
                    help="sampled frames per video for visual feature extraction")
    ap.add_argument("--no-faces", action="store_true",
                    help="skip face detection/cropping during training")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    os.makedirs(a.out, exist_ok=True)
    os.environ["DEEPFAKE_ARTIFACT_DIR"] = a.out
    summary: Dict = {
        "encoder": a.encoder,
        "train": a.train,
        "modalities": a.modalities,
        "steps": {},
        "training": {
            "batch_size": a.batch_size,
            "frames_per_video": a.frames,
            "epochs_image": a.epochs_image,
            "epochs_video": a.epochs_video,
            "seed": a.seed,
        },
    }
    mx = ["--max-items", str(a.max_items)] if a.max_items else []

    # ------------------------------------------------------------------ image
    if "image" in a.modalities:
        from train import features, train_probe, train_spectral

        def do_image_features():
            parts = []
            for k, d in enumerate(a.train):
                p = os.path.join(a.out, f"_img_feat_{k}.npz")
                features.main([
                    "--data", d, "--encoder", a.encoder, "--out", p,
                    "--batch-size", str(a.batch_size), "--frames", str(a.frames),
                    *([] if not a.no_faces else ["--no-faces"]), *mx,
                ])
                parts.append(np_load(p))
            merged = _merge_npz(parts)
            outp = os.path.join(a.out, "train_features.npz")
            import numpy as np
            np.savez(outp, **merged)
            for k in range(len(a.train)):
                try:
                    os.remove(os.path.join(a.out, f"_img_feat_{k}.npz"))
                except OSError:
                    pass
            return {"rows": int(len(merged["y"]))}

        _step("image_features", summary, do_image_features)
        _step("image_probe", summary, lambda: train_probe.main([
            "--train", os.path.join(a.out, "train_features.npz"),
            "--out", os.path.join(a.out, "probe.npz"),
        ]) or {})

        if a.finetune_image:
            from train import finetune_ln
            _step("image_finetune_ln", summary, lambda: finetune_ln.main([
                "--data", a.train[0], "--encoder", a.encoder, "--out", a.out,
                "--epochs", str(a.epochs_image), "--batch-size", str(a.batch_size),
                "--seed", str(a.seed),
                *([] if not a.no_faces else ["--no-faces"]), *mx,
            ]) or {})

        spec_args = sum([["--data", d] for d in a.train], []) + ["--out", a.out] + mx
        def do_spectral():
            r = train_spectral.main(spec_args)
            if isinstance(r, dict):
                return {"val_auc": r.get("val_auc")}
            return {}
        _step("spectral_head", summary, do_spectral)

    # ------------------------------------------------------------------ video
    if "video" in a.modalities:
        from train import train_temporal

        def do_temporal():
            # train_temporal currently accepts a single --data root
            results = []
            for d in a.train:
                args = ["--data", d, "--encoder", a.encoder, "--out", a.out,
                        "--epochs", str(a.epochs_video),
                        "--frames", str(a.frames),
                        "--batch-size", str(a.batch_size),
                        "--seed", str(a.seed),
                        *([] if not a.no_faces else ["--no-faces"]), *mx]
                r = train_temporal.main(args)
                results.append(r if isinstance(r, dict) else {"data": d})
            return {"runs": results}

        _step("video_temporal_head", summary, do_temporal)

    # ------------------------------------------------------------------ audio
    if "audio" in a.modalities:
        from train import train_audio

        def do_audio():
            args = sum([["--train", d] for d in a.train], [])  # train_audio uses --train
            # fix: train_audio expects --train as list of roots
            args = ["--train", *a.train, "--out", a.out]
            if a.audio_encoder:
                args += ["--encoder", a.audio_encoder]
            if a.finetune_audio:
                args += ["--finetune", "--epochs", str(a.epochs_audio),
                         "--batch-size", str(a.batch_size)]
            if a.audio_model:
                args += ["--audio-model", a.audio_model]
            if a.max_items:
                args += ["--max-items", str(a.max_items)]
            return train_audio.main(args) or {}

        _step("audio", summary, do_audio)

    # Keep this process aligned with the encoder paired with the final probe.
    image_finetune = summary["steps"].get("image_finetune_ln", {})
    tuned_encoder = image_finetune.get("result", {}).get("encoder_dir", a.encoder)
    os.environ["EMBED_MODEL_ID"] = tuned_encoder
    summary_path = os.path.join(a.out, "finetune_all_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    failed = [k for k, v in summary["steps"].items() if not v.get("ok")]
    print(json.dumps({
        "steps": {k: v.get("ok") for k, v in summary["steps"].items()},
        "failed": failed,
        "summary": summary_path,
        "hint": (
            "Next: fit calibration on a held-out set, then evaluate:\n"
            "  python -m train.fit_calibration --data data/calib --media image\n"
            "  python -m train.evaluate --data data/test --media image --out artifacts/eval.json\n"
            "Or use train.run_all for the full image/video pipeline."
        ),
    }, indent=2))
    return summary


def np_load(path):
    import numpy as np
    return np.load(path, allow_pickle=True)


def _merge_npz(parts):
    import numpy as np
    merged = {}
    for k in parts[0].files:
        if parts[0][k].ndim >= 1:
            merged[k] = np.concatenate([p[k] for p in parts])
        else:
            merged[k] = parts[0][k]
    return merged


if __name__ == "__main__":
    main()
