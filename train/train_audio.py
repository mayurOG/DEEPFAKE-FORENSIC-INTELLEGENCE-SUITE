"""Fine-tune / train audio deepfake detectors.

Two modes (both produce artifacts consumable by the cascade):

1. **Frozen encoder + linear head** (default, recommended for generalisation)
       Extract WavLM / HF-audio embeddings → logistic probe → ``audio_probe.npz``

2. **Light last-layer fine-tune** of a Hugging Face audio classifier
       ``--finetune`` updates the classification head (and optionally the last
       transformer block) of ``AUDIO_MODEL_ID``.

Dataset layout (same as image/video)::

    data/<dataset>/real/**.{wav,mp3,flac,m4a,ogg,...}
    data/<dataset>/fake/**.{wav,mp3,flac,m4a,ogg,...}

Video files that contain an audio track are also accepted when scanning with
``--media audio`` (the track is extracted).

Examples::

    # Probe on frozen WavLM features
    python -m train.train_audio --train data/asvspoof --out artifacts \\
        --encoder microsoft/wavlm-large

    # Light fine-tune of the HF audio classifier
    python -m train.train_audio --train data/asvspoof --out artifacts \\
        --finetune --epochs 5 --batch-size 8

    # End-to-end with calibration (from run_all or finetune_all)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from detector import metrics as MT
from detector.config import get_settings
from detector.models import l2_normalize
from train.data import Item, group_split, load_waveform, scan_folder

log = logging.getLogger("deepfake.train.audio")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
class AudioEncoder:
    """WavLM / wav2vec2 / HuBERT-style encoder that returns pooled embeddings."""

    def __init__(self, model_id: str, batch_size: int = 4, sr: int = 16000):
        self.model_id = model_id
        self.batch_size = batch_size
        self.sr = sr
        self._ready = False

    def _load(self):
        if self._ready:
            return
        import torch
        from transformers import AutoFeatureExtractor, AutoModel

        tok = get_settings().hf_token or None
        self.extractor = AutoFeatureExtractor.from_pretrained(
            self.model_id, token=tok
        )
        self.model = AutoModel.from_pretrained(self.model_id, token=tok).eval()
        from detector.models import get_device
        self.device = get_device()
        self.model.to(self.device)
        self.torch = torch
        self._ready = True

    def embed(self, waves: Sequence[np.ndarray]) -> np.ndarray:
        """waves: list of mono float32 arrays at self.sr. Returns (N, D)."""
        self._load()
        torch = self.torch
        out = []
        for i in range(0, len(waves), self.batch_size):
            batch = [np.asarray(w, dtype=np.float32) for w in waves[i:i + self.batch_size]]
            inputs = self.extractor(
                batch, sampling_rate=self.sr, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                hidden = self.model(**inputs).last_hidden_state
            # mean-pool over time
            emb = hidden.mean(dim=1).detach().float().cpu().numpy()
            out.append(emb)
        return np.concatenate(out, axis=0) if out else np.zeros((0, 1))


def extract_audio_features(
    items: Sequence[Item],
    encoder: AudioEncoder,
    max_seconds: float = 15.0,
) -> Dict[str, np.ndarray]:
    embs, ys, gs, ps = [], [], [], []
    for k, it in enumerate(items):
        wave = load_waveform(it, sr=encoder.sr, max_seconds=max_seconds)
        if wave is None:
            log.debug("no audio: %s", it.path)
            continue
        try:
            e = encoder.embed([wave])
        except Exception as ex:
            log.warning("skip %s: %s", it.path, ex)
            continue
        embs.append(e[0])
        ys.append(it.label)
        gs.append(it.group)
        ps.append(it.path)
        if (k + 1) % 50 == 0:
            log.info("audio features %d/%d", k + 1, len(items))
    if not embs:
        raise RuntimeError(
            "No audio features extracted. Check dataset layout "
            "(data/<set>/real|fake/**.wav) and that files contain speech."
        )
    return {
        "emb": np.stack(embs).astype(np.float32),
        "y": np.asarray(ys, dtype=np.int64),
        "groups": np.asarray(gs),
        "paths": np.asarray(ps),
        "encoder": np.asarray(encoder.model_id),
    }


# ---------------------------------------------------------------------------
# Linear probe (same spirit as train_probe)
# ---------------------------------------------------------------------------
def fit_audio_probe(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    Cs=(0.01, 0.1, 1.0, 10.0),
    seed: int = 0,
) -> Dict:
    from sklearn.linear_model import LogisticRegression

    Xn = l2_normalize(np.asarray(X, dtype=np.float64))
    g = groups if groups is not None else np.arange(len(y))
    # group-aware val split
    ug = np.unique(g)
    rng = np.random.default_rng(seed)
    rng.shuffle(ug)
    n_val = max(1, int(len(ug) * 0.2))
    val_g = set(ug[:n_val])
    va = np.array([gg in val_g for gg in g])
    tr = ~va
    Xt, yt, Xv, yv = Xn[tr], y[tr], Xn[va], y[va]
    mean = Xt.mean(axis=0)
    scale = Xt.std(axis=0) + 1e-6
    Xt_s, Xv_s = (Xt - mean) / scale, (Xv - mean) / scale
    best = None
    for C in Cs:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
        clf.fit(Xt_s, yt)
        auc = MT.auc(yv, clf.decision_function(Xv_s)) if len(np.unique(yv)) > 1 else float("nan")
        score = auc if auc == auc else -1.0
        if best is None or score > best[0]:
            best = (score, C, clf, auc)
    _, C, clf, auc = best
    # refit on all
    mean = Xn.mean(axis=0)
    scale = Xn.std(axis=0) + 1e-6
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
    clf.fit((Xn - mean) / scale, y)
    return {
        "w": clf.coef_.ravel().astype(np.float64),
        "b": float(clf.intercept_[0]),
        "mean": mean.astype(np.float64),
        "scale": scale.astype(np.float64),
        "C": float(C),
        "val_auc": float(auc) if auc == auc else None,
        "n_train": int(len(y)),
        "kind": "audio_probe",
    }


def save_audio_probe(path: str, probe: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, **{k: v for k, v in probe.items() if k != "kind"})
    meta = {k: probe[k] for k in ("C", "val_auc", "n_train", "kind") if k in probe}
    with open(path + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info("wrote %s (val_auc=%s)", path, probe.get("val_auc"))


# ---------------------------------------------------------------------------
# Light HF classifier fine-tune
# ---------------------------------------------------------------------------
def finetune_hf_audio(
    items: Sequence[Item],
    model_id: str,
    out_dir: str,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 2e-5,
    max_seconds: float = 10.0,
    sr: int = 16000,
    seed: int = 0,
) -> Dict:
    """Fine-tune the classification head of a HF audio model. Saves to
    ``out_dir/audio_finetuned/`` and returns metrics."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoFeatureExtractor,
        AutoModelForAudioClassification,
        get_linear_schedule_with_warmup,
    )

    from detector.models import get_device, find_fake_index

    train_items, val_items = group_split(list(items), 0.2, seed)
    if not train_items or not val_items:
        raise RuntimeError("Need both train and val groups with audio.")

    tok = get_settings().hf_token or None
    extractor = AutoFeatureExtractor.from_pretrained(model_id, token=tok)
    model = AutoModelForAudioClassification.from_pretrained(model_id, token=tok)
    device = get_device()
    model.to(device)
    fake_idx, _ = find_fake_index(model.config.id2label)

    class WaveDataset(Dataset):
        def __init__(self, its):
            self.items = its

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            it = self.items[i]
            wave = load_waveform(it, sr=sr, max_seconds=max_seconds)
            if wave is None:
                wave = np.zeros(sr, dtype=np.float32)
            return wave, it.label

    def collate(batch):
        waves, labels = zip(*batch)
        inputs = extractor(
            list(waves), sampling_rate=sr, return_tensors="pt", padding=True
        )
        inputs["labels"] = torch.tensor(labels, dtype=torch.long)
        return inputs

    train_loader = DataLoader(
        WaveDataset(train_items), batch_size=batch_size, shuffle=True,
        collate_fn=collate, num_workers=0,
    )
    val_loader = DataLoader(
        WaveDataset(val_items), batch_size=batch_size, shuffle=False,
        collate_fn=collate, num_workers=0,
    )

    # Freeze everything except classifier head (and optionally last block)
    for name, p in model.named_parameters():
        p.requires_grad = (
            "classifier" in name
            or "projector" in name
            or "score" in name
            or name.startswith("encoder.layers") and ".10." in name  # last block heuristic
            or name.startswith("encoder.layer") and ".11." in name
        )

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01
    )
    total_steps = max(1, epochs * len(train_loader))
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
    criterion = nn.CrossEntropyLoss()

    best_auc, best_state = -1.0, None
    history = []
    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            out = model(**batch)
            loss = criterion(out.logits, labels)
            loss.backward()
            opt.step()
            sched.step()
            tr_loss += loss.item()
        # validation
        model.eval()
        logits_all, y_all = [], []
        with torch.inference_mode():
            for batch in val_loader:
                labels = batch.pop("labels")
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                logits_all.append(out.logits.float().cpu().numpy())
                y_all.append(labels.numpy())
        logits = np.concatenate(logits_all)
        yv = np.concatenate(y_all)
        # fake log-odds
        if logits.shape[1] > 1:
            from detector.models import fake_logit_from_logits
            z = fake_logit_from_logits(logits, fake_idx)
        else:
            z = logits[:, 0]
        auc = MT.auc(yv, z) if len(np.unique(yv)) > 1 else float("nan")
        history.append({"epoch": ep + 1, "train_loss": tr_loss / max(1, len(train_loader)),
                        "val_auc": float(auc) if auc == auc else None})
        log.info("epoch %d  loss=%.4f  val_auc=%s", ep + 1, history[-1]["train_loss"], history[-1]["val_auc"])
        if auc == auc and auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    save_dir = os.path.join(out_dir, "audio_finetuned")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    extractor.save_pretrained(save_dir)
    meta = {
        "base_model": model_id,
        "val_auc": float(best_auc) if best_auc >= 0 else None,
        "epochs": epochs,
        "n_train": len(train_items),
        "n_val": len(val_items),
        "history": history,
        "kind": "audio_finetune",
    }
    with open(os.path.join(save_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log.info("saved fine-tuned audio model to %s (val_auc=%s)", save_dir, meta["val_auc"])
    return meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, nargs="+",
                    help="one or more dataset roots (real/ and fake/ subdirs)")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--encoder", default="",
                    help="WavLM / wav2vec2 id for frozen-feature probe "
                         "(default: settings.wavlm_model_id or microsoft/wavlm-large)")
    ap.add_argument("--finetune", action="store_true",
                    help="fine-tune HF audio classifier instead of (or after) probe")
    ap.add_argument("--audio-model", default="",
                    help="HF audio classifier id for --finetune "
                         "(default: settings.audio_model_id)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-seconds", type=float, default=15.0)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    os.makedirs(a.out, exist_ok=True)
    items: List[Item] = []
    for root in a.train:
        items.extend(scan_folder(root, media="audio", max_items=a.max_items, seed=a.seed))
        # also accept video files that have audio when the folder is mixed
        if not any(it.media_type == "audio" for it in items):
            items.extend(scan_folder(root, media="video", max_items=a.max_items, seed=a.seed))
    if not items:
        raise SystemExit(
            "No audio (or video-with-audio) files found under --train. "
            "Expected data/<set>/real|fake/**.{wav,mp3,...}"
        )
    n_real = sum(1 for it in items if it.label == 0)
    n_fake = sum(1 for it in items if it.label == 1)
    log.info("loaded %d items (%d real, %d fake) from %s",
             len(items), n_real, n_fake, a.train)
    if n_real == 0 or n_fake == 0:
        raise SystemExit("Need both real and fake classes.")

    results = {}

    # --- Probe path ---
    enc_id = a.encoder or get_settings().wavlm_model_id or "microsoft/wavlm-large"
    log.info("extracting features with %s", enc_id)
    encoder = AudioEncoder(enc_id, batch_size=max(1, a.batch_size // 2))
    feats = extract_audio_features(items, encoder, max_seconds=a.max_seconds)
    feat_path = os.path.join(a.out, "audio_features.npz")
    np.savez(feat_path, **feats)
    log.info("cached features → %s  shape=%s", feat_path, feats["emb"].shape)

    probe = fit_audio_probe(feats["emb"], feats["y"], feats["groups"], seed=a.seed)
    probe_path = os.path.join(a.out, "audio_probe.npz")
    save_audio_probe(probe_path, probe)
    results["probe"] = {"path": probe_path, "val_auc": probe.get("val_auc")}

    # --- Optional HF fine-tune ---
    if a.finetune:
        model_id = a.audio_model or get_settings().audio_model_id
        log.info("fine-tuning %s", model_id)
        meta = finetune_hf_audio(
            items, model_id, a.out,
            epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
            max_seconds=min(a.max_seconds, 10.0), seed=a.seed,
        )
        results["finetune"] = meta
        # Point the cascade at the fine-tuned checkpoint
        log.info(
            "To use the fine-tuned model set:\n"
            "  export AUDIO_MODEL_ID=%s",
            os.path.join(a.out, "audio_finetuned"),
        )

    summary_path = os.path.join(a.out, "audio_train_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    main()
