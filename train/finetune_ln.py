"""Parameter-efficient adaptation: tune only the LayerNorm parameters
(~0.03% of the weights) of a foundation vision encoder plus a linear head on
an L2-normalised (hyperspherical) feature space.

This follows the recipe reported to generalise best across deepfake
benchmarks (GenD, WACV 2026). Use it when the frozen linear probe is not
enough; try the probe first, it is faster and often just as good.

    python -m train.finetune_ln --data data/ffpp --encoder <model-id> \
        --epochs 3 --out artifacts

Outputs ``<out>/finetuned_encoder/`` (set EMBED_MODEL_ID to it) and
``<out>/probe.npz`` (the matching head).
"""
from __future__ import annotations

import argparse
import copy
import logging
import os
from typing import List, Sequence

import numpy as np
from PIL import Image, ImageFilter

from detector import metrics as MT
from detector import models as M
from detector.stability import _downscale, _jpeg, _noise
from train.data import Item, group_split, load_images, scan_folder
from train.features import face_crop

log = logging.getLogger("deepfake.train")


def augment(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Degradations that mimic social-media laundering, so the detector does
    not learn 'pristine = real'."""
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        img = _jpeg(img, int(rng.integers(40, 96)))
    if rng.random() < 0.3:
        img = _downscale(img, float(rng.uniform(0.4, 0.9)))
    if rng.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.5, 1.5))))
    if rng.random() < 0.2:
        img = _noise(img, float(rng.uniform(2, 6)), int(rng.integers(0, 1 << 30)))
    return img


def _pick_image(item: Item, rng) -> Image.Image:
    imgs = list(load_images(item, n_frames=4))
    return imgs[int(rng.integers(0, len(imgs)))]


def _features(model, pixel_values, torch):
    if hasattr(model, "get_image_features"):
        f = model.get_image_features(pixel_values=pixel_values)
        if not hasattr(f, "shape"):
            f = getattr(f, "pooler_output", f[0])
    else:
        o = model(pixel_values=pixel_values)
        f = getattr(o, "pooler_output", None)
        if f is None:
            f = o.last_hidden_state.mean(dim=1)
    return torch.nn.functional.normalize(f.float(), dim=-1)


def train(
    train_items: Sequence[Item], val_items: Sequence[Item], model_id: str,
    out_dir: str, epochs: int = 3, batch_size: int = 16, lr: float = 1e-5,
    head_lr: float = 1e-3, use_faces: bool = True, seed: int = 0,
    max_steps: int = 0,
) -> dict:
    import torch
    from torch import nn
    from transformers import AutoImageProcessor, AutoModel

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = M.get_device()
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    for p in model.parameters():
        p.requires_grad = False
    ln_params = []
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            for p in m.parameters():
                p.requires_grad = True
                ln_params.append(p)
    n_trainable = sum(p.numel() for p in ln_params)
    n_total = sum(p.numel() for p in model.parameters())
    log.info("tuning %d of %d parameters (%.3f%%)", n_trainable, n_total,
             100.0 * n_trainable / n_total)
    detector = M.get_face_detector() if use_faces else None

    def batch_tensors(items, aug):
        imgs = []
        for it in items:
            im = _pick_image(it, rng)
            im = face_crop(im, detector, use_faces)
            imgs.append(augment(im, rng) if aug else im)
        px = processor(images=imgs, return_tensors="pt")["pixel_values"].to(device)
        return px, torch.tensor([it.label for it in items], dtype=torch.float32, device=device)

    with torch.no_grad():
        dim = _features(model, batch_tensors(train_items[:2], False)[0], torch).shape[1]
    head = nn.Linear(dim, 1).to(device)
    n_fake = max(1, sum(i.label for i in train_items))
    pos_weight = torch.tensor([(len(train_items) - n_fake) / n_fake], device=device)
    opt = torch.optim.AdamW(
        [{"params": ln_params, "lr": lr}, {"params": head.parameters(), "lr": head_lr}],
        weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def evaluate():
        model.eval()
        ys, zs = [], []
        with torch.no_grad():
            for i in range(0, len(val_items), batch_size):
                px, y = batch_tensors(val_items[i:i + batch_size], False)
                zs += head(_features(model, px, torch)).squeeze(-1).cpu().tolist()
                ys += y.cpu().tolist()
        return MT.auc(ys, zs)

    step, history = 0, []
    best_auc, best_model_state, best_head_state = -1.0, None, None
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(train_items))
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            px, y = batch_tensors([train_items[j] for j in idx], True)
            loss = loss_fn(head(_features(model, px, torch)).squeeze(-1), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if max_steps and step >= max_steps:
                break
        auc = evaluate() if val_items else float("nan")
        history.append({"epoch": ep + 1, "loss": float(loss.item()), "val_auc": float(auc)})
        log.info("epoch %d loss %.4f val_auc %.4f", ep + 1, loss.item(), auc)
        if np.isfinite(auc) and auc > best_auc:
            best_auc = float(auc)
            best_model_state = copy.deepcopy(model.state_dict())
            best_head_state = copy.deepcopy(head.state_dict())
        if max_steps and step >= max_steps:
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        head.load_state_dict(best_head_state)
    enc_dir = os.path.join(out_dir, "finetuned_encoder")
    os.makedirs(enc_dir, exist_ok=True)
    model.save_pretrained(enc_dir)
    processor.save_pretrained(enc_dir)
    w = head.weight.detach().cpu().numpy()[0]
    np.savez(os.path.join(out_dir, "probe.npz"), w=w, b=float(head.bias.item()),
             mean=np.zeros(dim), scale=np.ones(dim), encoder=enc_dir)
    return {"trainable_params": n_trainable, "total_params": n_total,
            "history": history, "encoder_dir": enc_dir}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-faces", action="store_true")
    ap.add_argument("--max-items", type=int, default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    items = [i for i in scan_folder(a.data, max_items=a.max_items) if i.media_type in ("image", "video")]
    tr, va = group_split(items, 0.2, seed=a.seed)
    res = train(tr, va, a.encoder, a.out, a.epochs, a.batch_size, a.lr,
                a.head_lr, not a.no_faces, seed=a.seed)
    print(res)
    print(f"Set EMBED_MODEL_ID={res['encoder_dir']} to use the tuned encoder.")
    return res


if __name__ == "__main__":
    main()
