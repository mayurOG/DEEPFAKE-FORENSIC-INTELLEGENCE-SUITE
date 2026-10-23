"""Explanations that are checked, not just drawn.

* ``occlusion_map`` - model-agnostic importance map: how much does the fake
  score drop when a region is hidden? Works with any backbone.
* ``deletion_faithfulness`` - tests whether that map is *faithful*: deleting
  the regions it ranks highest must lower the fake score more than deleting
  random regions. The score is shown next to the heatmap, so a decorative
  but wrong map is visibly flagged.
* ``build_narrative`` - a plain-language summary generated from measured
  fields only (no free-text model), so it cannot hallucinate artifacts.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

ScoreFn = Callable[[List[Image.Image]], np.ndarray]


def _mask_cell(arr: np.ndarray, i: int, j: int, grid: int, fill: np.ndarray):
    H, W = arr.shape[:2]
    y0, y1 = i * H // grid, (i + 1) * H // grid
    x0, x1 = j * W // grid, (j + 1) * W // grid
    arr[y0:y1, x0:x1] = fill


def occlusion_map(score_fn: ScoreFn, img: Image.Image, grid: int = 6) -> Dict:
    """Importance[i, j] = base_logit - logit_with_cell_hidden. Positive means
    the region supports a *fake* call."""
    base_img = img.convert("RGB")
    arr = np.asarray(base_img).copy()
    fill = arr.reshape(-1, 3).mean(axis=0).astype(arr.dtype)
    variants = []
    for i in range(grid):
        for j in range(grid):
            a = arr.copy()
            _mask_cell(a, i, j, grid, fill)
            variants.append(Image.fromarray(a))
    scores = np.asarray(score_fn([base_img] + variants), dtype=np.float64)
    base = float(scores[0])
    imp = (base - scores[1:]).reshape(grid, grid)
    return {"base_logit": base, "importance": imp, "grid": grid}


def overlay_heatmap(
    img: Image.Image, importance: np.ndarray, alpha: float = 0.55
) -> Image.Image:
    """Red overlay on regions that push the score towards 'fake'."""
    pos = np.clip(importance, 0, None)
    if pos.max() > 0:
        pos = pos / pos.max()
    heat = Image.fromarray((pos * 255).astype(np.uint8)).resize(
        img.size, Image.BICUBIC
    )
    h = np.asarray(heat, dtype=np.float32) / 255.0
    base = np.asarray(img.convert("RGB"), dtype=np.float32)
    red = np.zeros_like(base)
    red[..., 0] = 255.0
    out = base * (1 - alpha * h[..., None]) + red * (alpha * h[..., None])
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def deletion_faithfulness(
    score_fn: ScoreFn, img: Image.Image, importance: np.ndarray,
    fractions: Sequence[float] = (0.1, 0.2, 0.35), n_random: int = 6,
    seed: int = 0,
) -> Dict:
    """Guided deletion vs random deletion.

    faithfulness in [-1, 1]: positive = hiding the map's top regions lowers
    the fake score more than hiding random regions (the map is informative);
    ~0 = no better than chance; negative = misleading. ``None`` when the
    image is not clearly scored as fake (nothing to explain).
    """
    grid = importance.shape[0]
    cells = grid * grid
    rgb = img.convert("RGB")
    arr = np.asarray(rgb).copy()
    fill = arr.reshape(-1, 3).mean(axis=0).astype(arr.dtype)
    order = np.argsort(-importance.reshape(-1))
    rng = np.random.default_rng(seed)

    def delete(cell_ids):
        a = arr.copy()
        for c in cell_ids:
            _mask_cell(a, c // grid, c % grid, grid, fill)
        return Image.fromarray(a)

    guided, rand = [], []
    for f in fractions:
        k = max(1, int(round(f * cells)))
        guided.append(delete(order[:k]))
        for _ in range(n_random):
            rand.append(delete(rng.choice(cells, size=k, replace=False)))
    scores = np.asarray(score_fn([rgb] + guided + rand), dtype=np.float64)
    base = float(scores[0])
    if base < 0.5:
        return {"faithfulness": None, "reason": "image not scored as fake"}
    g = scores[1:1 + len(guided)]
    r = scores[1 + len(guided):].reshape(len(fractions), n_random).mean(axis=1)
    guided_drop = float(np.mean(base - g))
    random_drop = float(np.mean(base - r))
    f = (guided_drop - random_drop) / (abs(base) + 1e-6)
    return {
        "faithfulness": float(np.clip(f, -1.0, 1.0)),
        "guided_drop": guided_drop, "random_drop": random_drop,
        "base_logit": base,
    }


# --------------------------------------------------------------------------
# multi-scale occlusion + counterfactual evidence
# --------------------------------------------------------------------------
def _resize_nn(a: np.ndarray, n: int) -> np.ndarray:
    idx = (np.arange(n) * a.shape[0] / n).astype(int)
    return a[np.ix_(idx, idx)]


def _pool(a: np.ndarray, g: int) -> np.ndarray:
    n = a.shape[0]
    edges = np.linspace(0, n, g + 1).astype(int)
    out = np.zeros((g, g))
    for i in range(g):
        for j in range(g):
            out[i, j] = a[edges[i]:edges[i + 1], edges[j]:edges[j + 1]].mean()
    return out


def multiscale_occlusion(score_fn: ScoreFn, img: Image.Image,
                         grids=(3, 6, 9), res: int = 18) -> Dict:
    """Occlusion at several window sizes, fused on a common ``res x res``
    grid. Coarse windows find the big structures (a whole swapped face),
    fine windows localise small ones (an edited eye); averaging the
    per-scale-normalised maps is less sensitive to the choice of one grid
    than a single-scale map."""
    maps, base = [], None
    for g in grids:
        om = occlusion_map(score_fn, img, grid=g)
        base = om["base_logit"] if base is None else base
        m = om["importance"]
        scale = np.abs(m).max()
        maps.append(_resize_nn(m / scale if scale > 0 else m, res))
    fused = np.mean(maps, axis=0)
    return {"base_logit": base, "importance": fused, "grid": res,
            "scales": list(grids)}


def counterfactual_evidence(
    score_fn: ScoreFn, img: Image.Image, importance: np.ndarray,
    threshold: float = 0.0, grid: int = 6, max_fraction: float = 0.6,
) -> Dict:
    """Minimal-evidence analysis on a ``grid x grid`` cell grid.

    * necessity: hide the top-ranked cells one by one; ``flip_fraction`` is
      the smallest share of the image whose removal drops the score below
      ``threshold`` (None if it never does within ``max_fraction``). Small =
      the call rests on a compact region; None = diffuse or unremovable
      evidence.
    * sufficiency: keep only the top-ranked cells (everything else hidden);
      ``sufficient_fraction`` is the smallest share that alone keeps the score
      above ``threshold``. Small = a compact region is enough on its own.
    Only meaningful for a Fake-leaning input (base score >= threshold)."""
    base_img = img.convert("RGB")
    arr = np.asarray(base_img).copy()
    fill = arr.reshape(-1, 3).mean(axis=0).astype(arr.dtype)
    imp = _pool(np.asarray(importance, dtype=np.float64), grid) \
        if np.asarray(importance).shape[0] != grid else np.asarray(importance, dtype=np.float64)
    order = np.argsort(-imp.reshape(-1))
    total = grid * grid
    kmax = max(1, int(round(total * max_fraction)))
    removal, keep_only = [], []
    for k in range(1, kmax + 1):
        a = arr.copy()
        for c in order[:k]:
            _mask_cell(a, int(c // grid), int(c % grid), grid, fill)
        removal.append(Image.fromarray(a))
        b = np.broadcast_to(fill, arr.shape).copy()
        for c in order[:k]:
            i, j = int(c // grid), int(c % grid)
            H, W = arr.shape[:2]
            y0, y1, x0, x1 = i * H // grid, (i + 1) * H // grid, j * W // grid, (j + 1) * W // grid
            b[y0:y1, x0:x1] = arr[y0:y1, x0:x1]
        keep_only.append(Image.fromarray(b))
    scores = np.asarray(score_fn([base_img] + removal + keep_only), dtype=np.float64)
    base = float(scores[0])
    rem, keep = scores[1:1 + kmax], scores[1 + kmax:]
    out = {"base_logit": base, "threshold": threshold, "grid": grid,
           "applicable": bool(base >= threshold),
           "removal_curve": [round(float(x), 3) for x in rem],
           "keep_curve": [round(float(x), 3) for x in keep],
           "flip_fraction": None, "sufficient_fraction": None}
    if not out["applicable"]:
        return out
    below = np.where(rem < threshold)[0]
    if len(below):
        out["flip_fraction"] = float((below[0] + 1) / total)
    above = np.where(keep >= threshold)[0]
    if len(above):
        out["sufficient_fraction"] = float((above[0] + 1) / total)
    return out


def _direction(logit: float) -> str:
    if logit > 0.5:
        return "points to manipulation"
    if logit < -0.5:
        return "points to authentic"
    return "is neutral"


def build_narrative(r: Dict) -> str:
    """Plain-language explanation built only from measured fields of an
    ``AnalysisResult.to_dict()``."""
    lines: List[str] = []
    verdict = r.get("verdict", "Inconclusive")
    p = r.get("p_fake")
    cal = r.get("calibrated", False)
    if not isinstance(p, (int, float)):
        pstr = "n/a"
    elif p > 0.99:
        pstr = ">99%"
    elif p < 0.01:
        pstr = "<1%"
    else:
        pstr = f"{p * 100:.0f}%"
    if verdict == "Inconclusive":
        lines.append(
            f"The system could not reach a decision it can stand behind "
            f"(estimated chance of manipulation: {pstr})."
        )
    else:
        lines.append(f"Verdict: {verdict} (estimated chance of manipulation: {pstr}).")
    if not cal:
        lines.append(
            "The detector has not been calibrated on your data, so these "
            "probabilities are rough guides, not guarantees."
        )
    if r.get("ood") is True:
        lines.append(
            "The media looks unlike the data the detector was calibrated on, "
            "so the result is treated with extra caution."
        )
    tier = r.get("tier")
    if tier == "tier1_early_exit":
        lines.append("The fast screen was confident, so no further experts were run.")
    used = [e for e in r.get("evidence", []) if e.get("status") == "used"]
    for e in sorted(used, key=lambda e: -abs(e.get("logit") or 0)):
        lines.append(
            f"- {e['name']}: {_direction(e.get('logit') or 0.0)} "
            f"(reliability {e.get('reliability', 0):.2f}"
            f"{', heuristic' if e.get('heuristic') else ''})."
        )
    skipped = [e for e in r.get("evidence", []) if e.get("status") == "skipped"]
    for e in skipped:
        lines.append(f"- {e['name']}: not used ({e.get('reason', 'unavailable')}).")
    st = r.get("stability")
    if st and st.get("stability_score") is not None:
        lines.append(
            f"Re-encoding stability: {st['stability_score'] * 100:.0f}% of "
            f"degraded copies kept the same call."
        )
    g = r.get("graph") or {}
    gf = g.get("fusion") or {}
    if gf.get("n_used"):
        lines.append(
            f"Evidence sufficiency {gf.get('sufficiency', 0):.2f} "
            f"({gf['n_used']} sources, about {gf.get('n_eff', 0):.1f} independent).")
    for c in (g.get("counterfactuals") or [])[:2]:
        if c.get("pivotal") and c.get("flips_decision"):
            lines.append(f"- Pivotal: without '{c['removed']}' the call would flip.")
    for c in (g.get("contradictions") or [])[:2]:
        lines.append(f"- Contradiction: {c['experts'][0]} and {c['experts'][1]} disagree "
                     f"(strength {c['strength']:.2f}).")
    for sfm in (g.get("shared_failure_modes") or [])[:2]:
        lines.append(f"- {sfm['experts'][0]} and {sfm['experts'][1]} may share a failure "
                     f"mode (rho {sfm['rho']:.2f}), so their agreement counts for less.")
    inv = r.get("investigation") or {}
    if inv.get("policy"):
        ran = [s["test"] for s in inv.get("steps", [])]
        lines.append(f"Investigation: ran {', '.join(ran) if ran else 'no extra tests'}; "
                     f"stopped because {inv.get('stop_reason', 'n/a').replace('_', ' ')}.")
    rb = r.get("robustness")
    if rb and rb.get("fragility") is not None:
        fr = rb.get("fragile_to") or []
        lines.append(f"Transformation robustness: fragility {rb['fragility']:.2f}"
                     + (f"; the call flips under {', '.join(fr)}." if fr else "; no flips."))
    cm = r.get("contradiction_map") or {}
    if cm.get("intervals"):
        spans = ", ".join(f"{i['t0']:.1f}-{i['t1']:.1f}s" for i in cm["intervals"][:3])
        lines.append(f"Audio and lip motion disagree at {spans}.")
    elif cm.get("status") == "ok" and cm.get("benign_latency"):
        lines.append("Audio-visual timing is consistent (a constant offset is treated as benign).")
    if r.get("abstention_reason"):
        lines.append(f"Abstention reason: {r['abstention_reason']}.")
    un = r.get("uncertainty") or {}
    if un.get("view_spread") is not None and un.get("tta_views", 1) > 1:
        flag = ("the views disagreed on the call" if un.get("sign_flip")
                else "the views agreed")
        lines.append(f"Consistency under mirror/crop views: spread "
                     f"{un['view_spread']:.2f} logits; {flag}.")
    ex = r.get("explanation") or {}
    cf = ex.get("counterfactual") or {}
    if cf.get("applicable"):
        if cf.get("flip_fraction") is not None:
            lines.append(
                f"Counterfactual: hiding the {cf['flip_fraction'] * 100:.0f}% most "
                f"influential regions flips the score away from 'fake'.")
        else:
            lines.append("Counterfactual: hiding the most influential regions does "
                         "not flip the score, so the evidence is spread out.")
        if cf.get("sufficient_fraction") is not None:
            lines.append(
                f"Those top {cf['sufficient_fraction'] * 100:.0f}% of regions are "
                f"enough on their own to keep the 'fake' score.")
    if ex.get("faithfulness") is not None:
        lines.append(
            f"Heatmap faithfulness check: {ex['faithfulness']:+.2f} "
            f"(positive = the highlighted regions really drive the score)."
        )
    for note in r.get("quality", {}).get("notes", []):
        lines.append(f"Note: {note}")
    return "\n".join(lines)
