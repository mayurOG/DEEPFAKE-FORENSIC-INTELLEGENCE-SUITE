"""Black-box evasion stress test over a folder of fake images.

    python -m train.evaluate_adversarial --fake data/test/fake --n 30 \\
        --out artifacts/adversarial.json

Reports flip rate vs L_inf budget. A lower bound on fragility only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from PIL import Image

from detector.adversarial import robustness_curve
from detector.pipeline import get_pipeline
from scipy.special import logit as _logit


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--eps", default="1,2,4,8")
    ap.add_argument("--queries", type=int, default=60)
    ap.add_argument("--out", default="artifacts/adversarial.json")
    a = ap.parse_args(argv)
    files = sorted(sum((glob.glob(os.path.join(a.fake, "**", f"*.{e}"), recursive=True)
                        for e in ("png", "jpg", "jpeg")), []))[:a.n]
    imgs = [Image.open(f).convert("RGB") for f in files]
    pipe = get_pipeline()

    def score(im):
        r = pipe.analyze(im, media_type="image", explain=False, stability=False, force_full=True)
        p = min(max(r.p_fake, 1e-6), 1 - 1e-6)
        return float(_logit(p))

    out = robustness_curve(score, imgs, [float(x) for x in a.eps.split(",")], queries=a.queries)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
