# v6 additions (what changed after the v5 review)

> **Status.** Everything below is implemented and unit-tested on synthetic data.
> **No real-data accuracy has been measured** and **no model was fine-tuned in
> the build environment** (no GPU, no datasets, no model downloads). Treat each
> item as a method to validate with `notebooks/train_on_colab.ipynb`.

| Review weakness / idea | Change | Where |
|---|---|---|
| Failing test (stale after dependency-aware fusion became default) | Test now asserts the real contract: flags off = legacy weighted mean; GLS on keeps verdict and sign | `tests/test_egafr_pipeline.py` |
| Coverage breaks under new generators | Weighted conformal (density-ratio, covariate shift) + adaptive conformal (online alpha) | `detector/shift_conformal.py` |
| No provenance lane | C2PA/JUMBF presence + IPTC AI-source markers; never claims signature validation; missing manifest is neutral | `detector/provenance.py` |
| Which generator made it? | Open-set generator-family attribution on spectral fingerprints (unknown when far/low-confidence); context only | `detector/attribution.py`, `train/train_attribution.py` |
| Adversarial fragility unknown | Black-box random-search evasion, flip rate vs L_inf budget (a lower bound) | `detector/adversarial.py`, `train/evaluate_adversarial.py` |
| Explanations can say unsupported things | Claims bound to recorded evidence; free text filtered to supported sentences | `detector/grounded_explain.py` |
| Audio lane thin | Default SSL encoder now WavLM Large; added mu-law codec and reverb laundering transforms | `detector/config.py`, `detector/transformation_lab.py` |
| No numbers to show | Colab cells to produce ablation, adaptive, robustness and adversarial tables | `notebooks/train_on_colab.ipynb` |

## Not done (needs your compute/data)
* Fine-tuning DINOv3 / SigLIP2 / XLS-R heads and fitting calibration on real splits.
* Cross-dataset benchmark table (train FF++, test Celeb-DF / DFDC) with bootstrap CIs.
* Distillation of a small screen model, SAM-based localisation scoring.
