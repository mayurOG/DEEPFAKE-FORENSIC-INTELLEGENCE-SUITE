# EGAFR: Evidence-Graph-Guided Active Forensic Reasoning

**Status: implemented research prototype, unvalidated hypothesis.** The code
runs and is unit-tested on synthetic data. Whether the combination beats the
baseline cascade on real deepfakes is an open empirical question that this
repository gives you the protocol to answer (`train/evaluate_*.py`). No
accuracy or improvement figure is claimed anywhere.

Everything is **optional and off by default**: with no flags set the verdicts
are those of the baseline cascade (a test asserts this). The evidence graph,
the audit chain and the robustness profile are always computed because they
only *describe* the decision.

## Map from idea to code

| Idea | Module | Switch |
|---|---|---|
| 1 Counterfactual evidence graph, dependency-aware fusion | `detector/evidence_graph.py` | graph always on; `DEEPFAKE_DEP_FUSION=1` fuses with it |
| 2 Value-of-information investigation controller | `detector/investigation_controller.py` | `DEEPFAKE_CONTROLLER=1`, `DEEPFAKE_CONTROLLER_POLICY`, `DEEPFAKE_RISK_TARGET`, `DEEPFAKE_BUDGET` |
| 3 Localized cross-modal contradiction map | `detector/contradiction_map.py` | video Tier-2 when face + audio exist |
| 4 Counterfactual transformation laboratory | `detector/transformation_lab.py` | profile on with stability; `DEEPFAKE_ROBUSTNESS_GATE=1` abstains |
| 5 Provenance-aware chain of custody | `detector/audit_chain.py` | always on; head hash is inside the signed report |
| 6 Evidence-conditional reliability calibration | `detector/conditional_reliability.py` | `DEEPFAKE_CONDITIONAL=1` after `fit_calibration --conditional` |
| Evidence sufficiency abstention | `build_graph` + pipeline | `DEEPFAKE_MIN_SUFFICIENCY` (0 = off) |

## The mathematics as implemented

**Dependency-aware fusion.** Expert `i` reports log-odds `z_i` with precision
`p_i = w_i * r_i` (prior weight x context reliability), i.e. variance
`1/p_i`. With error correlation `rho_ij` (prior by expert family, or
estimated from held-out data with `estimate_correlation`), the generalised
least-squares estimate of the common log-odds is

    z_hat = (1' S^-1 z) / (1' S^-1 1),   Var = 1 / (1' S^-1 1),
    S_ij = rho_ij * sigma_i * sigma_j

With `rho = 0` this is exactly the existing precision-weighted mean (tested),
so it is a strict generalisation; duplicated evidence is not double counted.
Experts that would get negative GLS weight are dropped and the fit repeated.
Disagreement beyond stated precision inflates the uncertainty by the Birge
ratio `max(1, Q/(k-1))`. **Evidence sufficiency** is
`min(1, |z_hat| / (1.645 * sd))`: a one-sided z-test that the fused log-odds
differs from zero. It is deliberately separate from classifier probability.

**Controller.**

    U(a | G) = ( E[dR] + lambda * E[dH] ) / ( C(a) + eps ),   a* = argmax U, C(a) <= B

`G` gives a Gaussian belief `N(mu, v)` over the common log-odds. Adding test
`a` (precision `p_a`, correlation with the evidence already held) changes the
variance to `v'`. Decision risk is `R = Phi(-|mu| / sd)`; the preposterior of
the updated mean is `N(mu, v - v')`, so `E[R']` is computed by Gauss-Hermite
quadrature, `dR = R - E[R']`, `dH = 0.5 ln(v / v')`. A test highly correlated
with existing evidence is worth little, which is the "two visual models share
a failure mode" case. Stop reasons are explicit: risk target met, budget
exhausted, no test worth its cost, no eligible test. Baselines (`fixed`,
`random`, `quality`) share the interface. **The precisions and correlations
are stated priors, not learned values**; that is the main thing to validate.

**Conditional calibration.** Platt scaling is shared; class-conditional
conformal thresholds are fitted per context group (compression x coverage x
resolution), falling back to global thresholds when a group has too few
samples. Per-expert reliability = `clip(2 P(correct | context) - 1, 0, 1)`.
Fitted only on held-out calibration data, in two passes so the calibrator
sees reliability-adjusted logits.

**Contradiction map.** Windowed mouth-motion vs audio-envelope correlation,
lag search centred on the clip's own global lag (a constant dubbing offset is
benign), tested against an empirical null (the same mouth window against
audio windows from elsewhere in the clip). `contradiction` = indistinguishable
from the null while the clip's intact windows are clearly coupled;
non-informative windows (no speech, no mouth motion, face lost) are reported,
not scored.

**Chain of custody.** Hash-linked entries (`verified` / `declared` /
`inferred` / `unknown`); editing, inserting, deleting, reordering or
truncating any entry is detected; the head hash is covered by the Ed25519
report signature. A C2PA manifest is recorded as *declared and not validated*.
It complements standards-based C2PA validation and does not replace it.

## What the tests establish (synthetic data only)

* GLS reduces to the weighted mean at rho = 0; correlated duplicates are not
  double counted; the controller prefers a cheap independent test over a
  correlated one and stops for stated reasons.
* On a synthetic mixture with an easy group and a hard group, global conformal
  prediction under-covers the hard group and conditional conformal restores
  it by abstaining more there.
* The contradiction map, on 30 synthetic clips (speech-like envelopes): 0/30
  false alarms on genuine clips, 26/30 swapped 5 s segments detected, mean
  interval IoU about 0.69, and a constant 240 ms offset is not flagged.
  Synthetic envelopes are far easier than real speech and faces.
* Every kind of audit-chain tampering is detected.

## What they do not establish

* Any benefit on real deepfakes, real dubbing, or real compression.
* That the controller's stated precisions match reality.
* That conditional calibration helps under *your* real distribution shifts.
* Anything about unseen generators.

## Known limitations found while building it

* The TTA "centre crop" view resamples the image, which removes detail that
  some detectors use; on noise-like content it can itself flip a call (this
  was observed and is why such images are sent to the full cascade).
* Calibrators fitted on a whole population and applied to the Tier-2 subset
  (which was selected for ambiguity) carry a selection effect. The ablation
  fits the final-stage calibrator on that subset to avoid it; the production
  `fit_calibration` fits it on everything.
* Windows in the contradiction map overlap, so many windows mean many chances
  of a false alarm; intervals therefore need at least two windows.
* A clip whose audio and video are mismatched *throughout* has no intact
  reference and is reported as `no_reference`; that case belongs to the
  whole-clip sync expert.

## How to run the evaluation (needs real data and a GPU)

```bash
# 1. train + calibrate as usual (see README), with the EGAFR extras
python -m train.run_all --train data/ffpp --calib data/calib --test data/celebdf \
    --encoder <encoder> --out artifacts --conditional
# 2. component contribution (A-G), paired bootstrap CIs
python -m train.evaluate_ablation  --calib data/calib --test data/celebdf --out ablation.json
# 3. controller vs fixed cascade vs fixed/random/quality policies, per budget
python -m train.evaluate_adaptive  --calib data/calib --test data/celebdf --budgets 0.2 0.5 1.0 --out adaptive.json
# 4. degradation matrix + does fragility predict errors?
python -m train.evaluate_robustness --calib data/calib --test data/celebdf --out robustness.json
```

Keep splits disjoint and the test set from a different dataset or generator
family. The scripts refuse overlapping calibration/test items. Report CIs, and
report the configurations where nothing improved.

## Suggested paper framing

One primary contribution plus two supporting ones: for example the evidence
graph with dependency-aware fusion (primary), the controller (supporting:
equal risk at lower cost, or lower risk at equal cost), and conditional
calibration (supporting: better risk-coverage under shift). Novelty claims
must be argued against active learning, selective classification and
evidence-fusion literature; the novelty matrix in your design notes lists what
each claim has to demonstrate.
