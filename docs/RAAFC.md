# Reliability-Aware Adaptive Forensic Cascade (RAAFC)

Working name for the research framing of this repository’s adaptive cascade.

> **Status.** The software implements a value-of-information controller,
> correlation-aware (GLS) fusion, conformal abstention and quality-gated
> experts. Calibration-only Ledoit-Wolf residual covariance and relative
> precision fitting, plus leave-one-domain-out calibration/risk evaluation,
> are available. **Real-world detection accuracy has not been measured in this
> repository.** Unit tests on synthetic data validate software behaviour only.
> Treat RAAFC as a **method to validate**, not as a proven state-of-the-art
> detector. The domain harness holds detector weights fixed and cannot certify
> that a checkpoint was trained without a held-out generator.

## Core loop

1. **Media & quality assessment** — compression, resolution, face visibility,
   audio quality, missing modalities.
2. **Select the next evidence source** — choose the detector with the best
   expected reduction in decision risk per unit cost (VOI controller).
3. **Update reliability and dependencies** — correlated errors, unreliable
   modalities, distribution shift (OOD).
4. **Decide whether more evidence is needed** — stop when risk target is met,
   or **abstain** when uncertainty remains too high.
5. **Auditable forensic result** — verdict, calibrated risk, localized
   evidence, cost and reproducibility record (signed report + audit chain).

## What is *not* the novelty claim

Combining DINO, SigLIP, audio embeddings and video temporal models is not by
itself a novel contribution; multimodal ensembles already exist. The
defensible contribution is a **reproducible adaptive, reliability-aware
forensic evidence acquisition protocol** with explicit abstention and audit.

## Required experiments (before any accuracy claim)

| Experiment | What to measure | Command / tool |
|------------|-----------------|----------------|
| Single expert vs fusion | ROC-AUC, PR-AUC, balanced accuracy | `train.evaluate`, per-expert logs |
| Adaptive vs fixed inference | Accuracy / FPR at matched compute budgets | `train.evaluate_adaptive` |
| Learned correlation/precision vs averaging and fixed weights | ECE, Brier, risk-coverage and paired differences | `train.fit_fusion_correlations`, `train.evaluate_ablation` |
| In-domain vs unseen generators / laundering domains | Per-domain AUC, risk-coverage, empirical conformal coverage | `train.evaluate_generator_shift` |
| JPEG / blur / resize / re-encode | Degradation + verdict stability | `train.evaluate_robustness` |
| Missing audio or video | Reliability under incomplete modalities | ablate streams in evaluate |
| Localization | Region metrics vs masks (when available) | tamper / occlusion maps |
| Deployment efficiency | median & p95 latency, peak RAM, throughput | `scripts.benchmark_inference` |

## Honest reporting checklist

- [ ] Train heads on real data (`train.finetune_all` / modality scripts)
- [ ] Calibrate on a **disjoint** split (`train.fit_calibration`)
- [ ] Evaluate on **unseen** generators (`train.evaluate`)
- [ ] Run domain folds (`train.evaluate_generator_shift`) with checkpoints trained
      without the held-out generators; include recompressed/re-uploaded media
- [ ] Publish expert readiness (`scripts.expert_audit`)
- [ ] Publish ablation table (`train.evaluate_ablation`)
- [ ] Publish latency / memory (`scripts.benchmark_inference`)
- [ ] Never cite unit-test counts as detection accuracy

## Related code

| Component | Path |
|-----------|------|
| VOI controller | `detector/investigation_controller.py` |
| GLS / evidence graph | `detector/evidence_graph.py` |
| Fusion | `detector/fusion.py` |
| Conditional reliability | `detector/conditional_reliability.py` |
| Robustness gate | `detector/transformation_lab.py` |
| Expert audit | `scripts/expert_audit.py` |
| Ablation study | `train/evaluate_ablation.py` |
| Adaptive eval | `train/evaluate_adaptive.py` |
| Learned fusion fitting | `train/fit_fusion_correlations.py` |
| Domain-shift selective benchmark | `train/evaluate_generator_shift.py` |
