# CALM-FORGE

**Cross-Asset Lineage and Manipulation Forensics with Reliability-Guided Evidence**

Provisional research name for the multi-asset extension of this suite.  
Not an established benchmark model. **No real-data accuracy is claimed.**

## Hypothesis (to be tested, not assumed)

Combining (1) cross-asset relationship estimation, (2) controlled tests of
evidence reliability under benign transforms, and (3) cost-aware forensic
investigation can improve reliability under distribution shift while reducing
unnecessary computation — relative to fixed single-file pipelines.

## Five questions the system targets

1. What evidence suggests manipulation?
2. Where / when does suspicious evidence occur?
3. Are multiple uploaded files related versions of the same content?
4. Is the system relying on meaningful evidence or shortcuts (compression, resolution)?
5. When is evidence too weak or contradictory to conclude?

## Module map

| Module | Role |
|--------|------|
| `detector/asset_registry.py` | Register files, hashes, metadata, analysis IDs |
| `detector/media_fingerprints.py` | Image/video/audio fingerprints |
| `detector/lineage_matcher.py` | Pairwise relationship scores + uncertainty |
| `detector/lineage_graph.py` | Graph of assets and probable links; role hypotheses |
| `detector/mutation_events.py` | Candidate edits / transforms between related assets |
| `detector/shortcut_audit.py` | Nuisance-factor interventions → fragility / shortcut flags |
| `detector/lineage_report.py` | Cross-asset narrative + JSON/Markdown report |
| `detector/cross_asset.py` | Session orchestrator |
| `scripts/investigate_lineage.py` | CLI |

Single-file analysis via `Pipeline` is unchanged. Cross-asset mode is additive.

## Contributions (research framing)

- **A — Cross-Asset Forensic Lineage:** graph of candidate relationships with confidence and uncertainty; unknown is allowed.
- **B — Causal Shortcut / Evidence-Reliability Audit:** measure prediction instability under JPEG, resize, blur, noise.
- **C — Evidence- and Compute-Aware Investigation:** existing VOI controller + reuse of fingerprints; evaluate at matched compute budgets.

## CLI

```bash
python -m scripts.investigate_lineage a.jpg b.jpg c.mp4 \
  --out artifacts/lineage_report.json

# Fingerprints + graph only (no model download)
python -m scripts.investigate_lineage a.jpg b.jpg --no-forensics --no-shortcut-audit
```

## Metrics you must report before claiming results

| Area | Metrics |
|------|---------|
| Classification | AUROC, balanced accuracy, F1 |
| Unseen generators | Cross-dataset drop |
| Calibration | Brier, ECE |
| Selective prediction | Coverage vs error |
| Localization | IoU / precision-recall when masks exist |
| Lineage | Pairwise precision/recall/F1, false-link rate |
| Shortcut audit | Nuisance flip rate, fragility vs errors |
| Efficiency | Median/p95 latency, expert invocations at matched budgets |

Use source-disjoint splits. Do not advertise accuracy from unit tests alone.

## Related work (context, not completeness)

Multimodal detection/localization/explanation and joint source-tracing appear
in recent venues (e.g. Omni-Fake and related CVPR 2026 lines). The defensible
claim is the **specific integration and measured gains**, not the mere use of
multiple models.
