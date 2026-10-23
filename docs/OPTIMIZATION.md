# Practical deployment optimisations

Aligned with the recommended model stack. Measure on your hardware after
enabling the corresponding backends / heads.

| Detector            | Optimisation                                                                  | What to measure                                   |
|---------------------|-------------------------------------------------------------------------------|---------------------------------------------------|
| SigLIP + DINOv2/3   | Cache embeddings; smaller backbone first; mixed precision where supported     | AUC, latency, memory                              |
| Frequency branch    | Reuse decoded image tensors; avoid redundant FFT calculations                 | Robustness under compression, incremental benefit |
| FTCN / VideoMAE     | Scene-aware frame sampling; bounded clip length; batch inference              | Video AUC, latency per minute                     |
| AASIST / WavLM      | Resample audio once; cache features; chunk long recordings                    | EER, AUC, CPU/RAM                                 |
| Audio-video fusion  | Align timestamps and handle missing streams explicitly                        | Sync error detection, false alarms                |
| rPPG / face signals | Run only on sufficiently long, visible, stable face tracks                    | Signal quality, incremental discrimination        |
| OOD and calibration | Fit on a separate validation split; recalibrate after model changes           | AUROC for OOD, ECE, coverage-risk                 |
| Localization        | Reuse detector features; segment only candidate regions                       | Localization F1, IoU, extra latency               |

Notes:

- The adaptive controller (`DEEPFAKE_CONTROLLER=1`) already limits expensive
  experts when early evidence is sufficient.
- Spectral features are computed once per image and shared with the fusion
  graph.
- Face-signal experts are quality-gated; missing pulse is never treated as
  proof of a deepfake.
