# 🛡️ Deepfake Forensic Intelligence Suite

Multi-modal (image, video, audio) deepfake analysis built as a **calibrated
evidence cascade**: every verdict is *Real*, *Fake* or *Inconclusive*, comes
with a calibrated probability, an explanation and a cryptographically signed
report.

> **Honest status.** The architecture, training, calibration and evaluation
> tooling are implemented and unit-tested on synthetic data with stub / tiny
> local models (`pytest`, 86 tests). **No accuracy figure is claimed.** Real
> accuracy depends on the backbone you train and the datasets you calibrate
> on. Measure it with `train.evaluate` on data the model has never seen
> (ideally a different dataset / generator than training).
>
> **Available model stack:** the repository has configurable DINO-style
> embeddings/probes, optional SigLIP, spectral and temporal heads, audio
> backends, and heuristic cross-modal/physiological branches. Availability of
> a code path does not mean a trained checkpoint is installed or validated.
> Adaptive VOI and GLS fusion are **on by default**; until calibration
> artifacts exist, expert precision/correlation fallbacks are priors, not
> learned measurements.

## How it works (6-stage architecture)

```
Stage 1  Secure media intake
         Format validation · Hashing · Metadata · Decode limits · C2PA
Stage 2  Intelligent triage
         Media type · Resolution · Face/speech · Quality · Compute budget
Stage 3  Adaptive expert controller (VOI)
         Image: SigLIP + DINOv2/3 + frequency + localization
         Video: frame screen + temporal (FTCN/VideoMAE) + timestamps
         Audio: AASIST-style + speech embeddings + quality
         Cross-modal: lip-sync · identity · temporal alignment
         Physiological (P1): rPPG gated by signal quality
Stage 4  Dependence-aware evidence fusion
         Calibrated scores · Correlated experts (GLS) · Missing modalities
Stage 5  Decision and explanation
         Real / Fake / Inconclusive · Localized evidence · Confidence limits
Stage 6  Auditable forensic report
         Model versions · Evidence timestamps · Provenance · Signed output
```

* **Abstains instead of guessing.** Conformal prediction sets give a coverage
  target (`DEEPFAKE_ALPHA`, default 0.1). Out-of-domain inputs (Mahalanobis
  distance on embeddings) become *Inconclusive*.
* **Operating modes:** `strict` (~1% false alarms), `balanced` (~5%),
  `sensitive` (~10%). Without a fitted calibration the system says so and uses
  a conservative margin around 0.5.
* **Test-time-augmentation consistency.** The Tier-1 screen scores the
  original, a mirror and a tight crop (`DEEPFAKE_TTA_VIEWS`, 1 = off). The
  mean is the score; the spread is an epistemic-uncertainty signal that lowers
  the evidence reliability and blocks the early exit when the call flips.
* **Spectral-fingerprint expert** (`detector/experts/spectral.py`): azimuthal
  Fourier-power features (spectral slope, high-frequency ratio, periodic
  upsampling peaks) plus a logistic head trained on CPU with
  `python -m train.train_spectral`. Skipped, never guessed, until a head
  exists; a head with validation AUC ~0.5 gets zero vote.
* **Counterfactual explanations.** Multi-scale occlusion (3/6/9 grids fused),
  a deletion-vs-random faithfulness score, and a minimal-evidence analysis:
  the smallest region whose removal flips the call, and the smallest region
  that alone keeps it. Results are reported only from measured scores.
* **Tier-2 heuristics** (rPPG, sync, geometry, tamper) are low-weight,
  explicitly labelled as heuristics, and cannot override a strong neural score.
* **C2PA:** presence of a manifest is reported; its signature is *not* verified.
* **Signed reports:** Ed25519 key auto-created at
  `artifacts/report_signing_key.pem` (keep it private, it is git-ignored).
  Verify with `python -m scripts.verify_report report.json`.

## Novel features (implemented)

These match the recommended prioritisation and are wired into the cascade:

1. **Evidence-driven adaptive inference** — `InvestigationController` (VOI)
   selects the next expert by expected risk/entropy reduction per cost.
   Default: **on** (`DEEPFAKE_CONTROLLER=1`).
2. **Correlation-aware forensic evidence fusion** — GLS fusion over an
   evidence graph that can load learned expert error correlations and relative
   precision weights. Fit these on calibration data with
   `train.fit_fusion_correlations`; without those artifacts, correlations and
   weights fall back to stated priors. Default: **on**
   (`DEEPFAKE_DEP_FUSION=1`).
3. **Counterfactual robustness fingerprint** — mild JPEG/resize/blur/noise
   transformations; abstain if the verdict flips. Default: **on**
   (`DEEPFAKE_ROBUSTNESS_GATE=1`).
4. **Fine-grained evidence localization** — occlusion heatmaps, tamper maps,
   suspicious video timestamps / audio intervals, linked to originating
   detectors (`DEEPFAKE_LOCALIZATION=1`).
5. **Generator-shift & forensic provenance** — DINOv2 embedding Mahalanobis
   OOD, metadata/C2PA reporting, model-version tracking in the signed report
   (`DEEPFAKE_PROVENANCE=1`).

| Variable | Effect |
|---|---|
| `DEEPFAKE_CONTROLLER=1` | VOI adaptive expert selection (default on) |
| `DEEPFAKE_DEP_FUSION=1` | correlation-aware GLS fusion (default on) |
| `DEEPFAKE_CONDITIONAL=1` | context-conditional conformal + learned reliabilities |
| `DEEPFAKE_ROBUSTNESS_GATE=1` | abstain on unstable re-encodings (default on) |
| `DEEPFAKE_LOCALIZATION=1` | heatmaps / timestamps / region linkage (default on) |
| `DEEPFAKE_PROVENANCE=1` | OOD + provenance reporting (default on) |
| `DEEPFAKE_MIN_SUFFICIENCY=0.5` | abstain when evidence is insufficient |
| `EMBED_MODEL_ID` | DINOv2/DINOv3 encoder (default `facebook/dinov2-base`) |
| `SIGLIP_MODEL_ID` | complementary semantic branch |
| `AUDIO_MODEL_ID` | AASIST-style / audio classifier |
| `WAVLM_MODEL_ID` | optional speech foundation features |
| `VIDEO_TEMPORAL_MODEL_ID` | FTCN / VideoMAE / TimeSformer preference |

Evaluate with `train.evaluate_ablation`, `train.evaluate_adaptive` and
`train.evaluate_robustness`. Measure real accuracy only after training and
calibration on held-out generators.

### P0 readiness (do these before any accuracy claim)

```bash
# 1. What is actually trained / loaded?
python -m scripts.expert_audit --out artifacts/expert_audit.json

# 2. Train heads on real data (image + video + audio)
python -m train.finetune_all --train data/ffpp data/asvspoof \
    --encoder facebook/dinov2-base --out artifacts --modalities image video audio

# 3. Calibrate on a DISJOINT split
python -m train.fit_calibration --data data/calib --media image

# 4. Evaluate on UNSEEN generators + fusion ablations
python -m train.evaluate --data data/celebdf --media image --out artifacts/eval.json
python -m train.evaluate_ablation --calib data/calib --test data/celebdf --media image --out artifacts/ablation.json

# 5. Deployment latency / memory
python -m scripts.benchmark_inference --image sample.jpg --runs 5 --out artifacts/bench.json
```

Research framing: [docs/RAAFC.md](docs/RAAFC.md) (Reliability-Aware Adaptive Forensic Cascade).
Unit tests (synthetic data) validate software behaviour only — **not** detection performance.


## Install

```bash
git clone https://github.com/mayurOG/DEEPFAKE-ANALYSYS-SYSTEM.git
cd DEEPFAKE-ANALYSYS-SYSTEM
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
# optional, better face detection (pins an old torch, so install without deps):
pip install --no-deps facenet-pytorch
# development / tests
pip install -r requirements-dev.txt && pytest
```

`ffmpeg` on PATH is recommended for audio-in-video. Without MTCNN the OpenCV
Haar detector is used; without either, the whole frame is analysed.

## Run

```bash
cp .env.example .env            # optional settings
streamlit run app.py            # UI: single, webcam, batch/CSV, compare, lab, signed report
python cli_scan.py -i suspect.mp4 -o reports --mode balanced
```

On Windows systems where the app dependencies are installed in Python 3.9, run
`run_app.bat` from the project folder. It checks the UI and inference imports
before launching Streamlit with that interpreter; optional Streamlit arguments
can be passed to the batch file.

## Training and fine-tuning (image · video · audio)

Do this on a GPU box (Colab / Kaggle recommended for vision). Folder layout for
every dataset:

```
data/<name>/real/**.{jpg,png,mp4,wav,mp3,...}
data/<name>/fake/**.{jpg,png,mp4,wav,mp3,...}
```

Splits are group-aware (frames / utterances of one source never land in both
train and validation). Keep **train / calib / test disjoint** — ideally different
generators for test.

### One-command multi-modal fine-tune

```bash
# Image (DINOv2 probe + spectral) + video temporal head + audio probe
python -m train.finetune_all \
    --train data/ffpp data/asvspoof \
    --encoder facebook/dinov2-base \
    --out artifacts \
    --modalities image video audio

# Also light-fine-tune vision LayerNorms and the HF audio classifier
python -m train.finetune_all \
    --train data/ffpp data/asvspoof \
    --encoder facebook/dinov2-base \
    --out artifacts \
    --modalities image video audio \
    --finetune-image --finetune-audio --epochs-audio 5
```

### Per-modality commands

**Image** (frozen DINOv2/DINOv3 + linear probe ± LayerNorm tuning + spectral):

```bash
python -m train.features --data data/ffpp --encoder facebook/dinov2-base --out feats.npz
python -m train.train_probe --train feats.npz --out artifacts/probe.npz
python -m train.finetune_ln --data data/ffpp --encoder facebook/dinov2-base --out artifacts   # optional
python -m train.train_spectral --data data/ffpp --out artifacts
```

**Video** (temporal head on frame embeddings):

```bash
python -m train.train_temporal --data data/ffpp_video --encoder facebook/dinov2-base --out artifacts
```

**Audio** (WavLM probe ± light HF classifier fine-tune):

```bash
# Frozen WavLM / wav2vec2 features → logistic probe → artifacts/audio_probe.npz
python -m train.train_audio --train data/asvspoof --out artifacts \
    --encoder microsoft/wavlm-large

# Also fine-tune the last layers of the HF audio classifier
python -m train.train_audio --train data/asvspoof --out artifacts \
    --finetune --epochs 5 --batch-size 8
# then: export AUDIO_MODEL_ID=artifacts/audio_finetuned
```

### Full image/video pipeline (features → probe → spectral → calibration → eval)

```bash
python -m train.run_all --train data/ffpp --calib data/calib --test data/celebdf \
    --encoder facebook/dinov2-base --out artifacts --finetune   # --media video for video
```

### Calibration and evaluation (after any fine-tune)

```bash
# Set AUDIO_MODEL_ID in .env to the trained audio artifact. Probe artifacts
# record their matching encoder and the detector loads that encoder automatically.
python -m train.fit_calibration --data data/calib --media image
python -m train.fit_calibration --data data/calib_video --media video
python -m train.fit_calibration --data data/calib_audio --media audio
python -m train.evaluate --data data/celebdf --media video --mode balanced --out eval.json
python -m scripts.run_canary --set-baseline && python -m scripts.run_canary
```

`notebooks/train_on_colab.ipynb` runs the image path on a Colab GPU.

One command does steps 1-5 (features, probe, spectral head, calibration,
evaluation) and enforces disjoint train / calib / test folders:

```bash
python -m train.run_all --train data/ffpp --calib data/calib --test data/celebdf \
    --encoder facebook/dinov2-base --out artifacts          # add --finetune for LayerNorm tuning
```
`notebooks/train_on_colab.ipynb` runs it on a Colab GPU. Once a probe exists,
the encoder saved in the probe metadata is used automatically (legacy probes
without encoder metadata fall back to `EMBED_MODEL_ID`). The probe becomes the
primary Tier-1 screen and the pretrained classifier becomes a second independent vote.

Outputs land in `artifacts/` (`probe.npz`, `spectral_head.npz`, `calibration.json`, `ood_*.npz`,
`fusion_weights.json`, temporal head). `fit_fusion_weights` in
`detector/fusion.py` can learn expert weights by logistic stacking.

For audio, fine-tune an XLS-R + AASIST style model externally and point
`AUDIO_MODEL_ID` at it.

## Configuration (env vars)

`IMAGE_MODEL_ID`, `EMBED_MODEL_ID`, `AUDIO_MODEL_ID`, `HF_TOKEN`,
`DEEPFAKE_ARTIFACT_DIR`, `DEEPFAKE_DEVICE`, `DEEPFAKE_ALPHA`,
`DEEPFAKE_TIER1_FRAMES`, `DEEPFAKE_FACE_SECONDS`, `DEEPFAKE_TTA_VIEWS`. See `.env.example`.

## Layout

```
detector/   pipeline, models, calibration, fusion, triage, explain, stability,
            canary, report_generator, experts/{temporal,tamper,spectral,facesignals},
            evidence_graph, investigation_controller, conditional_reliability,
            contradiction_map, transformation_lab, audit_chain
train/      data, features, train_probe, train_spectral, train_temporal,
            train_audio, finetune_ln, finetune_all, fit_calibration,
            train_temporal, fit_calibration, evaluate, run_all,
            evaluate_ablation, evaluate_adaptive, evaluate_robustness
notebooks/  train_on_colab.ipynb
scripts/    run_canary, verify_report
tests/      pytest suite (synthetic data, offline)
app.py  cli_scan.py
```


## Real-data evaluation protocol (required for any accuracy claim)

```bash
# 0. Organise corpora → data/<set>/real|fake/**  then build leakage-safe splits
python -m scripts.prepare_datasets --catalog
python -m scripts.prepare_datasets --validate data/ffpp
python -m scripts.prepare_datasets --split data/ffpp --out data/splits/ffpp \
    --train-frac 0.6 --calib-frac 0.2 --test-frac 0.2

# 1. Full protocol: train on train, calib+correlations on calib, metrics on test
python -m train.run_real_evaluation \
    --train data/splits/ffpp/train --calib data/splits/ffpp/calib \
    --test data/celebdf --encoder facebook/dinov2-base --media image \
    --out artifacts/real_eval

# 2. Learn calibration-only error correlations and precision weights
#    (writes expert_correlation.json and fusion_weights.json)
python -m train.fit_fusion_correlations --data data/splits/ffpp/calib --media image

# 3. Per-domain leave-one-out calibration + selective-risk / conformal report.
#    Include generator and laundering domains as separate --domain arguments.
python -m train.evaluate_generator_shift \
    --domain ffpp=data/splits/ffpp/test --domain flux=data/flux \
    --domain sd3=data/sd3 --domain social_reupload=data/social_reupload \
    --media image --out artifacts/generator_shift.json

# 4. C2PA manifest presence is detected, but this implementation does not
#    validate C2PA signatures (SDK installation alone does not enable trust).
```

```bash
# 5. Concurrent load test before public demo
python -m scripts.load_test --image sample.jpg --workers 8 --requests 40
python -m scripts.benchmark_inference --image sample.jpg --runs 5
```

`evaluate_generator_shift` fits calibration separately for each held-out
domain using scores from the other domains, then reports risk-coverage curves
and empirical conformal coverage. It holds the currently loaded detector
weights fixed; train those weights without the evaluation generators before
describing the result as leave-one-generator-out training evaluation.
Recompression and social-media re-upload are only measured when supplied as
their own labelled domains. Neither command creates benchmark data.



## CALM-FORGE (cross-asset lineage mode)

Research extension: investigate **sets of related files**, not only isolated uploads.
See [docs/CALM_FORGE.md](docs/CALM_FORGE.md).

```bash
python -m scripts.investigate_lineage photo.jpg photo_resized.jpg clip.mp4 \
    --out artifacts/lineage_report.json
```

Builds fingerprints, a lineage graph with uncertainty, optional shortcut audits
(JPEG/resize/blur stability), and a narrative report. Single-file `Pipeline`
behaviour is unchanged.

## Limitations (read before claiming results)

* **Accuracy is unmeasured** until you train, calibrate and evaluate on real
  media and *unseen* manipulation methods. Passing unit tests on stubs is not
  detection performance.
* Several Tier-2 branches are **heuristics** or require a trained head that is
  skipped until `artifacts/*.npz` exists — run `scripts.expert_audit`.
* Conformal guarantees assume exchangeable calibration data; new generators
  break that. The generator-shift harness reports empirical results but does
  not restore a distribution-free guarantee under arbitrary shift.
* GLS fusion and the VOI controller need learned correlations, precision
  weights and costs. The correlation fitter estimates the first two from
  calibration scores; evaluate learned GLS against averaging and fixed
  weights with `train.evaluate_ablation` before claiming an improvement.
* C2PA: presence is reported; **signatures are not verified** by the current
  implementation. Report Ed25519 signatures cover the JSON report body, not the
  upstream media provenance chain.
* Heuristic experts (rPPG, sync, geometry) are weak on compressed video;
  missing pulse is never treated as proof of a deepfake.
* Occlusion heatmaps explain model behaviour, not ground-truth manipulation
  masks. Detection results are probabilistic evidence, not legal proof.
* Public demos must measure cold-start, p95 latency and peak RAM
  (`scripts.benchmark_inference`) before promising capacity.

Built with ❤️ OGmayur


## v6 additions
See [docs/V6_ADDITIONS.md](docs/V6_ADDITIONS.md): shift-aware conformal prediction, C2PA provenance lane,
open-set generator attribution, black-box adversarial stress test, evidence-grounded explanations and audio
laundering transforms. Accuracy is still **unmeasured** until you run the Colab notebook on real data.
