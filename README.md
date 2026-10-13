# VisionGuard

**Person detection, multi-object tracking and camera-specific event analytics with an
evaluated, frozen baseline.** VisionGuard combines YOLO26n and ByteTrack with line crossing,
restricted-zone intrusion, loitering, annotated video and persisted run evidence.

The public repository contains the Python pipeline, configurations, tests and benchmark
scripts. The selected detector is a five-epoch CrowdHuman fine-tune at confidence **0.55**.

## What I Built

- A reusable OpenCV → YOLO → ByteTrack pipeline with trajectories and tracking diagnostics.
- Independent state machines for IN/OUT crossing, intrusion and dwell-time loitering.
- Validated YAML configuration, SQLite run/event persistence and optional JSONL exports.
- A trusted-local FastAPI/Redis/RQ service that runs the same pipeline asynchronously.
- Bounded asynchronous video encoding, FFmpeg H.264 NVENC and per-stage timing.
- Dataset preparation, detector fine-tuning, camera calibration/holdout evaluation,
  MOT17 benchmarking and ignore-aware CrowdHuman evaluation.
- A MOT evaluator checked against a pinned TrackEval reference on **24 prediction files**.

## Main Results

Confidence was selected on **camera 1**, then locked for the **camera-3 holdout** and public
MOT17 comparison. Higher HOTA/MOTA/IDF1 is better; lower IDSW and mMR is better.

| Tracking evaluation | Detector | Confidence | HOTA | MOTA | IDF1 | IDSW |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Camera 1 calibration | Pretrained YOLO26n | 0.30 | 40.905 | 38.789 | 49.548 | 78 |
| Camera 1 calibration | CrowdHuman fine-tuned | 0.55 | **43.946** | 37.502 | **54.621** | 81 |
| Camera 3 locked holdout | Pretrained YOLO26n | 0.30 | 57.349 | 63.790 | 77.504 | 15 |
| Camera 3 locked holdout | CrowdHuman fine-tuned | 0.55 | **65.696** | **77.681** | **86.777** | 16 |
| MOT17 train/FRCNN, 5,316 frames | Pretrained YOLO26n | 0.30 | 33.38 | 28.03 | 37.40 | 467 |
| MOT17 train/FRCNN, 5,316 frames | CrowdHuman fine-tuned | 0.55 | **39.62** | **39.46** | **48.79** | **351** |

| CrowdHuman full-body evaluation | AP | mMR |
| --- | ---: | ---: |
| Pretrained YOLO26n | 0.440238 | 0.906192 |
| CrowdHuman fine-tuned | **0.810561** | **0.597033** |

The candidate improves holdout tracking substantially, with a precision/false-positive
trade-off. Camera 1 MOTA remains 1.287 points below the pretrained reference. Camera 2 has
no ground truth: its manual shadow review accepted the candidate with one residual ID
transfer around **00:06–00:07**. These limitations are part of the result.

See [benchmark, methodology and evidence](docs/benchmark.md) for complete metrics,
selection decisions, reproduction commands and artifact hashes.

## Run The Baseline

Run commands from the repository root. The tested CUDA environment is defined in
`environment.yml`; inference dependencies are pinned in `requirements-lock.txt`.

```powershell
conda env create -f environment.yml
conda activate visionguard
```

Place the selected checkpoint at:

```text
runs/person_detection/crowdhuman_a/weights/best.pt
```

Expected SHA-256:

```text
c6a6fa5082445a7b2e4a87bf37f57cd2523652bdceadff8b8d09d72b7bcb9837
```

Weights, source media and datasets are excluded from Git. A fresh clone needs the checkpoint
or a new training run; the repository does not provide a download link for the private
checkpoint. The [benchmark guide](docs/benchmark.md#reproduce) includes the training recipe.

```powershell
python -m scripts.run_video `
  --config configs/person_tracking_final.yaml `
  --source path/to/input.mp4 `
  --output data/outputs/annotated.mp4 `
  --no-display
```

`person_tracking_final.yaml` enables tracking-only inference with confidence 0.55,
640-pixel detector input, evaluated ByteTrack parameters, and asynchronous NVENC output.
FFmpeg must be on `PATH`, with a working NVIDIA `h264_nvenc` encoder. For a host without
NVENC, use a local copy with `output.encoder: opencv`; retain the detector/tracker settings.

For event analytics, use a camera-specific configuration such as
`configs/week4_video_a.yaml` and replace its line/ROI coordinates for the input scene.
The historical event templates use the pretrained detector at confidence 0.40; they are
separate from the selected tracking baseline. Event coordinates use original video pixels
and the bottom-center of each person's box.

## Pipeline And Repository

```text
CLI / queued API job + YAML
  → AnalysisService + RunRecorder
  → VideoPipeline
  → YOLO26n + ByteTrack
  → crossing / intrusion / loitering
  → annotated video + SQLite events + optional JSONL
```

| Location | Purpose |
| --- | --- |
| `configs/` | Frozen tracking baseline and historical camera-specific event templates. |
| `src/` | Detection, tracking, event engines, pipeline, persistence and local API adapters. |
| `scripts/` | Run videos, prepare datasets, evaluate models and validate the MOT evaluator. |
| `tests/` | Regression tests and a tracked MOT assignment-tie fixture. |
| `docs/architecture.md` | Runtime/configuration design, event semantics and optional service operation. |
| `docs/benchmark.md` | Training, experiment decisions, metrics, reproducibility and evidence. |

## Verification And Performance

```powershell
python -m compileall -q src scripts
python -m pytest tests/ -q
```

The evaluated baseline has **206 passing tests**. CI installs CPU PyTorch, compiles the
Python sources and runs the baseline suite independently of local application files.

Selected controlled encoder runs on an NVIDIA RTX 3050 6 GB Laptop GPU measured:

| Input | Synchronous OpenCV | Async OpenCV | Async NVENC |
| --- | ---: | ---: | ---: |
| 1080p, 25 FPS | 9.423 FPS | 14.435 FPS | **27.752 FPS** |
| 4K, 30 FPS | 5.042 FPS | 10.097 FPS | **16.853 FPS** |

These isolated measurements are described in the benchmark guide. Throughput varies with
scene density, encoding, temperature and concurrent load; they do not establish universal
real-time performance for the promoted detector.

## Scope And Limitations

- This is an evaluated portfolio/research prototype with a trusted-local service.
- Camera 2 is visually reviewed but has no annotated tracking-accuracy measurement.
- Counting and intrusion depend on stable IDs; small people, merged boxes and occlusion
  can cause missed or duplicate events.
- Geometry is camera-specific and does not compensate for camera movement or perspective.
- The local API has no authentication and accepts paths within configured server roots.
- Full camera experiments require private source videos/annotations. Public MOT17 results
  use train ground truth and are not a MOTChallenge test-server submission.
- Dataset files, checkpoints and generated outputs are intentionally outside version control.
