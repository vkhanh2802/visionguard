# MOT17 Tracking Baseline

## Setup

- Dataset: MOT17 train split
- Variant: FRCNN image folders only; DPM and SDP contain the same source frames
- Sequences: 02, 04, 05, 09, 10, 11, and 13
- Total frames: 5,316
- Detector: `yolo26n.pt`, person confidence 0.40
- Tracker: ByteTrack with `track_buffer: 75`
- Matching threshold: IoU 0.50 for CLEAR and identity metrics
- Config: `configs/week6.yaml`

The in-repository evaluator scores valid MOT17 pedestrian annotations, removes unmatched
predictions that overlap distractor regions, and reports HOTA, DetA, AssA, LocA, MOTA,
MOTP, IDF1, identity switches, fragmentations, precision, and recall.

The camera-1 and camera-3 results below were regenerated after the evaluator was corrected.
Local validation against TrackEval commit `12c8791b303e0a0b50f753af204249e622d0281a` matched every reported
metric and count. The earlier MOT17 tables are retained as historical experiment notes, but
were produced before that correction. The MOT17 ground truth is not currently present in the
workspace, so those tables must not be used as final benchmark evidence until they are rerun.

## Aggregate Results

| Output | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw ByteTrack | 32.018 | 23.346 | 44.209 | 25.644 | 34.637 | 332 | 92.441 | 28.249 |
| Canonical continuity | 29.525 | 23.331 | 37.616 | 25.482 | 30.436 | 491 | 92.449 | 28.225 |

End-to-end throughput was 16.325 FPS across all sequences.

## Interpretation

The detector is precise but misses many MOT17 pedestrians, especially small and crowded
targets. Detector recall is the primary bottleneck before tracker tuning.

The Week 6 continuity rules are camera-specific and do not generalize to MOT17. They
increase identity switches and reduce HOTA, AssA, and IDF1 on this benchmark. Continuity
therefore remains disabled by default and should only be enabled in a validated camera
configuration.

The next controlled experiment should improve detection recall first, either by lowering
the confidence threshold on MOT17 or fine-tuning the detector with pedestrian data such as
CrowdHuman. Tracker or ReID changes should be considered only after detection recall is no
longer the dominant error source.

## Confidence Sweep

Continuity was disabled so the sweep measured raw YOLO and ByteTrack behavior. All runs
used the runtime Week 6 ByteTrack configuration and the same 5,316 MOT17 frames.

| Confidence | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | Precision | Recall | FPS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | **34.312** | **28.334** | 41.983 | **30.678** | **38.376** | 460 | 89.939 | **35.004** | 16.524 |
| 0.15 | 34.255 | 27.851 | 42.532 | 30.226 | 38.165 | 473 | 90.387 | 34.295 | 16.675 |
| 0.20 | 33.834 | 27.329 | 42.292 | 29.705 | 37.564 | 478 | 90.657 | 33.593 | 18.280 |
| 0.25 | 33.662 | 26.807 | 42.672 | 29.148 | 37.293 | 487 | 90.915 | 32.866 | **18.383** |
| 0.30 | 33.334 | 25.903 | 43.271 | 28.239 | 36.781 | 451 | 91.314 | 31.651 | 17.711 |
| 0.40 | 32.147 | 23.376 | **44.504** | 25.652 | 35.039 | **329** | **92.393** | 28.272 | 17.002 |

Confidence 0.10 is the strongest MOT17 result by HOTA, DetA, MOTA, IDF1, and recall.
However, it is not safe for the Week 6 camera. Camera regression produced the following:

| Confidence | IN | OUT | Intrusion | Loitering | Raw tracks | Duplicate pairs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 6 | 4 | 1 | 4 | 239 | 67 |
| 0.30 | 6 | 3 | 1 | 3 | 236 | 26 |
| 0.40 | 4 | 3 | 1 | 1 | 132 | 9 |

The production camera must retain confidence 0.40. Confidence 0.10 is useful only as a
MOT17 benchmark result and evidence that detector recall is limiting general tracking.
Improving both domains requires detector fine-tuning or camera-specific calibration, not a
global confidence reduction.

## Experiment A: MOT17 Fine-Tuning

YOLO26n was fine-tuned on the MOT17 sequence split with sequences 02, 04, 05, 10, and 11
for training and sequences 09 and 13 for validation. Training used 640-pixel images, batch
size 16, AdamW with an initial learning rate of 0.0005, cosine decay, and early stopping
with patience 5. Training stopped after 12 total epochs. The selected checkpoint is stored
at `runs/person_detection/mot17_a/weights/best.pt` and is intentionally ignored by Git.

The selected checkpoint did not improve the standard detector validation metrics:

| Detector | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| Pretrained YOLO26n | **0.696** | **0.485** | **0.564** | **0.299** |
| MOT17 experiment A | 0.699 | 0.436 | 0.530 | 0.279 |

Despite the lower detector mAP, the checkpoint produced substantially more matched MOT17
detections in the tracking pipeline. Continuity remained disabled for all tracking runs.

| Confidence | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | Precision | Recall | FPS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 43.723 | 47.390 | 41.216 | 43.030 | 46.184 | 1834 | 74.825 | **67.310** | **13.224** |
| 0.20 | 43.537 | **47.525** | 40.650 | 45.541 | 46.540 | 1831 | 78.304 | 65.250 | 7.607 |
| 0.30 | 44.840 | 47.446 | 43.065 | 46.922 | 49.190 | 1734 | 81.222 | 63.041 | 8.747 |
| 0.40 | **45.532** | 46.448 | **45.243** | **48.725** | **51.411** | **1134** | **86.266** | 59.152 | 8.205 |

Confidence 0.40 was also the strongest operating point for the fine-tuned checkpoint, but
the identity-switch count remained much higher than the pretrained detector's 329 switches
at the same threshold.

The previous Week 6 camera-specific regression failed:

| Detector | IN | OUT | Intrusion | Loitering | Raw tracks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Required production result | 4 | 3 | 1 | 1 | 132 baseline |
| MOT17 experiment A | 3 | 1 | 7 | 4 | 312 |

This prevents Experiment A from directly replacing the detector in the existing Week 6
configuration. Week 6 is retained as a regression for that specific camera setup, not as a
general-purpose benchmark. The runtime configuration continues to use `yolo26n.pt` until
the two holdout camera videos have ground-truth annotations and confirm the calibration
result.

## Initial Camera A/B

Three additional videos in `datasets/person_tracking/camera` were run with the pretrained
detector and Experiment A at confidence 0.40. Events, continuity, and duplicate suppression
were disabled so the comparison measured raw detector and ByteTrack behavior. At the time of
this initial run, the videos did not have bounding boxes or track-ID ground truth, so the
following values are diagnostic proxies rather than accuracy metrics.

| Video | Model | Avg active tracks | Total tracks | Churn/1k observations | Median observed frames | FPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1.mp4 | Pretrained | 8.996 | 63 | 6.927 | 91.0 | 10.351 |
| 1.mp4 | MOT17 A | 12.884 | 271 | 20.805 | 12.0 | 10.789 |
| 2.mp4 | Pretrained | 9.189 | 44 | 13.720 | 60.0 | 25.938 |
| 2.mp4 | MOT17 A | 13.934 | 100 | 20.563 | 20.5 | 25.383 |
| 3.mp4 | Pretrained | 18.842 | 63 | 9.805 | 71.0 | 24.348 |
| 3.mp4 | MOT17 A | 40.358 | 221 | 16.059 | 17.0 | 21.273 |

Visual inspection of representative frames confirms that Experiment A detects many real
small and distant pedestrians missed by the pretrained detector. It also produces much
higher track churn and overlapping tracks in crowds. The checkpoint is therefore useful as
a high-recall candidate, but it is not ready for promotion without annotated camera metrics
and tracker calibration. The next gate is to annotate these three videos, use one for
calibration, and reserve two for testing.

## Camera 1 Ground-Truth Calibration

`1.mp4` was annotated across all 1,011 frames in CVAT and exported as MOT 1.1. Validation
found 23,223 person boxes and 72 tracks. Fifteen boxes belonged to people leaving the image
and extended slightly beyond the left or bottom edge; the canonical benchmark copy clips
these boxes to the 3,840 x 2,160 image bounds while preserving the original export.

The first ground-truth sweep used the Week 6 ByteTrack parameters with continuity and event
logic excluded:

| Model | Confidence | HOTA | DetA | AssA | MOTA | IDF1 | Precision | Recall | IDSW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pretrained | 0.30 | **40.905** | **32.252** | 51.928 | **38.789** | **49.548** | 94.021 | **41.782** | 78 |
| Pretrained | 0.40 | 40.166 | 29.495 | **54.704** | 36.025 | 49.205 | 96.185 | 37.670 | 35 |
| Pretrained | 0.50 | 34.525 | 24.501 | 48.653 | 30.151 | 42.338 | **97.485** | 31.051 | **23** |
| MOT17 A | 0.30 | 28.190 | **26.740** | 31.151 | -3.380 | 30.184 | 48.724 | **39.388** | 306 |
| MOT17 A | 0.40 | **30.526** | 25.870 | 36.871 | 12.212 | 34.666 | 61.638 | 34.573 | 196 |
| MOT17 A | 0.50 | 29.325 | 23.216 | **37.503** | **18.680** | **34.678** | **73.719** | 29.665 | **95** |

The proxy increase in active tracks did not translate to better ground-truth accuracy.
Experiment A produced thousands more false positives and substantially weaker association
scores. The pretrained model remains the promotion choice. Confidence 0.30 is the strongest
camera-1 point by HOTA, MOTA, IDF1, and recall; confidence 0.40 is a reasonable lower-churn
alternative with 35 rather than 78 identity switches.

## Camera 3 Locked Holdout

The CVAT task was named `camera-2-test`, but its 341-frame count and visual content match
`3.mp4`; `2.mp4` has 349 frames and a different scene. The export is therefore recorded as
camera 3 ground truth rather than being evaluated against the wrong video. Validation found
10,986 person boxes and 65 tracks across all 341 frames, with no invalid or out-of-bounds
boxes.

The holdout used the confidence selected independently on camera 1: 0.30 for the pretrained
baseline and 0.40 for MOT17 A. No threshold was selected using camera 3 results.

| Model | Confidence | HOTA | DetA | AssA | MOTA | IDF1 | Precision | Recall | IDSW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pretrained | 0.30 | **57.349** | 50.171 | **65.805** | **63.790** | **77.504** | **95.304** | 67.240 | **15** |
| MOT17 A | 0.40 | 54.872 | **50.644** | 59.797 | 54.114 | 73.444 | 71.886 | **90.051** | 79 |

MOT17 A recovers substantially more people, but its 3,869 false positives versus 364 for the
baseline reduce MOTA and identity quality. Camera 3 therefore confirms the camera-1 model
selection: retain `yolo26n.pt`. The more complex `2.mp4` remains unannotated, so these results
must not be presented as validation for that scene. The production configuration is not
modified automatically.

Each regenerated camera report records the SHA-256 hashes of the video, ground truth,
predictions, configuration, tracker configuration, tracking implementation, evaluator,
benchmark script, and model checkpoints.
It also records the Git state, command, Python/package versions, CUDA version, and GPU. The
per-prediction TrackEval comparisons are stored under each report directory's `validation/`
folder and require an empty `differences` object to pass. A sanitized summary of the local
results and artifact hashes is tracked in [the evaluation evidence manifest](evaluation_evidence.md).

## Final Person-Tracking Regression

`configs/person_tracking_final.yaml` freezes the selected detector at confidence 0.30 with
the evaluated ByteTrack parameters. It is intentionally separate from `configs/week6.yaml`.
Camera-specific events, continuity, and duplicate suppression are disabled because no shared
line or zone geometry is valid across the three videos.

| Video | Frames | Avg active tracks | Total tracks | Churn/1k observations | Max gap | End-to-end FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1.mp4` | 1,011 | 10.208 | 107 | 10.368 | 69 | 10.206 |
| `2.mp4` | 349 | 10.415 | 59 | 16.231 | 72 | 18.975 |
| `3.mp4` | 341 | 22.730 | 82 | 10.579 | 49 | 13.897 |

All output videos contain the full source frame count, resolution, and frame rate. Visual
sampling confirms that boxes and trajectories are rendered correctly. Camera 2 has the
highest churn and longest gap, but without ground truth those values are only a qualitative
stress-test signal. Sustained batch speed varies with laptop power, temperature, and other
GPU workloads, so controlled isolated runs are used for the optimization comparison below.

## Throughput Optimization

Profiling on `2.mp4` showed inference as the largest stage and synchronous OpenCV MP4
encoding as the largest avoidable cost. The optimized pipeline keeps inference at FP32 and
640 pixels, overlaps encoding through a bounded writer queue, uses FFmpeg H.264 NVENC, and
reuses the loaded YOLO model while resetting ByteTrack state between videos.

| Input | Synchronous OpenCV | Async OpenCV | Async NVENC | Improvement |
| --- | ---: | ---: | ---: | ---: |
| 1080p, 25 FPS | 9.423 | 14.435 | **27.752** | **+194.5%** |
| 4K, 30 FPS | 5.042 | 10.097 | **16.853** | **+234.3%** |

The controlled 1080p run now exceeds its 25 FPS source rate. Full-resolution 4K improves by
more than three times but remains below 30 FPS because decode and inference are sequential.
Tracking outputs are unchanged: both A/B runs produced the same track count, observations,
churn, gaps, and maximum gap.

Alternatives that did not pass the quality/performance gate were excluded:

- `imgsz=576` increased isolated inference speed from 27.827 to 36.073 FPS but reduced HOTA
  from 57.349 to 53.928 and recall from 67.240% to 60.486% on camera 3.
- `imgsz=512` reduced HOTA to 45.553 and recall to 46.914%.
- FP16 did not provide stable acceleration and slightly changed predictions.
- Concurrent decode reduced 4K E2E throughput from 16.853 to 12.182 FPS due to CPU and
  preprocessing contention.
- Pre-resizing 4K frames to 1080p preserved metrics but added enough resize cost to reduce
  isolated throughput from 13.862 to 12.882 FPS.

`ffmpeg` must be available on `PATH`, and its encoder list must include `h264_nvenc`. Other
configs retain the portable synchronous OpenCV default.

## Reproduce

```powershell
python -m scripts.evaluate_mot17 `
  --dataset datasets/MOT17 `
  --config C:/VisionGuard/configs/week6.yaml `
  --output-dir data/mot17_benchmark/mot17_a/conf_040 `
  --variant FRCNN `
  --conf 0.40 `
  --model runs/person_detection/mot17_a/weights/best.pt `
  --disable-continuity
```

Generated predictions and detailed JSON/Markdown reports are written to
`data/mot17_benchmark/`, which is intentionally ignored by Git.

The unannotated camera comparison can be reproduced with:

```powershell
python -m scripts.compare_person_tracking_videos `
  --video-dir datasets/person_tracking/camera `
  --config C:/VisionGuard/configs/week6.yaml `
  --output-dir data/outputs/person_tracking_ab `
  --baseline-model yolo26n.pt `
  --candidate-model runs/person_detection/mot17_a/weights/best.pt `
  --conf 0.40
```

Prepare and evaluate the camera-1 ground truth with:

```powershell
python scripts/prepare_camera_ground_truth.py `
  --video datasets/person_tracking/camera/1.mp4 `
  --source-gt datasets/person_tracking/final_mot_1/gt/gt.txt `
  --output-dir datasets/person_tracking/camera_ground_truth/camera-1 `
  --name camera-1-calibration

python -m scripts.evaluate_camera_tracking `
  --video datasets/person_tracking/camera/1.mp4 `
  --ground-truth datasets/person_tracking/camera_ground_truth/camera-1/gt/gt.txt `
  --config configs/person_tracking_final.yaml `
  --output-dir data/outputs/camera_tracking_benchmark `
  --confidences 0.30 0.40 0.50
```

Prepare the correctly identified camera-3 export and run the locked holdout with:

```powershell
python scripts/prepare_camera_ground_truth.py `
  --video datasets/person_tracking/camera/3.mp4 `
  --source-gt datasets/person_tracking/camera-2-test-mot/gt/gt.txt `
  --output-dir datasets/person_tracking/camera_ground_truth/camera-3 `
  --name camera-3-test

python -m scripts.evaluate_camera_tracking `
  --video datasets/person_tracking/camera/3.mp4 `
  --ground-truth datasets/person_tracking/camera_ground_truth/camera-3/gt/gt.txt `
  --config configs/person_tracking_final.yaml `
  --output-dir data/outputs/camera_tracking_holdout/camera-3 `
  --baseline-confidence 0.30 `
  --candidate-confidence 0.40
```

Validate a prediction file against an official TrackEval checkout with:

```powershell
python scripts/validate_mot_evaluator.py `
  --ground-truth datasets/person_tracking/camera_ground_truth/camera-3/gt/gt.txt `
  --predictions data/outputs/camera_tracking_holdout/camera-3/predictions/baseline_conf_0p30.txt `
  --trackeval-root C:/path/to/TrackEval `
  --trackeval-python C:/path/to/python-with-scipy.exe
```

Run the final selected model on all camera videos with:

```powershell
python -m scripts.compare_person_tracking_videos `
  --video-dir datasets/person_tracking/camera `
  --config configs/person_tracking_final.yaml `
  --output-dir data/outputs/person_tracking_final `
  --baseline-model yolo26n.pt `
  --conf 0.30 `
  --baseline-only
```
