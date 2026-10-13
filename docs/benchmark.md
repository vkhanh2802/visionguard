# Benchmark And Evaluation Evidence

This is the consolidated record of detector selection, tracking validation, event
experiments and performance engineering. Scores are local measured results. Raw media,
annotations, predictions, detailed reports and checkpoints are excluded from Git; hashes
identify original artifacts but do not make private experiments reproducible from a clone.

## 1. Selected Baseline

HOTA balances detection and association quality; MOTA penalizes missed detections, false
positives and identity switches; IDF1 measures identity matching. Tracking metric values
are expressed on a 0–100 scale, IDSW is a count, and detector AP/mMR are on a 0–1 scale.

| Setting | Value |
| --- | --- |
| Detector | YOLO26n fine-tuned on CrowdHuman for five epochs. |
| Checkpoint | `runs/person_detection/crowdhuman_a/weights/best.pt` |
| Runtime config | `configs/person_tracking_final.yaml` |
| Confidence | **0.55**, selected on camera 1 before the camera-3 comparison. |
| Input size | 640 pixels; FP32 inference. |
| Tracker | ByteTrack; `configs/bytetrack_week6.yaml`, `track_buffer: 75`. |
| Tracking benchmark | Events, continuity and duplicate suppression disabled. |
| Pretrained reference | `yolo26n.pt`, locked confidence 0.30. |

Earlier MOT17 confidence-sweep tables predated evaluator corrections. Their numbers are
historical and are not used here as final public benchmark evidence. The tables below use
the corrected evaluator and recorded validation scope.

## 2. Training And Detector Validation

CrowdHuman conversion yielded 15,000 training images and 4,368 available validation images,
with 439,000 full-body boxes. It excluded 127,455 ignored boxes from YOLO labels and clipped
68,002 boxes to image bounds. Two of the official 4,370 validation source images were missing.

Training used five epochs, 640-pixel images, batch 8, AdamW, initial learning rate 0.0005,
cosine decay, AMP, deterministic mode and seed 42. Epoch five was resumed from `last.pt`
after the first terminal run timed out; optimizer state was retained. The selected artifact
is `best.pt`.

| YOLO-native validation | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| Pretrained | 0.651 | 0.430 | 0.491 | 0.245 |
| CrowdHuman fine-tuned | **0.813** | **0.656** | **0.773** | **0.461** |

YOLO-native validation omits ignore regions. The original ODGT annotations were therefore
also evaluated with the upstream [CrowdDetection](https://github.com/megvii-model/CrowdDetection)
full-body Caltech matching protocol, pinned at
`9786f58869a55af3e0b51fc78f8638a825dae4a2`.

| Ignore-aware CrowdHuman evaluation | AP | Log-average miss rate (mMR) |
| --- | ---: | ---: |
| Pretrained | 0.440238 | 0.906192 |
| CrowdHuman fine-tuned | **0.810561** | **0.597033** |

AP is higher-is-better; mMR is lower-is-better. All 4,370 records were retained, with empty
detection lists for the two missing images. These detector score-curve metrics are separate
from tracking at the locked runtime confidence of 0.55.

## 3. Camera Calibration And Locked Holdout

Camera 1 has 1,011 annotated frames, 23,223 person boxes and 72 identities. Its canonical
copy clips 15 out-of-bounds boxes to the 3840×2160 frame while preserving the source export.
Camera 3 has 341 frames, 10,986 person boxes and 65 identities. An export named
`camera-2-test` was verified to belong to **3.mp4**, not the unannotated 349-frame **2.mp4**.

Candidate confidence was selected by HOTA, then IDF1, then MOTA on camera 1. The initial
sweep was extended when its optimum reached the 0.50 boundary:

| CrowdHuman confidence | HOTA | MOTA | IDF1 | Precision | Recall | IDSW |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.30 | 34.754 | 0.323 | 35.976 | 50.791 | 63.609 | 385 |
| 0.40 | 42.027 | 25.789 | 47.978 | 64.531 | 59.643 | 249 |
| 0.50 | 43.097 | 36.632 | 52.724 | 76.882 | 53.171 | 128 |
| **0.55** | **43.946** | **37.502** | **54.621** | 81.326 | 49.132 | 81 |
| 0.60 | 42.180 | 36.692 | 53.495 | 84.964 | 44.843 | 50 |
| 0.65 | 40.483 | 34.858 | 51.767 | 89.385 | 39.706 | 31 |
| 0.70 | 37.285 | 31.370 | 48.062 | 93.599 | 33.751 | 17 |

The thresholds were then locked for camera 3:

| Sequence | Detector | Conf | HOTA | MOTA | IDF1 | Precision | Recall | IDSW |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Camera 1 | Pretrained | 0.30 | 40.905 | **38.789** | 49.548 | **94.021** | 41.782 | **78** |
| Camera 1 | MOT17 fine-tuned | 0.40 | 30.526 | 12.212 | 34.666 | 61.638 | 34.573 | 196 |
| Camera 1 | CrowdHuman fine-tuned | 0.55 | **43.946** | 37.502 | **54.621** | 81.326 | **49.132** | 81 |
| Camera 3 | Pretrained | 0.30 | 57.349 | 63.790 | 77.504 | **95.304** | 67.240 | **15** |
| Camera 3 | MOT17 fine-tuned | 0.40 | 54.872 | 54.114 | 73.444 | 71.886 | **90.051** | 79 |
| Camera 3 | CrowdHuman fine-tuned | 0.55 | **65.696** | **77.681** | **86.777** | 89.496 | 88.176 | 16 |

On camera 1, CrowdHuman raises HOTA by 3.041 and IDF1 by 5.073 points, while MOTA drops
1.287 points and identity switches rise by three. On camera 3, HOTA rises 8.347, MOTA 13.891,
IDF1 9.273 and recall 20.936 points. False negatives fall from 3,599 to 1,299, while false
positives rise from 364 to 1,137. This is a recall/false-positive trade-off, not improvement
on every metric.

The earlier MOT17 fine-tune was rejected: it recovered more pedestrians but degraded
camera accuracy and identity quality. Better training-set tracking numbers alone did not
justify replacing the pretrained reference. CrowdHuman became the selected baseline after
the annotated-camera, public-dataset and manual shadow checks.

## 4. Public MOT17 Tracking Gate

All seven MOT17 train/FRCNN sequences (02, 04, 05, 09, 10, 11, 13) were rerun, totaling
5,316 frames. FRCNN folders avoid evaluating the identical source frames again under DPM/SDP.
The detector is YOLO, not the supplied FRCNN detections. Confidence stayed locked to the
camera-1 choices, and camera-specific event/continuity rules were disabled.

| Detector | Conf | HOTA | DetA | AssA | MOTA | IDF1 | IDSW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pretrained | 0.30 | 33.38 | 25.70 | 43.75 | 28.03 | 37.40 | 467 |
| CrowdHuman fine-tuned | 0.55 | **39.62** | **33.08** | **47.91** | **39.46** | **48.79** | **351** |

HOTA improves 6.24, MOTA 11.43 and IDF1 11.39 points; ID switches decrease by 116. Every
one of the fourteen model/sequence prediction files matched the pinned TrackEval reference.
This is evaluation on public **train** ground truth, not a MOTChallenge test-server result.

## 5. Camera 2 Shadow Review And Promotion

Camera 2 is a 1920×1080 station clip with 349 frames at 25 FPS (13.96 seconds) and **no
tracking ground truth**. Its shadow metrics are runtime diagnostics only:

| Detector | Conf | Total IDs | Average active | Churn/1k observations | Max gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pretrained | 0.30 | 59 | 10.415 | 16.231 | 72 |
| CrowdHuman fine-tuned | 0.55 | 40 | 12.003 | 9.549 | 52 |

Khanh completed the full side-by-side review and accepted all six relative checks:
candidate-only detections represented people, no persistent duplicate was introduced,
obvious baseline-tracked people were not persistently missed, ID resets/transfers were not
worse overall, boxes remained usable on difficult people, and no important region regressed.
One ID transfer to the wrong person at approximately **00:06–00:07**, involving very small
people, was explicitly accepted as a residual issue.

The selected checkpoint/confidence were promoted to `person_tracking_final.yaml`. A
post-promotion NVENC smoke run preserved all 349 frames, source resolution and 25 FPS.
Manual acceptance is a qualitative regression check; lower churn does not establish
accuracy on this scene.

## 6. Historical Event Evaluation And Engineering Decisions

The earlier pretrained event configs were manually evaluated on three selected clips:

| Clip | GT intrusion | Initial predicted intrusion | Stabilized predicted intrusion | Predicted loitering |
| --- | ---: | ---: | ---: | ---: |
| A | 6 | 5 | 5 | 0 |
| B | 10 | 12 | 10 | 1 |
| C | 2 | 2 | 2 | 0 |

Initial manual intrusion matching reported TP=17, FP=2, FN=1: precision 89.47%, recall
94.44%, micro F1 91.89%. The single positive loitering event was detected. The stabilization
rerun used six consecutive inside/outside observations and reduced B's count from 12 to 10.
Event timestamps and matching tolerances were not retained, so count agreement does not
justify replacing the original FP/FN metrics or claiming broad loitering accuracy. Clip C's
historical ROI extended outside its image. These are exploratory event results, separate
from the promoted detector's annotated tracking benchmarks.

Main lessons retained from the experiments:

- Merged detections, missing people and ID fragmentation propagate into event errors.
- A finite line segment, 5-pixel dead zone and three-frame side confirmation reduce
  crossing jitter; excessive confirmation/dead zones can miss brief crossings.
- Position-based continuity and duplicate heuristics are camera-specific and remain off
  in the selected general tracking baseline.
- Reduced detector input sizes and FP16 did not pass the tested quality/performance trade-off;
  the selected tracking workflow retains 640-pixel FP32 inference.

### Controlled Encoder Throughput

Selected isolated runs on an RTX 3050 6 GB Laptop GPU compared the same model/tracker across
writers; tracking counts/diagnostics were unchanged between encoder variants.

| Input | Sync OpenCV | Async OpenCV | Async NVENC |
| --- | ---: | ---: | ---: |
| 1080p, 25 FPS | 9.423 FPS | 14.435 FPS | 27.752 FPS |
| 4K, 30 FPS | 5.042 FPS | 10.097 FPS | 16.853 FPS |

These historical encoder measurements are not a stable-throughput guarantee for every
scene or the promoted checkpoint. End-to-end speed depends on decoding, density, encoding,
thermal state and GPU contention; stage timing distinguishes those costs.

## 7. Evaluator Integrity

The MOT evaluator scores valid marked pedestrian annotations after distractor preprocessing.
Sequence aggregation pools matching and association statistics rather than taking a simple
arithmetic mean of sequence scores.

- Reference: [TrackEval](https://github.com/JonathonLuiten/TrackEval), clean checkout at
  `12c8791b303e0a0b50f753af204249e622d0281a`.
- CLEAR/identity IoU threshold: 0.50. HOTA uses its threshold-averaged protocol.
- Compared fields: HOTA, DetA, AssA, LocA, MOTA, MOTP, IDF1, TP, FP, FN, ID switches
  and fragmentations.
- Ten camera predictions plus fourteen public MOT17 predictions produced `differences={}`.
- `tests/fixtures/mot_assignment_tie/` preserves a regression where LAPJV and SciPy choose
  different optimal assignments; the implementation matches the pinned reference.

Reports record hashes for inputs, predictions, configs, checkpoints and implementation,
plus the command, Git state and environment. Only the selected CrowdHuman camera points
were reference-validated; the entire confidence sweep is not claimed as independently
checked against TrackEval.

## Reproduce

Use the environment in the root README. Camera commands need the original private media
and correctly identified MOT ground truth. Public datasets must be acquired separately.
Paths below follow the original local layout; `crownhuman` is the retained directory name.

### Prepare And Train

```powershell
python -m scripts.prepare_person_dataset `
  --sources crowdhuman `
  --crowdhuman-root datasets/person_tracking/crownhuman `
  --output-root datasets/yolo --link-mode hardlink

yolo detect train `
  model=yolo26n.pt data=datasets/yolo/crowdhuman/data.yaml `
  epochs=5 imgsz=640 batch=8 device=0 workers=0 `
  optimizer=AdamW lr0=0.0005 lrf=0.1 cos_lr=True `
  warmup_epochs=1 patience=3 close_mosaic=2 `
  seed=42 deterministic=True amp=True max_det=500 `
  project=runs/person_detection name=crowdhuman_a
```

A retraining run need not produce the identical checkpoint/hash, even with the documented
seed. Replaying scored predictions requires the original artifacts or fresh evaluations.

### Camera Calibration And Locked Holdout

```powershell
python -m scripts.evaluate_camera_tracking `
  --video datasets/person_tracking/camera/1.mp4 `
  --ground-truth datasets/person_tracking/camera_ground_truth/camera-1/gt/gt.txt `
  --config configs/person_tracking_final.yaml `
  --output-dir data/outputs/crowdhuman_a/camera-1-calibration `
  --candidate-model runs/person_detection/crowdhuman_a/weights/best.pt `
  --candidate-label crowdhuman_a `
  --confidences 0.30 0.40 0.50 0.55 0.60 0.65 0.70

python -m scripts.evaluate_camera_tracking `
  --video datasets/person_tracking/camera/3.mp4 `
  --ground-truth datasets/person_tracking/camera_ground_truth/camera-3/gt/gt.txt `
  --config configs/person_tracking_final.yaml `
  --output-dir data/outputs/crowdhuman_a/camera-3-holdout `
  --candidate-model runs/person_detection/crowdhuman_a/weights/best.pt `
  --candidate-label crowdhuman_a `
  --baseline-confidence 0.30 --candidate-confidence 0.55
```

Use `scripts/prepare_camera_ground_truth.py` to validate/normalize CVAT exports before scoring.

### Public MOT17 And Reference Validation

```powershell
python -m scripts.evaluate_mot17 `
  --dataset datasets/person_tracking/MOT17 `
  --config configs/person_tracking_final.yaml `
  --output-dir data/mot17_benchmark/crowdhuman_gate/crowdhuman_a_conf_055 `
  --variant FRCNN --conf 0.55 `
  --model runs/person_detection/crowdhuman_a/weights/best.pt --disable-continuity

python scripts/validate_mot_evaluator.py `
  --ground-truth datasets/person_tracking/camera_ground_truth/camera-3/gt/gt.txt `
  --predictions data/outputs/crowdhuman_a/camera-3-holdout/predictions/crowdhuman_a_conf_0p55.txt `
  --trackeval-root C:/path/to/TrackEval `
  --trackeval-python C:/path/to/python-with-scipy.exe
```

For the pretrained MOT17 reference, use `--model yolo26n.pt --conf 0.30` and a separate
output directory. Pin TrackEval to the commit recorded above before comparison.

### CrowdHuman Score-Curve Evaluation

```powershell
yolo detect val `
  model=runs/person_detection/crowdhuman_a/weights/best.pt `
  data=datasets/yolo/crowdhuman/data.yaml `
  imgsz=640 batch=8 device=0 workers=0 conf=0.001 max_det=500 save_json=True `
  project=runs/person_detection name=crowdhuman_official_candidate

python -m scripts.evaluate_crowdhuman `
  --ground-truth datasets/person_tracking/crownhuman/annotation_val.odgt `
  --predictions runs/person_detection/crowdhuman_official_candidate/predictions.json `
  --images-dir datasets/person_tracking/crownhuman/CrowdHuman_val/Images `
  --evaluator-root C:/path/to/CrowdDetection `
  --output-dir data/outputs/crowdhuman_a/official-evaluation/crowdhuman_a
```

Clone CrowdDetection from the linked upstream and check out its pinned commit before
evaluation. Use the actual `predictions.json` path printed by the installed Ultralytics CLI.

## Evidence Identifiers

<details>
<summary>Expand the SHA-256 manifest for inputs, predictions and generated reports</summary>

### Inputs And Training

| Artifact | SHA-256 |
| --- | --- |
| Pretrained checkpoint | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| MOT17 experiment-A checkpoint | `b1d99134173966f3adec088a25a1f5b695d25c9fcc78cd781e90a56b9ac84e24` |
| Selected CrowdHuman checkpoint | `c6a6fa5082445a7b2e4a87bf37f57cd2523652bdceadff8b8d09d72b7bcb9837` |
| Selected tracking config | `5824c6d86564bb0ebd07e140ea56d1802109df149f4d3dc9a52bcf5ecff3da29` |
| ByteTrack config | `7491720ab6866ff76f0d69174a31b9b668bd32889c5db02ed8be5b71da2a52c2` |
| Camera 1 video | `b9e636a275aedb08999cbf21ed77c81a2ebeea117f4528aca849d8e1720ce7e4` |
| Camera 1 ground truth | `ee70b49d0206e39cb4c9b08cf09754779d9190a0f9db9e019dabd30766d279ac` |
| Camera 3 video | `8b277b8d03cf1770c3b933d23ad760e3d6a27413b3c88ceb1e79b4aa61ccab43` |
| Camera 3 ground truth | `b50aad4a0d6a20e092084951dc59a7d35232900382ecaadad1d7f0aca71d7e6a` |
| CrowdHuman conversion report | `70e70457ea0ac967d44f513d7e9da499fc99148fd512fe1b7de0dd1af1f4bc3f` |
| CrowdHuman training arguments | `fadc69a4a8f950eb67788c2bf8b292541bd1751590d60abc90cfefd2afac1a9c` |
| CrowdHuman training results | `59c05728eb6ea93380750c0cdb0fdf4999c0641bdee85137a70fd4f3f9db1237` |

### TrackEval-Checked Camera Predictions

| Sequence | Detector / confidence | Prediction SHA-256 |
| --- | --- | --- |
| Camera 1 | Pretrained / 0.30 | `b5b3063014eaa967a05ae697b3a8ec1533af1c28ffb5be6e8899bf399093ebc5` |
| Camera 1 | Pretrained / 0.40 | `cf8e5114d19d93ec1732cfc0445639cf6c62d9ebdb181e87e3693a50b6952ce7` |
| Camera 1 | Pretrained / 0.50 | `0dbcbdd7d5939dba17e92e84ea525099ddeffb6850550156773c0bdcdc297410` |
| Camera 1 | MOT17 A / 0.30 | `73e97eba748da78bf3a32393cdc144a0bc4592ceeda363caa255669f4f289adf` |
| Camera 1 | MOT17 A / 0.40 | `768d650fbf73a2f935a2586ac8884ebbe7ca1f0edac5bcb94a298f5accf27f19` |
| Camera 1 | MOT17 A / 0.50 | `d40eec34950755172c7d2cb7ebe77cbab13adffd2b8d40f01d31522f803590cb` |
| Camera 1 | CrowdHuman / 0.55 | `d6ffca0ca0fbde1af424cd2841bfb02fd4fa9aad57378540183bcbb2a1a93e7e` |
| Camera 3 | Pretrained / 0.30 | `904bb361a83f5ba19e9e7d5f0d47bf499ac972f68bf02517444f49ddff78f2bd` |
| Camera 3 | MOT17 A / 0.40 | `aade93e6e34a37ecb1132434b69f75264d916c9859da7704878a87554a54e2ca` |
| Camera 3 | CrowdHuman / 0.55 | `4b73970ed2f4f8629ce8cfd527d6b561666f11bf2b2afe84d634ee7f449bb730` |

### Public Gate, Official Evaluation And Shadow Artifacts

| Artifact | SHA-256 |
| --- | --- |
| MOT17 pretrained aggregate report | `3bf3fd4f4f152afe3de4037d68d0c071baaa3754d2f5ecc7539a45320fd55c0b` |
| MOT17 CrowdHuman aggregate report | `5931ac6dabb83263f1e5ecc7d4c1bdc1e401d4f40c669ee4199d426e5d4fdf49` |
| CrowdHuman pretrained predictions | `6ad0f9584055bc5a1dfd07511d9d37465b2cb3c9c5cecce1feca14d465681994` |
| CrowdHuman selected-model predictions | `2725278429becb75c01792caac662f676c9b4b2a29d2360f4065b5bfafa29e1d` |
| CrowdHuman pretrained official report | `51a13cc0d3ad92d31d1470380226738cf80c5e8947efdbd85917ead2d81b1c59` |
| CrowdHuman selected-model official report | `da51c066f1df6a5be4a577101d27fac9f910b6ee61cd37553326e30d24f1619e` |
| Camera 2 source | `58a9318f429fbe0db31841804568342d6872ad898bdb5a7615c01f5eb45cf165` |
| Camera 2 shadow report | `1eece2fc59177d3f6f82583dca6e322495cb6a660107519bbbd91f6bcbed9809` |
| Camera 2 pretrained annotated video | `59e38970a9a69f68891aba2dd83d51243c6808fc55886bb746d9c97db4746076` |
| Camera 2 selected-model annotated video | `f03d461ac56e16e192c7eee591b6511792e4a7b9880bd46a47346854a95be76c` |
| Camera 2 side-by-side video | `1436cee0b18835dd255f4ec45c5acf002a96ad31d95b8ecd62bc91bf32fec421` |
| Camera 2 contact sheet | `34123e4699ed0bf21ac538cd8b8c4bd0955616034b7b1e62a6a984e72344ff3d` |
| Post-promotion camera-2 smoke video | `388d0630c7b1dff23b636bb73675013451ff6b88e23c09adb3e8bdde51d99a66` |

The public MOT17 aggregate reports retain per-sequence prediction hashes. Detailed generated
files may have been cleaned from the workspace; benchmark code, this sanitized manifest and
the recorded decisions remain the public evidence.

</details>
