# Evaluation Evidence Manifest

This document is a sanitized manifest of the local camera evaluations completed on
2026-09-30 and 2026-10-04, followed by manual shadow acceptance on 2026-10-06. Raw videos,
annotations, model checkpoints, predictions, and generated reports are intentionally
excluded from Git because of privacy, licensing, and size constraints.
The hashes below identify the exact private artifacts used, but they do not make the
benchmark independently reproducible from this repository alone.

## TrackEval Validation

- Reference: [TrackEval](https://github.com/JonathonLuiten/TrackEval)
- Commit: `12c8791b303e0a0b50f753af204249e622d0281a`
- Checkout state: clean
- IoU threshold: `0.50`
- Compared fields: HOTA, DetA, AssA, LocA, MOTA, MOTP, IDF1, TP, FP, FN, ID switches,
  and fragmentations
- Result: all 24 files produced an empty `differences` object: ten camera predictions and
  fourteen public MOT17 predictions

| Sequence | Model | Confidence | Prediction SHA-256 | HOTA | MOTA | IDF1 | IDSW | Match |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Camera 1 calibration | Pretrained YOLO26n | 0.30 | `b5b3063014eaa967a05ae697b3a8ec1533af1c28ffb5be6e8899bf399093ebc5` | 40.905 | 38.789 | 49.548 | 78 | exact |
| Camera 1 calibration | Pretrained YOLO26n | 0.40 | `cf8e5114d19d93ec1732cfc0445639cf6c62d9ebdb181e87e3693a50b6952ce7` | 40.166 | 36.025 | 49.205 | 35 | exact |
| Camera 1 calibration | Pretrained YOLO26n | 0.50 | `0dbcbdd7d5939dba17e92e84ea525099ddeffb6850550156773c0bdcdc297410` | 34.525 | 30.151 | 42.338 | 23 | exact |
| Camera 1 calibration | MOT17 fine-tuned | 0.30 | `73e97eba748da78bf3a32393cdc144a0bc4592ceeda363caa255669f4f289adf` | 28.190 | -3.380 | 30.184 | 306 | exact |
| Camera 1 calibration | MOT17 fine-tuned | 0.40 | `768d650fbf73a2f935a2586ac8884ebbe7ca1f0edac5bcb94a298f5accf27f19` | 30.526 | 12.212 | 34.666 | 196 | exact |
| Camera 1 calibration | MOT17 fine-tuned | 0.50 | `d40eec34950755172c7d2cb7ebe77cbab13adffd2b8d40f01d31522f803590cb` | 29.325 | 18.680 | 34.678 | 95 | exact |
| Camera 1 calibration | CrowdHuman fine-tuned | 0.55 | `d6ffca0ca0fbde1af424cd2841bfb02fd4fa9aad57378540183bcbb2a1a93e7e` | 43.946 | 37.502 | 54.621 | 81 | exact |
| Camera 3 locked holdout | Pretrained YOLO26n | 0.30 | `904bb361a83f5ba19e9e7d5f0d47bf499ac972f68bf02517444f49ddff78f2bd` | 57.349 | 63.790 | 77.504 | 15 | exact |
| Camera 3 locked holdout | MOT17 fine-tuned | 0.40 | `aade93e6e34a37ecb1132434b69f75264d916c9859da7704878a87554a54e2ca` | 54.872 | 54.114 | 73.444 | 79 | exact |
| Camera 3 locked holdout | CrowdHuman fine-tuned | 0.55 | `4b73970ed2f4f8629ce8cfd527d6b561666f11bf2b2afe84d634ee7f449bb730` | 65.696 | 77.681 | 86.777 | 16 | exact |

The tracked tie-assignment fixture in `tests/fixtures/mot_assignment_tie/` additionally
covers a case where LAPJV and SciPy choose different optimal assignments. It matches the
pinned TrackEval checkout for every compared metric and count.

## Locked Public MOT17 Gate

The restored MOT17 train/FRCNN ground truth covers seven sequences and 5,316 frames. The
thresholds below were locked before this gate. Each aggregate report contains the individual
prediction hashes and per-sequence TrackEval comparisons.

| Model | Confidence | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | TrackEval | Report SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Pretrained YOLO26n | 0.30 | 33.38 | 25.70 | 43.75 | 28.03 | 37.40 | 467 | 7/7 exact | `3bf3fd4f4f152afe3de4037d68d0c071baaa3754d2f5ecc7539a45320fd55c0b` |
| CrowdHuman fine-tuned | 0.55 | 39.62 | 33.08 | 47.91 | 39.46 | 48.79 | 351 | 7/7 exact | `5931ac6dabb83263f1e5ecc7d4c1bdc1e401d4f40c669ee4199d426e5d4fdf49` |

## Input Artifacts

| Artifact | SHA-256 |
| --- | --- |
| Camera 1 video | `b9e636a275aedb08999cbf21ed77c81a2ebeea117f4528aca849d8e1720ce7e4` |
| Camera 1 ground truth | `ee70b49d0206e39cb4c9b08cf09754779d9190a0f9db9e019dabd30766d279ac` |
| Camera 3 video | `8b277b8d03cf1770c3b933d23ad760e3d6a27413b3c88ceb1e79b4aa61ccab43` |
| Camera 3 ground truth | `b50aad4a0d6a20e092084951dc59a7d35232900382ecaadad1d7f0aca71d7e6a` |
| Pretrained YOLO26n checkpoint | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| MOT17 fine-tuned checkpoint | `b1d99134173966f3adec088a25a1f5b695d25c9fcc78cd781e90a56b9ac84e24` |
| CrowdHuman fine-tuned checkpoint | `c6a6fa5082445a7b2e4a87bf37f57cd2523652bdceadff8b8d09d72b7bcb9837` |
| CrowdHuman conversion report | `70e70457ea0ac967d44f513d7e9da499fc99148fd512fe1b7de0dd1af1f4bc3f` |
| CrowdHuman training arguments | `fadc69a4a8f950eb67788c2bf8b292541bd1751590d60abc90cfefd2afac1a9c` |
| CrowdHuman training results | `59c05728eb6ea93380750c0cdb0fdf4999c0641bdee85137a70fd4f3f9db1237` |
| Final tracking config | `5824c6d86564bb0ebd07e140ea56d1802109df149f4d3dc9a52bcf5ecff3da29` |
| ByteTrack config | `7491720ab6866ff76f0d69174a31b9b668bd32889c5db02ed8be5b71da2a52c2` |

## CrowdHuman Experiment

The local conversion contained 19,368 images and 439,000 full-body boxes. Two of 4,370
validation images were missing, and 127,455 ignored boxes were excluded from the YOLO
conversion. Five-epoch fine-tuning improved YOLO-native validation mAP50 from 0.491 to 0.773
and mAP50-95 from 0.245 to 0.461.

The original ODGT annotations were also evaluated with full-body Caltech matching and ignore
regions using the upstream `megvii-model/CrowdDetection` implementation at commit
`9786f58869a55af3e0b51fc78f8638a825dae4a2`. All 4,370 records were retained; the two
missing source images were assigned empty detection lists.

| Model | AP | mMR (lower is better) | Predictions SHA-256 | Report SHA-256 |
| --- | ---: | ---: | --- | --- |
| Pretrained YOLO26n | 0.440238 | 0.906192 | `6ad0f9584055bc5a1dfd07511d9d37465b2cb3c9c5cecce1feca14d465681994` | `51a13cc0d3ad92d31d1470380226738cf80c5e8947efdbd85917ead2d81b1c59` |
| CrowdHuman fine-tuned | 0.810561 | 0.597033 | `2725278429becb75c01792caac662f676c9b4b2a29d2360f4065b5bfafa29e1d` | `da51c066f1df6a5be4a577101d27fac9f910b6ee61cd37553326e30d24f1619e` |

Confidence 0.55 was selected on camera 1 before opening the camera-3 holdout. On camera 3,
the CrowdHuman checkpoint improved HOTA from 57.349 to 65.696, MOTA from 63.790 to 77.681,
and IDF1 from 77.504 to 86.777. The checkpoint remains outside Git. It was promoted to the
production configuration after the camera-2 shadow review was accepted.

## Camera 2 Shadow Run

Camera 2 has no ground truth. Its metrics are diagnostic only and cannot establish tracking
accuracy. The complete source is 349 frames at 25 FPS (13.96 seconds).

| Model | Confidence | Tracks | Avg active | Churn/1k observations | Max gap | E2E FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pretrained YOLO26n | 0.30 | 59 | 10.415 | 16.231 | 72 | 13.769 |
| CrowdHuman fine-tuned | 0.55 | 40 | 12.003 | 9.549 | 52 | 17.069 |

| Artifact | SHA-256 |
| --- | --- |
| Camera 2 source video | `58a9318f429fbe0db31841804568342d6872ad898bdb5a7615c01f5eb45cf165` |
| Shadow report | `1eece2fc59177d3f6f82583dca6e322495cb6a660107519bbbd91f6bcbed9809` |
| Baseline annotated video | `59e38970a9a69f68891aba2dd83d51243c6808fc55886bb746d9c97db4746076` |
| Candidate annotated video | `f03d461ac56e16e192c7eee591b6511792e4a7b9880bd46a47346854a95be76c` |
| Side-by-side video | `1436cee0b18835dd255f4ec45c5acf002a96ad31d95b8ecd62bc91bf32fec421` |
| Contact sheet | `34123e4699ed0bf21ac538cd8b8c4bd0955616034b7b1e62a6a984e72344ff3d` |
| Post-promotion production smoke video | `388d0630c7b1dff23b636bb73675013451ff6b88e23c09adb3e8bdde51d99a66` |

The completed [camera2_shadow_review.md](camera2_shadow_review.md) accepted the candidate.
It records one residual ID transfer at approximately 00:06-00:07 while the people were very
small in the frame; the reviewer judged the candidate acceptable despite that issue. A
post-promotion run using `configs/person_tracking_final.yaml` then completed all 349 frames
at the source 1920x1080 resolution and 25 FPS with the NVENC output backend.

## Controlled Throughput

The local optimization comparison used an NVIDIA GeForce RTX 3050 6 GB Laptop GPU with
the same model and tracking configuration for each encoder variant.

| Input | Synchronous OpenCV | Async OpenCV | Async NVENC |
| --- | ---: | ---: | ---: |
| 1080p, 25 FPS | 9.423 FPS | 14.435 FPS | 27.752 FPS |
| 4K, 30 FPS | 5.042 FPS | 10.097 FPS | 16.853 FPS |

These are disclosed local controlled runs, not a claim of stable real-time throughput.
The private source videos and generated timing reports are not distributed, so these
performance values cannot be independently rerun from the repository alone.
