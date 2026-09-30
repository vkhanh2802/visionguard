# Evaluation Evidence Manifest

This document is a sanitized manifest of the local camera evaluation completed on
2026-09-30. Raw videos, annotations, model checkpoints, predictions, and generated reports
are intentionally excluded from Git because of privacy, licensing, and size constraints.
The hashes below identify the exact private artifacts used, but they do not make the
benchmark independently reproducible from this repository alone.

## TrackEval Validation

- Reference: [TrackEval](https://github.com/JonathonLuiten/TrackEval)
- Commit: `12c8791b303e0a0b50f753af204249e622d0281a`
- Checkout state: clean
- IoU threshold: `0.50`
- Compared fields: HOTA, DetA, AssA, LocA, MOTA, MOTP, IDF1, TP, FP, FN, ID switches, and fragmentations
- Result: all eight files produced an empty `differences` object

| Sequence | Model | Confidence | Prediction SHA-256 | HOTA | MOTA | IDF1 | IDSW | Match |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Camera 1 calibration | Pretrained YOLO26n | 0.30 | `b5b3063014eaa967a05ae697b3a8ec1533af1c28ffb5be6e8899bf399093ebc5` | 40.905 | 38.789 | 49.548 | 78 | exact |
| Camera 1 calibration | Pretrained YOLO26n | 0.40 | `cf8e5114d19d93ec1732cfc0445639cf6c62d9ebdb181e87e3693a50b6952ce7` | 40.166 | 36.025 | 49.205 | 35 | exact |
| Camera 1 calibration | Pretrained YOLO26n | 0.50 | `0dbcbdd7d5939dba17e92e84ea525099ddeffb6850550156773c0bdcdc297410` | 34.525 | 30.151 | 42.338 | 23 | exact |
| Camera 1 calibration | MOT17 fine-tuned | 0.30 | `73e97eba748da78bf3a32393cdc144a0bc4592ceeda363caa255669f4f289adf` | 28.190 | -3.380 | 30.184 | 306 | exact |
| Camera 1 calibration | MOT17 fine-tuned | 0.40 | `768d650fbf73a2f935a2586ac8884ebbe7ca1f0edac5bcb94a298f5accf27f19` | 30.526 | 12.212 | 34.666 | 196 | exact |
| Camera 1 calibration | MOT17 fine-tuned | 0.50 | `d40eec34950755172c7d2cb7ebe77cbab13adffd2b8d40f01d31522f803590cb` | 29.325 | 18.680 | 34.678 | 95 | exact |
| Camera 3 locked holdout | Pretrained YOLO26n | 0.30 | `904bb361a83f5ba19e9e7d5f0d47bf499ac972f68bf02517444f49ddff78f2bd` | 57.349 | 63.790 | 77.504 | 15 | exact |
| Camera 3 locked holdout | MOT17 fine-tuned | 0.40 | `aade93e6e34a37ecb1132434b69f75264d916c9859da7704878a87554a54e2ca` | 54.872 | 54.114 | 73.444 | 79 | exact |

The tracked tie-assignment fixture in `tests/fixtures/mot_assignment_tie/` additionally
covers a case where LAPJV and SciPy choose different optimal assignments. It matches the
pinned TrackEval checkout for every compared metric and count.

## Input Artifacts

| Artifact | SHA-256 |
| --- | --- |
| Camera 1 video | `b9e636a275aedb08999cbf21ed77c81a2ebeea117f4528aca849d8e1720ce7e4` |
| Camera 1 ground truth | `ee70b49d0206e39cb4c9b08cf09754779d9190a0f9db9e019dabd30766d279ac` |
| Camera 3 video | `8b277b8d03cf1770c3b933d23ad760e3d6a27413b3c88ceb1e79b4aa61ccab43` |
| Camera 3 ground truth | `b50aad4a0d6a20e092084951dc59a7d35232900382ecaadad1d7f0aca71d7e6a` |
| Pretrained YOLO26n checkpoint | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| MOT17 fine-tuned checkpoint | `b1d99134173966f3adec088a25a1f5b695d25c9fcc78cd781e90a56b9ac84e24` |
| Final tracking config | `9e612f7080c260eac3b53c1852e04ff4e8714fa1bc7de0a33a87fed2d84be0e0` |
| ByteTrack config | `7491720ab6866ff76f0d69174a31b9b668bd32889c5db02ed8be5b71da2a52c2` |

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
