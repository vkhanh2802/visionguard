# VisionGuard Case Study

## Problem

Build a reviewable people-tracking system that turns local video into persistent events
and annotated evidence. The result needed to cover model inference, identity association,
camera-specific spatial rules, asynchronous execution, and a UI without hiding failure
states.

## Approach

VisionGuard uses YOLO26 for person detection and ByteTrack for identity association.
Tracks feed independent line-crossing, intrusion, and loitering state machines. FastAPI
assigns an immutable run ID and server-owned artifact path, Redis/RQ executes the video job, SQLite stores lifecycle
and analytics data, and React polls and displays the result. Each run receives its own
artifact directory.

## Evaluation

Two custom videos were annotated in CVAT and normalized to MOTChallenge format. Camera 1
was used for confidence calibration; camera 3 was kept as a locked holdout. A MOT17
fine-tuned checkpoint was compared with the pretrained baseline.

The custom evaluator initially undercounted identity matches and differed from TrackEval
in HOTA assignment and CLEAR continuity semantics. Synthetic regressions exposed the
errors. After correction, local validation found that all metrics and counts matched TrackEval commit
`12c8791b303e0a0b50f753af204249e622d0281a` on all eight regenerated prediction files.
The sanitized [evidence manifest](evaluation_evidence.md) records the metrics and artifact
hashes while excluding private media and model files.

| Holdout model | HOTA | MOTA | IDF1 | Precision | Recall | IDSW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pretrained YOLO26n, confidence 0.30 | **57.349** | **63.790** | **77.504** | **95.304** | 67.240 | **15** |
| MOT17 fine-tuned, confidence 0.40 | 54.872 | 54.114 | 73.444 | 71.886 | **90.051** | 79 |

The fine-tuned model detected more people but introduced enough false positives and ID
switches to lose overall. The pretrained checkpoint was retained. Rejecting the trained
candidate was an evaluation result, not a failed implementation.

## Performance Engineering

Profiling identified synchronous encoding as the largest avoidable cost after inference.
A bounded asynchronous writer and FFmpeg H.264 NVENC increased controlled throughput from
9.423 to 27.752 FPS at 1080p and from 5.042 to 16.853 FPS at 4K. The writer continuously
drains stderr, applies shutdown timeouts, kills hung subprocesses, and propagates failures
to the persisted run state.

## Reliability Work

- Server-owned `outputs/<run_id>/` artifacts prevent cross-run overwrite.
- Dashboard details refresh on lifecycle changes and stale state is cleared on selection.
- `/health` and `/readiness` separate process liveness from job-processing capability.
- Reports include hashes for inputs, checkpoints, configs, evaluator source, and scripts.
- CI compiles Python, runs the backend suite, installs the frontend lockfile, and builds it.

## Limitations

The service is designed for trusted local use and has no authentication or upload storage
boundary. Camera 2 remains unannotated. Geometry is camera-specific. NVENC has safe failure
handling but no automatic fallback. The project demonstrates an engineered prototype, not
a claim of production readiness or universal real-time performance.
