# VisionGuard 90-Second Demo Script

## Preparation

- Start Redis, one RQ worker, FastAPI, and the Vite dashboard.
- Open `/readiness`; confirm the service is `ready` and any capability required by the selected encoder is `true`.
- Use a short local input whose line/ROI coordinates match the selected config.
- Clear unrelated browser tabs and keep one completed run available as backup.

## Shot List

| Time | Screen | Narration |
| --- | --- | --- |
| 0:00-0:10 | Dashboard overview | VisionGuard turns local video into tracked events and reviewable run artifacts. |
| 0:10-0:22 | Submit source and config | FastAPI persists a queued run; the server owns the output directory. |
| 0:22-0:32 | Run changes queued to running | Redis/RQ keeps inference outside the request lifecycle. |
| 0:32-0:48 | Completed counters and event timeline | YOLO and ByteTrack feed line-crossing, intrusion, and loitering engines. |
| 0:48-1:02 | Tracking and timing diagnostics | The dashboard exposes identity gaps and stage timing instead of hiding bottlenecks. |
| 1:02-1:15 | Download/open annotated video | The downloaded video belongs to this run ID and cannot overwrite another run. |
| 1:15-1:27 | Benchmark table | Camera 3 holdout reached HOTA 57.349 and IDF1 77.504 with metrics validated against TrackEval. |
| 1:27-1:30 | Repository/CI | Tests, artifact hashes, and CI make the implementation reviewable. |

## Recording Notes

- Record at 1080p and keep the pointer movement deliberate.
- Do not claim stable real-time throughput; say "up to 27.752 FPS in a controlled 1080p run."
- Do not show private absolute paths, faces, credentials, or unlicensed footage in a public demo.
- If processing exceeds the recording window, cut from the running state to the prepared
  completed run and label the cut.
