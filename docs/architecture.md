# VisionGuard Architecture

## Overview

VisionGuard processes a local video into annotated output video, structured events,
and run metadata.

```text
CLI
  |
  v
YAML Config -> AppConfig
  |
  v
VideoPipeline
  |
  |-- OpenCV VideoCapture
  |-- YOLO + ByteTrack
  |-- TrackHistory
  |-- LineCrossingEngine
  |-- IntrusionEngine
  |-- LoiteringEngine
  |-- Visualization
  |-- OpenCV VideoWriter
  |
  v
PipelineResult + Event JSONL + Run Metadata JSON
```

## Module Responsibilities

| Module | Responsibility |
| ------ | -------------- |
| `src/config.py` | Loads YAML and validates application configuration with Pydantic. |
| `src/pipeline/video_pipeline.py` | Owns the video loop, runtime state, resources, event dispatch, and output result. |
| `src/tracking/` | Converts YOLO/ByteTrack output into `Track` objects and stores trajectories. |
| `src/events/` | Implements line-crossing, intrusion, loitering, geometry, and per-track state. |
| `src/visualization.py` | Draws overlays only; it does not contain event logic. |
| `src/event_jsonl.py` | Streams individual events to JSONL and writes run metadata. |
| `src/logging_config.py` | Configures readable application logging. |
| `scripts/run_video.py` | Parses CLI arguments, resolves overrides, creates the pipeline, and prints a summary. |

## Runtime Flow

```text
Frame
  -> YOLOByteTracker.track()
  -> Track[]
  -> TrackHistory.update()
  -> LineCrossingEngine.process()
  -> IntrusionEngine.process()
  -> LoiteringEngine.process()
  -> Event[]
  -> Event handler -> console logger + events.jsonl
  -> Visualization -> annotated output frame
```

Event engines receive `Track[]`, `frame_id`, and video timestamp. They do not
depend on OpenCV, command-line parsing, YAML files, or persistence.

## Configuration

YAML configuration is validated before the pipeline starts. Configuration covers
the model, detector confidence, tracking state retention, line geometry, ROI zones,
event thresholds, output settings, logging paths, and optional FPS override.

```text
CLI override > YAML config > model default
```

Week 4 baseline configurations are stored in:

```text
configs/week4_video_a.yaml
configs/week4_video_b.yaml
configs/week4_video_c.yaml
```

Line and ROI coordinates use the original video coordinate system. A configuration
must be changed or transformed when the camera view or input resolution changes.

## Pipeline API

`VideoPipeline` is independent of `argparse` and can be called directly by a future
FastAPI endpoint:

```python
from pathlib import Path

config = load_config("configs/default.yaml")
pipeline = VideoPipeline(config)

result = pipeline.run(
    source_path=Path("input.mp4"),
    output_path=Path("output.mp4"),
)
```

`PipelineResult` includes source and effective FPS, core-processing and end-to-end
throughput, processed frame count, event counters, output path, elapsed time, and
whether the user stopped the preview early.

Each `run()` creates a new tracker, trajectory history, and event-engine set. State
therefore does not leak from one analyzed video into the next.

## Timing

| Field | Meaning |
| ----- | ------- |
| `source_fps` | FPS declared by the input video. |
| `effective_fps` | Source FPS or validated `video.fps_override`; used for video timestamps. |
| `core_processing_fps` | Rolling throughput for tracking and event processing. |
| `end_to_end_fps` | Frame-loop throughput including read, events, visualization, and output writing. |

Event timestamps are based on source-video time, not wall-clock inference time.
If a source does not provide a valid FPS, the pipeline requires an explicit positive
FPS override instead of silently assuming a value.

## Persistence

Each run writes two structured artifacts configured under `logging`:

```text
events.jsonl
run_metadata.json
```

`events.jsonl` has one JSON object per emitted event. It supports streaming and
retains events already written if a later frame fails. Each record contains `run_id`,
`frame_id`, event type, track ID, video timestamp, position, optional direction,
zone ID, duration, and wall-clock log time.

After a successful run, `run_metadata.json` records the resolved config, result
summary, counters, FPS metrics, whether preview was stopped early, source path,
output path, and run ID. This is sufficient for a later API or database layer to
associate artifacts from the same run.

## API Boundary

A future FastAPI layer should load config, call `VideoPipeline.run()`, and return a
`PipelineResult` or persisted run metadata. It must not import `scripts/run_video.py`.
The CLI remains an adapter for local execution; OpenCV preview and console summary
belong outside the API layer.
