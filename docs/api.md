# VisionGuard API Guide

## Start The Server

Run from the project root after activating the `visionguard` Conda environment:

```bash
python -m uvicorn src.api.app:app --reload
```

Swagger UI is available at `http://127.0.0.1:8000/docs`.

## Run Lifecycle

```text
POST /analyze
  -> analysis_runs.status = queued
  -> FastAPI enqueues a Redis job
  -> RQ worker changes status to running and runs VideoPipeline
  -> emitted events are inserted into SQLite
  -> completed: metrics and counters are written
  -> failed: error_message is written
```

`POST /analyze` returns before inference finishes. Use the returned `run_id` with
`GET /runs/{run_id}` to observe `queued`, `running`, `completed`, or `failed` status.

FastAPI and the RQ worker run as separate processes. Redis retains queued jobs across
an API restart. Keep one RQ worker running for local operation; on Windows VisionGuard
uses a heartbeat-enabled RQ `SimpleWorker` to avoid Unix-only fork APIs while keeping
long-running jobs visible to readiness checks.

## Endpoints

| Method | Path | Success | Description |
| ------ | ---- | ------- | ----------- |
| `GET` | `/` | `200` | Service name and main endpoint links. |
| `GET` | `/health` | `200` | SQLite connectivity status. |
| `GET` | `/readiness` | `200`/`503` | Job-processing dependency status. |
| `GET` | `/runs` | `200` | Paginated runs ordered newest first. |
| `GET` | `/runs/{run_id}` | `200` | One persisted run. |
| `GET` | `/runs/{run_id}/analytics` | `200` | Counters and aggregate event metrics. |
| `GET` | `/runs/{run_id}/output` | `200` | Download completed annotated video. |
| `GET` | `/events` | `200` | Paginated events. |
| `POST` | `/analyze` | `202` | Create and start an analysis job. |

`/readiness` returns `503` when database writes, Redis, a current RQ worker heartbeat, or
output storage is unavailable. `ffmpeg_available` and `nvenc_available` are informational
capability fields because each analysis config selects its own encoder; an OpenCV-backed job
does not require either capability. The NVENC result comes from a cached one-frame encode
probe, not only from FFmpeg's encoder list.

## Create An Analysis Job

```http
POST /analyze
Content-Type: application/json

{
  "source_path": "C:/VisionGuard/videos/test.mp4",
  "config_path": "C:/VisionGuard/configs/default.yaml"
}
```

`config_path` is optional and defaults to `C:/VisionGuard/configs/default.yaml`. The API always
disables the OpenCV preview window, regardless of the YAML `output.display` setting.

For trusted local use, the default path policy permits:

```text
Requested source videos: C:\VisionGuard\videos\
Server-owned output videos: C:\VisionGuard\outputs\<run_id>\annotated.mp4
Requested configuration: C:\VisionGuard\configs\
```

The API resolves every requested source/config path and rejects paths outside these roots
with `422`. It generates the output path after assigning the run ID, so concurrent runs
cannot select or overwrite a shared filename.
Override the defaults before starting Uvicorn when another local storage layout is
needed:

```powershell
$env:VISIONGUARD_SOURCE_ROOT = "C:\camera-input"
$env:VISIONGUARD_OUTPUT_ROOT = "C:\camera-output"
$env:VISIONGUARD_CONFIG_ROOT = "C:\visionguard-configs"
$env:VISIONGUARD_ALLOWED_ORIGINS = "http://127.0.0.1:5173"
```

Successful acceptance:

```json
{
  "run_id": "018f8ed4-2c5a-7baa-8fe4-184cf0e83afc",
  "status": "queued"
}
```

## Query Runs And Events

```text
GET /runs?limit=100&offset=0
GET /runs/{run_id}
GET /events?run_id={run_id}&event_type=intrusion&limit=100&offset=0
```

`limit` must be from `1` to `500`; `offset` must be zero or greater. Event filters are
optional. All API read endpoints use SQLite and do not read JSONL artifacts.

## Analytics

```text
GET /runs/{run_id}/analytics
```

The response combines final counters from `analysis_runs`, aggregate event data, and
tracking diagnostics collected from detections in every processed frame. Existing
runs completed before diagnostics were added return `"tracking": null`.

```json
{
  "run_id": "018f8ed4-2c5a-7baa-8fe4-184cf0e83afc",
  "status": "completed",
  "processed_frames": 722,
  "in_count": 2,
  "out_count": 1,
  "net_count": 1,
  "intrusion_count": 3,
  "loitering_count": 0,
  "events": {
    "recorded_event_count": 5,
    "unique_track_count": 3,
    "first_event_timestamp": 4.63,
    "last_event_timestamp": 9.73,
    "by_type": {
      "intrusion": 3,
      "line_crossing": 2
    }
  },
  "tracking": {
    "total_track_count": 12,
    "new_track_count_in_roi": 3,
    "tracks_with_gaps": 4,
    "total_missing_frames": 17,
    "max_gap_frames": 6,
    "median_observed_frames": 42.5,
    "track_lifetimes": [],
    "timing": {
      "read_seconds": 1.2,
      "tracking_seconds": 18.7,
      "analytics_seconds": 0.4,
      "drawing_seconds": 1.1,
      "write_enqueue_seconds": 0.2,
      "encoding_seconds": 5.8,
      "writer_flush_seconds": 0.3,
      "frame_loop_seconds": 22.1,
      "tracking_fps": 38.6
    },
    "continuity": {
      "replacement_match_count": 2,
      "pending_match_count": 3,
      "rejected_ambiguous_match_count": 1,
      "rejected_gap_match_count": 4
    },
    "deduplication": {
      "confirmed_pair_count": 2,
      "pending_duplicate_observation_count": 4,
      "suppressed_observation_count": 17,
      "ambiguous_pair_count": 0,
      "handoff_count": 0,
      "confirmed_pairs": []
    }
  }
}
```

The optional `tracking.continuity` section reports replacement-ID decisions. A match is
only promoted after consecutive compatible observations; ambiguous nearby-person and
overlong-gap candidates are left as new tracks instead of being merged. When two boxes
remain visible at the same time, continuity deliberately does not merge them; that case
belongs to duplicate-detection/event suppression rather than lost-track reassociation.

The optional `tracking.deduplication` section reports overlapping active-track decisions.
Pending duplicate candidates are quarantined from event processing during confirmation;
raw tracking metrics and annotated boxes remain unchanged. State retention controls
confirmation across short gaps independently from handoff, which can be disabled when a
duplicate must never inherit the primary event ID after the primary disappears. Week 6
allows a 5-frame handoff for confirmed pairs to preserve one visit across a short primary
track gap. The camera-specific loitering configuration also keeps a visit eligible for
reassociation for up to 750 frames within a 60-pixel position gate, preventing a long
chain of IDs for one person from generating repeated loitering events.

## Download Output

```text
GET /runs/{run_id}/output
```

The endpoint returns the annotated output file as an attachment only when the run is
`completed`. It returns `409` for `running` or `failed` runs, and `404` if the run or
its output file does not exist.

## Error Behavior

| Situation | Status | Response behavior |
| --------- | -----: | ----------------- |
| Unknown run | `404` | `Run not found: {run_id}`. |
| Invalid pagination | `422` | FastAPI validation response. |
| Missing or invalid analysis config | `422` | Configuration error details. |
| Output requested before completion | `409` | Output unavailable for current run status. |
| Completed run whose output was deleted | `404` | Output file not found. |
| Required processing dependency unavailable | `503` | `/readiness` returns every dependency and capability check. |

## Persistence

SQLite at `data/visionguard.db` is the source of truth. It contains:

```text
analysis_runs: lifecycle, config, paths, metrics, counters, errors
```

JSONL event and metadata files are optional exports. Configure both logging paths to
enable them; they share the SQLite `run_id` but are not read by the API. For API and
dashboard submissions, the API creates one directory per `run_id`, so reruns never
overwrite a prior run's artifacts.

## Browser Access

The API allows CORS requests from the Vite development dashboard at
`http://127.0.0.1:5173` and `http://localhost:5173`. Configure
`VISIONGUARD_ALLOWED_ORIGINS` with a comma-separated list before serving the dashboard
from another origin. The Vite development server also proxies `/api` to FastAPI, so the
default local dashboard works without a browser cross-origin request.
