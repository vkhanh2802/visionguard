# Week 6 Summary: Persistence And HTTP API

## Objective

Week 6 moved VisionGuard from file-oriented local execution to a persisted video
analytics service. The pipeline remains the single implementation of detection,
tracking, event processing, and visualization; SQLite and FastAPI are adapters around
that pipeline.

## Delivered

### SQLite Persistence

- Added `analysis_runs` for lifecycle, paths, resolved configuration, metrics,
  counters, and error messages.
- Added `events` with a foreign key to `analysis_runs.run_id` and indexes for run/time
  and event type queries.
- Added migration logic for the `error_message` column on existing databases.
- Added repository methods for run/event queries, analytics, health checks, and
  interrupted-run recovery.

### Pipeline Integration

- Added `RunRecorder` as the persistence boundary for emitted events and final result.
- Added `AnalysisService` so CLI and FastAPI execute the same pipeline orchestration.
- Created one UUID `run_id` per analysis and shared it between SQLite and optional
  JSONL artifacts.
- Persisted `completed` results and `failed` error messages through the same lifecycle.
- Kept JSONL and metadata as optional paired artifacts; SQLite is the source of truth.

### FastAPI

- Added health, runs, events, analysis, analytics, and output-download endpoints.
- Added Pydantic request/response schemas and pagination validation.
- Added `POST /analyze`, which accepts a local source/output/config request and starts
  inference in a background task.
- Added per-run analytics with event counts, unique tracks, timestamp range, and final
  counters.
- Added download for completed annotated videos.
- Added a root API directory and OpenAPI documentation.

### Reliability

- API startup marks runs left `running` by an interrupted API process as `failed`.
- Output download only succeeds for a completed run whose file still exists.
- Repository queries use SQLite parameter binding for request inputs.

## Final API Surface

```text
GET  /
GET  /health
GET  /runs
GET  /runs/{run_id}
GET  /runs/{run_id}/analytics
GET  /runs/{run_id}/output
GET  /events
POST /analyze
```

## Verification

The final Week 6 test suite passed:

```text
102 passed
```

Manual smoke checks also confirmed:

- CLI persistence to SQLite for a 722-frame video.
- API `POST /analyze` returns `202` and persists a completed 722-frame run.
- API analytics returns persisted aggregate data.
- API output download returned the generated MP4 as an attachment.
- API startup recovery changed a simulated interrupted run from `running` to `failed`.

## Operational Constraints

- The current API accepts local filesystem paths; it is intended for trusted local use.
- Analysis runs in a FastAPI in-process background task. Use one API worker.
- Restarting the API interrupts active inference. The run is marked failed at next
  startup and is not resumed.
- Input videos, generated outputs, model weights, and SQLite databases are ignored by
  Git.
- Camera-specific line and ROI coordinates remain configuration responsibilities.

## Follow-Up Work

- Move analysis jobs to a durable external worker/queue for multi-worker deployment.
- Add authentication and file-upload/storage controls before exposing the API beyond a
  trusted local environment.
- Add a frontend/dashboard using the persisted runs, events, analytics, and output
  endpoints.
- Re-evaluate detector/tracker tuning across additional cameras and longer videos.
