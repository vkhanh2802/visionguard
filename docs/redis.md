# VisionGuard Redis Service

Redis is the queue broker for the RQ worker. SQLite remains the source of truth for
video-analysis runs, events, metrics, and errors.

## Start Redis

Open Docker Desktop, wait until it reports that the engine is running, then execute
from the repository root:

```powershell
docker compose up -d redis
docker compose ps
docker compose exec redis redis-cli ping
```

The final command must return:

```text
PONG
```

## Configuration

`compose.yaml` creates one local Redis container:

```text
Container: visionguard-redis
Address:   127.0.0.1:6379
Database:  0
Persistence volume: visionguard-redis-data
```

Redis binds only to localhost. It is not exposed to other devices on the network.
Append-only persistence is enabled, so queued jobs survive a Redis container restart.

## Lifecycle Commands

```powershell
# Show service status and health.
docker compose ps

# View Redis logs.
docker compose logs -f redis

# Stop the container while retaining queued data.
docker compose stop redis

# Start it again.
docker compose start redis

# Remove the container while retaining the named volume.
docker compose down
```

Do not run `docker compose down --volumes` unless queued Redis data can be discarded.

## Role In VisionGuard

```text
FastAPI -> enqueue job in Redis -> RQ worker -> VideoPipeline -> SQLite
```

FastAPI accepts and validates an analysis request, persists a `queued` run, and
enqueues the job. The separate RQ worker changes it to `running` and executes the
CPU/GPU-intensive video pipeline. Dashboard status continues to come from SQLite
rather than Redis.

## Start The Worker

After Redis is healthy, start a dedicated worker in another terminal:

```powershell
& "C:\Users\Khanh\miniconda3\envs\visionguard\python.exe" -m scripts.run_worker
```

On Windows the command uses RQ `SimpleWorker`, which does not depend on Unix process
forking APIs. On other platforms it uses the standard RQ worker. Keep this terminal
open while jobs are being processed.
