import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from redis import Redis
from redis.exceptions import RedisError
from rq import Queue, Worker

from src.analysis_jobs import execute_queued_analysis
from src.analysis_service import scope_artifact_paths_for_run
from src.api.dependencies import get_queue, get_repository, get_settings
from src.api.settings import ApiSettings
from src.api.schemas import (
    ApiIndexResponse,
    AnalyzeAcceptedResponse,
    AnalyzeRequest,
    EventAnalyticsResponse,
    EventListResponse,
    EventResponse,
    HealthResponse,
    ReadinessResponse,
    RunAnalyticsResponse,
    RunListResponse,
    RunResponse,
    TrackingDiagnosticsResponse,
)
from src.database import SQLiteRepository
from src.config import AppConfig, load_config


def create_app(
    database_path: str | Path = "data/visionguard.db",
    settings: ApiSettings | None = None,
    queue: Queue | None = None,
) -> FastAPI:
    repository = SQLiteRepository(database_path)
    settings = settings or ApiSettings.from_environment()
    queue = queue or Queue(
        settings.queue_name,
        connection=Redis.from_url(
            settings.redis_url,
            socket_connect_timeout=2,
            socket_timeout=2,
        ),
    )

    app = FastAPI(
        title="VisionGuard API",
        version="0.1.0",
        description="Video analytics runs and events API.",
    )
    app.state.repository = repository
    app.state.settings = settings
    app.state.queue = queue
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/", response_model=ApiIndexResponse)
    def index() -> ApiIndexResponse:
        return ApiIndexResponse(
            service="VisionGuard API",
            docs_url="/docs",
            health_url="/health",
            readiness_url="/readiness",
            runs_url="/runs",
            events_url="/events",
        )

    @app.get("/health", response_model=HealthResponse)
    def health(
        repository: SQLiteRepository = Depends(get_repository),
    ) -> HealthResponse:
        if not repository.is_healthy():
            raise HTTPException(
                status_code=503,
                detail="Database is unavailable.",
            )

        return HealthResponse(status="ok", database="connected")

    @app.get("/readiness", response_model=ReadinessResponse)
    def readiness(
        repository: SQLiteRepository = Depends(get_repository),
        settings: ApiSettings = Depends(get_settings),
        queue: Queue = Depends(get_queue),
    ) -> ReadinessResponse:
        checks = _readiness_checks(repository, settings, queue)
        required_checks = (
            checks["database_writable"],
            checks["redis_connected"],
            checks["worker_available"],
            checks["output_writable"],
        )
        if not all(required_checks):
            raise HTTPException(status_code=503, detail=checks)
        return ReadinessResponse(status="ready", **checks)

    @app.get("/runs", response_model=RunListResponse)
    def list_runs(
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        repository: SQLiteRepository = Depends(get_repository),
    ) -> RunListResponse:
        runs = repository.list_runs(limit=limit, offset=offset)
        return RunListResponse(
            items=[RunResponse.model_validate(run) for run in runs],
            limit=limit,
            offset=offset,
        )

    @app.get("/runs/{run_id}", response_model=RunResponse)
    def get_run(
        run_id: str,
        repository: SQLiteRepository = Depends(get_repository),
    ) -> RunResponse:
        run = repository.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

        return RunResponse.model_validate(run)

    @app.get("/runs/{run_id}/output")
    def download_run_output(
        run_id: str,
        repository: SQLiteRepository = Depends(get_repository),
        settings: ApiSettings = Depends(get_settings),
    ) -> FileResponse:
        run = repository.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

        if run["status"] != "completed":
            raise HTTPException(
                status_code=409,
                detail=f"Output is unavailable while run status is {run['status']}.",
            )

        try:
            output_path = settings.output_path(run["output_path"])
        except ValueError as error:
            raise HTTPException(status_code=404, detail="Output file not found.") from error

        if not output_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Output file not found: {output_path}",
            )

        return FileResponse(path=output_path, filename=output_path.name)

    @app.get(
        "/runs/{run_id}/analytics",
        response_model=RunAnalyticsResponse,
    )
    def get_run_analytics(
        run_id: str,
        repository: SQLiteRepository = Depends(get_repository),
    ) -> RunAnalyticsResponse:
        analytics = repository.get_run_analytics(run_id)
        if analytics is None:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

        in_count = analytics["in_count"]
        out_count = analytics["out_count"]
        net_count = (
            int(in_count) - int(out_count)
            if in_count is not None and out_count is not None
            else None
        )

        return RunAnalyticsResponse(
            run_id=analytics["run_id"],
            status=analytics["status"],
            processed_frames=analytics["processed_frames"],
            in_count=in_count,
            out_count=out_count,
            net_count=net_count,
            intrusion_count=analytics["intrusion_count"],
            loitering_count=analytics["loitering_count"],
            events=EventAnalyticsResponse(
                recorded_event_count=analytics["recorded_event_count"],
                unique_track_count=analytics["unique_track_count"],
                first_event_timestamp=analytics["first_event_timestamp"],
                last_event_timestamp=analytics["last_event_timestamp"],
                by_type=analytics["event_counts"],
            ),
            tracking=(
                TrackingDiagnosticsResponse.model_validate(
                    analytics["tracking_diagnostics"]
                )
                if analytics["tracking_diagnostics"] is not None
                else None
            ),
        )

    @app.get("/events", response_model=EventListResponse)
    def list_events(
        run_id: str | None = None,
        event_type: str | None = None,
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        repository: SQLiteRepository = Depends(get_repository),
    ) -> EventListResponse:
        events = repository.list_events(
            run_id=run_id,
            event_type=event_type,
            limit=limit,
            offset=offset,
        )
        return EventListResponse(
            items=[EventResponse.model_validate(event) for event in events],
            limit=limit,
            offset=offset,
        )

    @app.post(
        "/analyze",
        response_model=AnalyzeAcceptedResponse,
        status_code=202,
    )
    def analyze(
        request: AnalyzeRequest,
        repository: SQLiteRepository = Depends(get_repository),
        settings: ApiSettings = Depends(get_settings),
        queue: Queue = Depends(get_queue),
    ) -> AnalyzeAcceptedResponse:
        try:
            source_path, config_path = settings.validate_analysis_paths(
                request.source_path,
                request.config_path,
            )
            config = _load_api_config(config_path)
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        run_id = str(uuid4())
        output_path = settings.run_output_path(run_id)
        try:
            config = scope_artifact_paths_for_run(config, output_path.parent)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        config_data = config.model_dump(mode="json")
        repository.create_run(
            run_id=run_id,
            source_path=source_path,
            output_path=output_path,
            config_data=config_data,
            status="queued",
        )

        try:
            queue.enqueue(
                execute_queued_analysis,
                run_id,
                config_data,
                str(source_path),
                str(output_path),
                str(repository.database_path),
                job_id=run_id,
                job_timeout=7200,
            )
        except RedisError as error:
            repository.fail_run(run_id, f"Failed to enqueue analysis: {error}")
            raise HTTPException(
                status_code=503,
                detail="Analysis queue is unavailable.",
            ) from error

        return AnalyzeAcceptedResponse(run_id=run_id, status="queued")

    return app


def _load_api_config(config_path: str) -> AppConfig:
    config = load_config(config_path)
    config_data = config.model_dump(mode="python")
    config_data["output"]["display"] = False
    return AppConfig.model_validate(config_data)


def _readiness_checks(
    repository: SQLiteRepository,
    settings: ApiSettings,
    queue: Queue,
) -> dict[str, bool]:
    redis_connected = False
    worker_available = False
    try:
        redis_connected = bool(queue.connection.ping())
        worker_available = _worker_available(queue)
    except RedisError:
        pass

    ffmpeg_available, nvenc_available = _ffmpeg_capabilities()
    return {
        "database_writable": repository.is_writable(),
        "redis_connected": redis_connected,
        "worker_available": worker_available,
        "output_writable": _directory_is_writable(settings.output_root),
        "ffmpeg_available": ffmpeg_available,
        "nvenc_available": nvenc_available,
    }


def _worker_available(queue: Queue) -> bool:
    for worker in Worker.all(queue=queue):
        state = worker.get_state()
        heartbeat = worker.last_heartbeat
        if heartbeat is None or state not in {"idle", "busy"}:
            continue
        if state == "busy":
            max_age_seconds = 2 * getattr(worker, "job_monitoring_interval", 30) + 60
        else:
            max_age_seconds = getattr(worker, "worker_ttl", 420) + 60
        heartbeat_age = (datetime.now(timezone.utc) - heartbeat).total_seconds()
        if heartbeat_age <= max_age_seconds:
            return True
    return False


def _directory_is_writable(path: Path) -> bool:
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path):
            pass
    except OSError:
        return False
    return True


@lru_cache(maxsize=1)
def _ffmpeg_capabilities() -> tuple[bool, bool]:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        return False, False
    try:
        completed = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return False, False
    if "h264_nvenc" not in completed.stdout:
        return True, False
    try:
        nvenc_probe = subprocess.run(
            [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=size=320x180:rate=1",
                "-frames:v",
                "1",
                "-c:v",
                "h264_nvenc",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return True, False
    return True, nvenc_probe.returncode == 0


app = create_app()
