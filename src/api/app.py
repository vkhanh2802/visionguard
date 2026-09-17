from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query

from src.analysis_service import create_analysis_job, run_background_analysis
from src.api.dependencies import get_repository
from src.api.schemas import (
    AnalyzeAcceptedResponse,
    AnalyzeRequest,
    EventListResponse,
    EventResponse,
    HealthResponse,
    RunListResponse,
    RunResponse,
)
from src.database import SQLiteRepository
from src.config import AppConfig, load_config


def create_app(database_path: str | Path = "data/visionguard.db") -> FastAPI:
    app = FastAPI(
        title="VisionGuard API",
        version="0.1.0",
        description="Video analytics runs and events API.",
    )
    app.state.repository = SQLiteRepository(database_path)

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
        background_tasks: BackgroundTasks,
        repository: SQLiteRepository = Depends(get_repository),
    ) -> AnalyzeAcceptedResponse:
        try:
            config = _load_api_config(request.config_path)
            job = create_analysis_job(
                config=config,
                source_path=request.source_path,
                output_path=request.output_path,
                repository=repository,
            )
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        background_tasks.add_task(run_background_analysis, job)
        return AnalyzeAcceptedResponse(run_id=job.run_id, status="running")

    return app


def _load_api_config(config_path: str) -> AppConfig:
    config = load_config(config_path)
    config_data = config.model_dump(mode="python")
    config_data["output"]["display"] = False
    return AppConfig.model_validate(config_data)


app = create_app()
