from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query

from src.api.dependencies import get_repository
from src.api.schemas import (
    EventListResponse,
    EventResponse,
    HealthResponse,
    RunListResponse,
    RunResponse,
)
from src.database import SQLiteRepository


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

    return app


app = create_app()
