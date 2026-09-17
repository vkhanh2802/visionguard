from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"]
    database: Literal["connected"]


class RunResponse(BaseModel):
    run_id: str
    source_path: str
    output_path: str
    status: Literal["running", "completed", "failed"]
    created_at: str
    completed_at: str | None
    error_message: str | None
    processed_frames: int | None
    source_fps: float | None
    effective_fps: float | None
    core_processing_fps: float | None
    end_to_end_fps: float | None
    elapsed_seconds: float | None
    stopped_early: bool | None
    in_count: int | None
    out_count: int | None
    intrusion_count: int | None
    loitering_count: int | None


class EventResponse(BaseModel):
    event_id: int
    run_id: str
    frame_id: int
    event_type: str
    track_id: int
    video_timestamp: float
    position_x: int
    position_y: int
    direction: str | None
    zone_id: str | None
    duration_seconds: float | None
    logged_at: str


class RunListResponse(BaseModel):
    items: list[RunResponse]
    limit: int
    offset: int


class EventListResponse(BaseModel):
    items: list[EventResponse]
    limit: int
    offset: int


class AnalyzeRequest(BaseModel):
    source_path: str = Field(min_length=1)
    output_path: str = Field(min_length=1)
    config_path: str = "configs/default.yaml"


class AnalyzeAcceptedResponse(BaseModel):
    run_id: str
    status: Literal["running"]
