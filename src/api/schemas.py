from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"]
    database: Literal["connected"]


class ApiIndexResponse(BaseModel):
    service: Literal["VisionGuard API"]
    docs_url: str
    health_url: str
    runs_url: str
    events_url: str


class RunResponse(BaseModel):
    run_id: str
    source_path: str
    output_path: str
    status: Literal["queued", "running", "completed", "failed"]
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
    config_path: str = "C:/VisionGuard/configs/default.yaml"


class AnalyzeAcceptedResponse(BaseModel):
    run_id: str
    status: Literal["queued"]


class EventAnalyticsResponse(BaseModel):
    recorded_event_count: int
    unique_track_count: int
    first_event_timestamp: float | None
    last_event_timestamp: float | None
    by_type: dict[str, int]


class RunAnalyticsResponse(BaseModel):
    run_id: str
    status: Literal["queued", "running", "completed", "failed"]
    processed_frames: int | None
    in_count: int | None
    out_count: int | None
    net_count: int | None
    intrusion_count: int | None
    loitering_count: int | None
    events: EventAnalyticsResponse
    tracking: "TrackingDiagnosticsResponse | None"


class TrackLifetimeResponse(BaseModel):
    track_id: int
    first_seen_frame: int
    last_seen_frame: int
    observed_frames: int
    lifetime_frames: int
    total_missing_frames: int
    longest_gap_frames: int
    first_seen_in_roi: bool


class TrackingDiagnosticsResponse(BaseModel):
    total_track_count: int
    new_track_count_in_roi: int
    tracks_with_gaps: int
    total_missing_frames: int
    max_gap_frames: int
    median_observed_frames: float | None
    track_lifetimes: list[TrackLifetimeResponse]
    continuity: "ContinuityDiagnosticsResponse | None" = None
    deduplication: "DeduplicationDiagnosticsResponse | None" = None


class ContinuityDiagnosticsResponse(BaseModel):
    replacement_match_count: int
    pending_match_count: int
    rejected_ambiguous_match_count: int
    rejected_gap_match_count: int
    replacement_matches: list["ReplacementMatchResponse"] = Field(default_factory=list)


class ReplacementMatchResponse(BaseModel):
    canonical_track_id: int
    replacement_track_id: int
    frame_id: int
    gap_frames: int
    distance_px: float


class DeduplicationDiagnosticsResponse(BaseModel):
    confirmed_pair_count: int
    pending_duplicate_observation_count: int
    suppressed_observation_count: int
    ambiguous_pair_count: int
    handoff_count: int
    confirmed_pairs: list["ConfirmedDuplicatePairResponse"] = Field(
        default_factory=list
    )


class ConfirmedDuplicatePairResponse(BaseModel):
    primary_track_id: int
    duplicate_track_id: int
    confirmed_frame_id: int
    containment_ratio: float
    iou: float
    bottom_distance_px: float
