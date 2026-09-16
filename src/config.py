from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.utils.geometry import Point, segments_intersect

Polygon = tuple[Point, ...]

class DetectionConfig(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    model_path: str = "yolo26n.pt"
    confidence: float = Field(default=0.4, ge =0.0, le=1.0)
    target_classes: set[str] = Field(default_factory = lambda: {"person"})

class TrackingConfig(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    history_length: int = Field(default=30, ge=1)
    max_missing_frames: int = Field(default=30, ge = 0)


class LineCrossingConfig(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    enabled: bool = True
    start: Point
    end: Point
    dead_zone_px: float = Field(default= 5.0, ge = 0.0)
    confirmation_frames: int = Field(default=3, ge = 1)
    negative_to_positive: Literal["IN", "OUT"]= "IN"
    positive_to_negative: Literal["IN", "OUT"]= "OUT"

    @model_validator(mode = "after")
    def validate_line(self):
        if self.start == self.end:
            raise ValueError("Line start and end points cannot be the same.")

        if self.negative_to_positive == self.positive_to_negative:
            raise ValueError("negative_to_positive and positive_to_negative cannot be the same.")

        return self

class ZoneConfig(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    polygon: list[Point]

    @model_validator(mode="after")
    def validate_polygon(self):
        if len(self.polygon) < 3:
            raise ValueError("A polygon must contain at least three points.")

        if len(self.polygon) != len(set(self.polygon)):
            raise ValueError("A polygon cannot have duplicate points.")

        edge_count = len(self.polygon)

        for index in range(edge_count):
            start_a = self.polygon[index]
            end_a = self.polygon[(index + 1) % edge_count]

            for other_index in range(index+ 1, edge_count):
                if (
                    other_index == index + 1
                    or (index == 0 and other_index == edge_count -1)
                ):
                    continue

                start_b = self.polygon[other_index]
                end_b = self.polygon[(other_index + 1) % edge_count]

                if segments_intersect(start_a, end_a, start_b, end_b):
                    raise ValueError("A polygon cannot have intersecting edges.")

        return self
        
class IntrusionConfig(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    enabled: bool = True
    zone_id: str

class LoiteringConfig(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    enabled: bool = True
    zone_id: str
    dwell_threshold_seconds: float = Field(gt=0.0)

class EventsConfig(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    line_crossing: LineCrossingConfig
    zones: dict[str, ZoneConfig]
    intrusion: IntrusionConfig
    loitering: LoiteringConfig

    @model_validator(mode = "after")
    def validate_zone_references(self):
        enabled_zone_ids = []

        if self.intrusion.enabled:
            enabled_zone_ids.append(self.intrusion.zone_id)

        if self.loitering.enabled:
            enabled_zone_ids.append(self.loitering.zone_id)

        for zone_id in enabled_zone_ids:
            if zone_id not in self.zones:
                raise ValueError(f"Unknown zone_id: {zone_id}")

        return self

class OutputConfig(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    display: bool = True
    codec: str = "mp4v"

class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra= "forbid")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    event_jsonl_path: Path | None = None
    run_metadata_path: Path | None = None

    @model_validator(mode="after")
    def validate_artifact_paths(self):
        if (self.event_jsonl_path is None) != (self.run_metadata_path is None):
            raise ValueError(
                "event_jsonl_path and run_metadata_path must be configured together."
            )

        return self

class VideoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fps_override: float | None = Field(default=None, gt=0.0)

class DatabaseConfig(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    path: Path = Path("data/visionguard.db")

class AppConfig(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    detection: DetectionConfig
    tracking: TrackingConfig
    events: EventsConfig
    video: VideoConfig = Field(default_factory=VideoConfig)
    output: OutputConfig
    logging: LoggingConfig
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)

def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open(encoding = "utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        raise ValueError(f"Config file is empty: {config_path}")

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a dictionary at the top level: {config_path}")

    return AppConfig.model_validate(data)


