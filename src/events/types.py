from dataclasses import dataclass

@dataclass
class Event:
    event_type: str
    track_id: int
    timestamp: float
    position: tuple[int, int]  # (x, y)
    direction: str | None = None
    zone_id:str | None = None
    duration_seconds: float | None = None

