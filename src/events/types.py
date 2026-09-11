from dataclasses import dataclass

@dataclass
class Event:
    event_type: str
    track_id: int
    direction: str
    timestamp: float
    position: tuple[int, int]  # (x, y)
    