from dataclasses import dataclass
from src.tracking.types import Track
from src.utils.geometry import Polygon, point_in_polygon

from .types import Event

@dataclass
class LoiteringVisit:
    entry_timestamp: float
    event_triggered: bool
    last_seen_frame: int

class LoiteringEngine:
    def __init__(
        self,
        polygon: Polygon,
        dwell_threshold_seconds: float,
        zone_id: str = "restricted-zone-1",
        max_missing_frames: int = 30,
    ):
        if len(polygon) < 3:
            raise ValueError("A polygon must contain at least three points.")

        if dwell_threshold_seconds <= 0:
            raise ValueError("dwell_threshold_seconds must be positive.")

        if max_missing_frames < 0:
            raise ValueError("max_missing_frames must be non-negative.")

        self.polygon = polygon
        self.dwell_threshold_seconds = dwell_threshold_seconds
        self.zone_id = zone_id
        self.max_missing_frames = max_missing_frames
        self.visits: dict[int, LoiteringVisit] = {}
        self.loitering_count = 0

    def _cleanup_stale_visits(self, frame_id: int) -> None:
        stale_ids = [
            track_id
            for track_id, visit in self.visits.items()
            if frame_id - visit.last_seen_frame > self.max_missing_frames
        ]

        for track_id in stale_ids:
            self.visits.pop(track_id, None)

    def get_duration(self, track_id: int, timestamp: float) -> float | None:
        visit = self.visits.get(track_id)

        if visit is None:
            return None

        return timestamp - visit.entry_timestamp

    def process(self, tracks: list[Track], frame_id: int, timestamp: float,) -> list[Event]:
        self._cleanup_stale_visits(frame_id)
        events = []

        for track in tracks:
            track_id = track.track_id
            position = track.bottom_center
            inside = point_in_polygon(position, self.polygon)
            visit = self.visits.get(track_id)

            if not inside:
                self.visits.pop(track_id, None)
                continue

            if visit is None:
                self.visits[track_id] = LoiteringVisit(
                    entry_timestamp=timestamp,
                    event_triggered=False,
                    last_seen_frame=frame_id,
                )
                continue

            visit.last_seen_frame = frame_id
            duration = timestamp - visit.entry_timestamp

            if duration >= self.dwell_threshold_seconds and not visit.event_triggered:
                events.append(
                    Event(
                        event_type="loitering",
                        track_id=track_id,
                        timestamp=timestamp,
                        position=position,
                        zone_id=self.zone_id,
                        duration_seconds=duration,
                    )
                )
                visit.event_triggered = True
                self.loitering_count += 1

        return events