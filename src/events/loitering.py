from dataclasses import dataclass
from math import dist

from src.tracking.types import Track
from src.utils.geometry import Point, Polygon, point_in_polygon

from .types import Event

@dataclass
class LoiteringVisit:
    entry_timestamp: float
    event_triggered: bool
    last_seen_frame: int
    last_position: Point

class LoiteringEngine:
    def __init__(
        self,
        polygon: Polygon,
        dwell_threshold_seconds: float,
        zone_id: str = "restricted-zone-1",
        max_missing_frames: int = 30,
        id_reassociation_frames: int = 0,
        id_reassociation_distance_px: float = 0.0,
    ):
        if len(polygon) < 3:
            raise ValueError("A polygon must contain at least three points.")

        if dwell_threshold_seconds <= 0:
            raise ValueError("dwell_threshold_seconds must be positive.")

        if max_missing_frames < 0:
            raise ValueError("max_missing_frames must be non-negative.")

        if id_reassociation_frames < 0:
            raise ValueError("id_reassociation_frames must be non-negative.")

        if id_reassociation_distance_px < 0:
            raise ValueError("id_reassociation_distance_px must be non-negative.")

        if bool(id_reassociation_frames) != bool(id_reassociation_distance_px):
            raise ValueError(
                "id_reassociation_frames and id_reassociation_distance_px "
                "must be configured together."
            )

        self.polygon = polygon
        self.dwell_threshold_seconds = dwell_threshold_seconds
        self.zone_id = zone_id
        self.max_missing_frames = max_missing_frames
        self.id_reassociation_frames = id_reassociation_frames
        self.id_reassociation_distance_px = id_reassociation_distance_px
        self.visit_retention_frames = max(
            max_missing_frames,
            id_reassociation_frames,
        )
        self.visits: dict[int, LoiteringVisit] = {}
        self.loitering_count = 0

    def _cleanup_stale_visits(self, frame_id: int) -> None:
        stale_ids = [
            track_id
            for track_id, visit in self.visits.items()
            if frame_id - visit.last_seen_frame - 1 > self.visit_retention_frames
        ]

        for track_id in stale_ids:
            self.visits.pop(track_id, None)

    def _reassociate_visit(
        self,
        track_id: int,
        position: Point,
        frame_id: int,
        observed_track_ids: set[int],
    ) -> LoiteringVisit | None:
        if self.id_reassociation_frames == 0:
            return None

        candidates = [
            (candidate_id, visit)
            for candidate_id, visit in self.visits.items()
            if candidate_id != track_id
            and candidate_id not in observed_track_ids
            and frame_id - visit.last_seen_frame <= self.id_reassociation_frames
            and dist(position, visit.last_position) <= self.id_reassociation_distance_px
        ]

        if not candidates:
            return None

        candidate_id, visit = min(
            candidates,
            key=lambda candidate: dist(position, candidate[1].last_position),
        )
        self.visits.pop(candidate_id)
        return visit

    def get_duration(self, track_id: int, timestamp: float) -> float | None:
        visit = self.visits.get(track_id)

        if visit is None:
            return None

        return timestamp - visit.entry_timestamp

    def process(self, tracks: list[Track], frame_id: int, timestamp: float,) -> list[Event]:
        self._cleanup_stale_visits(frame_id)
        events = []
        observed_track_ids = {track.track_id for track in tracks}

        for track in tracks:
            track_id = track.track_id
            position = track.bottom_center
            inside = point_in_polygon(position, self.polygon)
            visit = self.visits.get(track_id)

            if not inside:
                self.visits.pop(track_id, None)
                continue

            if visit is None:
                visit = self._reassociate_visit(
                    track_id=track_id,
                    position=position,
                    frame_id=frame_id,
                    observed_track_ids=observed_track_ids,
                )

            if visit is None:
                self.visits[track_id] = LoiteringVisit(
                    entry_timestamp=timestamp,
                    event_triggered=False,
                    last_seen_frame=frame_id,
                    last_position=position,
                )
                continue

            self.visits[track_id] = visit
            visit.last_seen_frame = frame_id
            visit.last_position = position
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
