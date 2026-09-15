from src.tracking.types import Track
from src.utils.geometry import Polygon, point_in_polygon

from .types import Event


class IntrusionEngine:
    def __init__(
        self,
        polygon: Polygon,
        zone_id: str = "restricted-zone-1",
        max_missing_frames: int = 30,
    ):
        if len(polygon) < 3:
            raise ValueError("A polygon must contain at least three points.")

        if max_missing_frames < 0:
            raise ValueError("max_missing_frames must be non-negative.")

        self.polygon = polygon
        self.zone_id = zone_id
        self.max_missing_frames = max_missing_frames

        self.previous_inside: dict[int, bool] = {}
        self.last_seen_frame: dict[int, int] = {}
        self.intrusion_count = 0

    def _cleanup_stale_tracks(self, frame_id: int) -> None:
        stale_ids = [
            track_id
            for track_id, last_seen_frame in self.last_seen_frame.items()
            if frame_id - last_seen_frame - 1 > self.max_missing_frames
        ]

        for track_id in stale_ids:
            self.previous_inside.pop(track_id, None)
            self.last_seen_frame.pop(track_id, None)


    def process(self, tracks: list[Track], frame_id: int, timestamp: float,) -> list[Event]:
        self._cleanup_stale_tracks(frame_id)
        events = []

        for track in tracks:
            track_id = track.track_id
            position = track.bottom_center
            current_inside = point_in_polygon(position, self.polygon)

            previous_inside = self.previous_inside.get(track_id)

            # Observation đầu: khởi tạo state, không alert.
            if previous_inside is None:
                self.previous_inside[track_id] = current_inside
                self.last_seen_frame[track_id] = frame_id
                continue

            if not previous_inside and current_inside:
                events.append(
                    Event(
                        event_type="intrusion",
                        track_id=track_id,
                        timestamp=timestamp,
                        position=position,
                        zone_id=self.zone_id,
                    )
                )
                self.intrusion_count += 1

            self.previous_inside[track_id] = current_inside
            self.last_seen_frame[track_id] = frame_id

        return events