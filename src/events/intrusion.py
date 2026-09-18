from src.tracking.types import Track
from src.utils.geometry import Polygon, point_in_polygon

from .types import Event


class IntrusionEngine:
    def __init__(
        self,
        polygon: Polygon,
        zone_id: str = "restricted-zone-1",
        max_missing_frames: int = 30,
        entry_confirmation_frames: int = 1,
        exit_confirmation_frames: int = 1,
    ):
        if len(polygon) < 3:
            raise ValueError("A polygon must contain at least three points.")

        if max_missing_frames < 0:
            raise ValueError("max_missing_frames must be non-negative.")

        if entry_confirmation_frames < 1:
            raise ValueError("entry_confirmation_frames must be at least 1.")

        if exit_confirmation_frames < 1:
            raise ValueError("exit_confirmation_frames must be at least 1.")

        self.polygon = polygon
        self.zone_id = zone_id
        self.max_missing_frames = max_missing_frames
        self.entry_confirmation_frames = entry_confirmation_frames
        self.exit_confirmation_frames = exit_confirmation_frames

        self.previous_inside: dict[int, bool] = {}
        self.inside_confirmation_count: dict[int, int] = {}
        self.outside_confirmation_count: dict[int, int] = {}
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
            self.inside_confirmation_count.pop(track_id, None)
            self.outside_confirmation_count.pop(track_id, None)
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
                self.inside_confirmation_count[track_id] = 0
                self.outside_confirmation_count[track_id] = 0
                self.last_seen_frame[track_id] = frame_id
                continue

            previous_frame = self.last_seen_frame[track_id]
            if frame_id - previous_frame > 1:
                self.inside_confirmation_count[track_id] = 0
                self.outside_confirmation_count[track_id] = 0

            if previous_inside:
                if current_inside:
                    self.outside_confirmation_count[track_id] = 0
                else:
                    confirmation_count = (
                        self.outside_confirmation_count.get(track_id, 0) + 1
                    )
                    self.outside_confirmation_count[track_id] = confirmation_count

                    if confirmation_count >= self.exit_confirmation_frames:
                        self.previous_inside[track_id] = False
                        self.outside_confirmation_count[track_id] = 0
            elif current_inside:
                confirmation_count = self.inside_confirmation_count.get(track_id, 0) + 1
                self.inside_confirmation_count[track_id] = confirmation_count

                if confirmation_count >= self.entry_confirmation_frames:
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
                    self.previous_inside[track_id] = True
                    self.inside_confirmation_count[track_id] = 0
            else:
                self.inside_confirmation_count[track_id] = 0

            self.last_seen_frame[track_id] = frame_id

        return events
