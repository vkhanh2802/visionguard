from src.tracking import Track
from src.utils.geometry import classify_side, has_crossed_line, signed_distance_to_line

from .types import Event

class LineCrossingEngine:
    def __init__(
        self,
        line_start: tuple[int, int],
        line_end: tuple[int, int],
        dead_zone_px: float = 5.0,
        max_missing_frames: int = 30,
        confirmation_frames: int = 2,
        negative_to_positive: str = "IN",
        positive_to_negative: str = "OUT",
    ):
        if line_start == line_end:
            raise ValueError("line_start and line_end must be different points.")

        if dead_zone_px < 0:
            raise ValueError("dead_zone_px must be non-negative.")
        if confirmation_frames < 1:
            raise ValueError("confirmation_frames must be at least 1.")
        
        if max_missing_frames < 0:
            raise ValueError("max_missing_frames must be non-negative.")

        if {negative_to_positive, positive_to_negative} != {"IN", "OUT"}:
            raise ValueError("Directions must contain exactly IN and OUT.")

        self.line_start = line_start
        self.line_end = line_end
        self.dead_zone_px = dead_zone_px
        self.max_missing_frames = max_missing_frames
        self.confirmation_frames = confirmation_frames
        self.negative_to_positive = negative_to_positive
        self.positive_to_negative = positive_to_negative

        self.last_stable_side: dict[int, int] = {}
        self.last_stable_point: dict[int, tuple[int, int]] = {}
        self.last_seen_frame: dict[int, int] = {}

        self.pending_side: dict[int, int] = {}
        self.pending_count: dict[int, int] = {}
        self.pending_point: dict[int, tuple[int, int]] = {}

        self.in_count = 0
        self.out_count = 0

    def _get_direction(self, previous_side: int, current_side: int) -> str:
        if previous_side == -1 and current_side == 1:
            return self.negative_to_positive

        if previous_side == 1 and current_side == -1:
            return self.positive_to_negative

        raise ValueError(f"Invalid side transition: {previous_side} -> {current_side}")

    def _update_count(self, direction: str) -> None:
        if direction == "IN":
            self.in_count += 1
        elif direction == "OUT":
            self.out_count += 1

    @property
    def total_count(self) -> int:
        return self.in_count + self.out_count

    @property
    def net_count(self) -> int:
        return self.in_count - self.out_count

    def _clear_pending(self, track_id: int) -> None:
        self.pending_side.pop(track_id, None)
        self.pending_count.pop(track_id, None)
        self.pending_point.pop(track_id, None)

    def _cleanup_stale_tracks(self, frame_id: int) -> None:
        stale_ids = [
            track_id
            for track_id, last_frame in self.last_seen_frame.items()
            if frame_id - last_frame > self.max_missing_frames
        ]

        for track_id in stale_ids:
            self.last_seen_frame.pop(track_id, None)
            self.last_stable_side.pop(track_id, None)
            self.last_stable_point.pop(track_id, None)
            self.pending_side.pop(track_id, None)
            self.pending_count.pop(track_id, None)
            self.pending_point.pop(track_id, None)

    def process(self, tracks: list[Track], frame_id: int, timestamp: float) -> list[Event]:
        events = []

        for track in tracks:
            track_id = track.track_id
            point = track.bottom_center

            self.last_seen_frame[track_id] = frame_id

            distance = signed_distance_to_line(point, self.line_start, self.line_end)
            current_side = classify_side(distance, self.dead_zone_px)

            if current_side == 0:
                continue

            if track_id not in self.last_stable_side:
                self.last_stable_side[track_id] = current_side
                self.last_stable_point[track_id] = point
                continue

            stable_side = self.last_stable_side[track_id]

            if current_side == stable_side:
                self.last_stable_point[track_id] = point
                self._clear_pending(track_id)
                continue

            if self.pending_side.get(track_id) != current_side:
                self.pending_side[track_id] = current_side
                self.pending_count[track_id] = 1
                self.pending_point[track_id] = point
                continue

            self.pending_count[track_id] += 1
            self.pending_point[track_id] = point

            if self.pending_count[track_id] < self.confirmation_frames:
                continue

            previous_point = self.last_stable_point[track_id]

            if has_crossed_line(previous_point, point, self.line_start, self.line_end, self.dead_zone_px):
                direction = self._get_direction(stable_side, current_side)
                events.append(
                    Event(
                        event_type="line_crossing",
                        track_id=track_id,
                        direction=direction,
                        timestamp=timestamp,
                        position=point,
                    )
                )
                self._update_count(direction)

            self.last_stable_side[track_id] = current_side
            self.last_stable_point[track_id] = point
            self._clear_pending(track_id)

        self._cleanup_stale_tracks(frame_id)

        return events