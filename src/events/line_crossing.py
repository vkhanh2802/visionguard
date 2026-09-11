from src.tracking import Track
from src.utils.geometry import classify_side, has_crossed_line, side_of_line

from .types import Event

class LineCrossingEngine:
    def __init__(
        self,
        line_start: tuple[int, int],
        line_end: tuple[int, int],
        epsilon: float = 1.0,
        max_missing_frames: int = 30,
        negative_to_positive: str = "IN",
        positive_to_negative: str = "OUT",
    ):
        if line_start == line_end:
            raise ValueError("line_start and line_end must be different points.")

        self.line_start = line_start
        self.line_end = line_end
        self.epsilon = epsilon
        self.max_missing_frames = max_missing_frames
        self.negative_to_positive = negative_to_positive
        self.positive_to_negative = positive_to_negative

        self.last_stable_side: dict[int, int] = {}
        self.last_stable_point: dict[int, tuple[int, int]] = {}
        self.last_seen_frame: dict[int, int] = {}

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

    def process(self, tracks: list[Track], frame_id: int, timestamp: float) -> list[Event]:
        events = []

        for track in tracks:
            track_id = track.track_id
            point = track.bottom_center

            self.last_seen_frame[track_id] = frame_id

            side_value = side_of_line(point, self.line_start, self.line_end)
            current_side = classify_side(side_value, self.epsilon)

            if current_side == 0:
                continue

            if track_id not in self.last_stable_side:
                self.last_stable_side[track_id] = current_side
                self.last_stable_point[track_id] = point
                continue

            previous_side = self.last_stable_side[track_id]
            previous_point = self.last_stable_point[track_id]

            if current_side == previous_side:
                self.last_stable_point[track_id] = point
                continue

            crossed = has_crossed_line(
                previous_point,
                point,
                self.line_start,
                self.line_end,
                self.epsilon,
            )

            if crossed:
                direction = self._get_direction(previous_side, current_side)
                event = Event(
                    event_type="line_crossing",
                    track_id=track_id,
                    direction=direction,
                    timestamp=timestamp,
                    position=point,
                )

                events.append(event)
                self._update_count(direction)

            self.last_stable_side[track_id] = current_side
            self.last_stable_point[track_id] = point

        self._cleanup_stale_tracks(frame_id)

        return events