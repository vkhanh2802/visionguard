from collections import defaultdict, deque

from .types import Track


class TrackHistory:
    def __init__(self, max_length: int = 30, max_missing_frames: int = 30):
        self.history = defaultdict(lambda: deque(maxlen=max_length))
        self.last_seen = {}
        self.frame_id = 0
        self.max_missing_frames = max_missing_frames

    def update(self, tracks: list[Track]) -> None:
        self.frame_id += 1

        for track in tracks:
            self.history[track.track_id].append(track.centroid)
            self.last_seen[track.track_id] = self.frame_id

        stale_ids = [
            track_id
            for track_id, last_frame in self.last_seen.items()
            if self.frame_id - last_frame > self.max_missing_frames
        ]

        for track_id in stale_ids:
            self.history.pop(track_id, None)
            self.last_seen.pop(track_id, None)

    def get(self, track_id: int) -> list[tuple[int, int]]:
        return list(self.history.get(track_id, []))