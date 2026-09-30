from dataclasses import dataclass
from statistics import median

from src.utils.geometry import Point, Polygon, point_in_polygon

from .types import Track


@dataclass
class _TrackLifetime:
    first_seen_frame: int
    last_seen_frame: int
    observed_frames: int
    total_missing_frames: int
    longest_gap_frames: int
    first_seen_in_roi: bool


class TrackingDiagnostics:
    def __init__(self, zones: tuple[Polygon, ...]):
        self.zones = zones
        self._tracks: dict[int, _TrackLifetime] = {}

    def observe(self, tracks: list[Track], frame_id: int) -> None:
        for track in tracks:
            track_id = track.track_id
            lifetime = self._tracks.get(track_id)

            if lifetime is None:
                self._tracks[track_id] = _TrackLifetime(
                    first_seen_frame=frame_id,
                    last_seen_frame=frame_id,
                    observed_frames=1,
                    total_missing_frames=0,
                    longest_gap_frames=0,
                    first_seen_in_roi=self._is_in_roi(track.bottom_center),
                )
                continue

            missing_frames = max(0, frame_id - lifetime.last_seen_frame - 1)
            lifetime.last_seen_frame = frame_id
            lifetime.observed_frames += 1
            lifetime.total_missing_frames += missing_frames
            lifetime.longest_gap_frames = max(
                lifetime.longest_gap_frames,
                missing_frames,
            )

    def snapshot(self) -> dict[str, object]:
        track_lifetimes = [
            {
                "track_id": track_id,
                "first_seen_frame": lifetime.first_seen_frame,
                "last_seen_frame": lifetime.last_seen_frame,
                "observed_frames": lifetime.observed_frames,
                "lifetime_frames": lifetime.last_seen_frame - lifetime.first_seen_frame + 1,
                "total_missing_frames": lifetime.total_missing_frames,
                "longest_gap_frames": lifetime.longest_gap_frames,
                "first_seen_in_roi": lifetime.first_seen_in_roi,
            }
            for track_id, lifetime in sorted(self._tracks.items())
        ]
        observed_frames = [item["observed_frames"] for item in track_lifetimes]
        gaps = [item["longest_gap_frames"] for item in track_lifetimes]

        return {
            "total_track_count": len(track_lifetimes),
            "new_track_count_in_roi": sum(
                item["first_seen_in_roi"] for item in track_lifetimes
            ),
            "tracks_with_gaps": sum(gap > 0 for gap in gaps),
            "total_missing_frames": sum(
                item["total_missing_frames"] for item in track_lifetimes
            ),
            "max_gap_frames": max(gaps, default=0),
            "median_observed_frames": median(observed_frames) if observed_frames else None,
            "track_lifetimes": track_lifetimes,
        }

    def _is_in_roi(self, position: Point) -> bool:
        return any(point_in_polygon(position, zone) for zone in self.zones)
