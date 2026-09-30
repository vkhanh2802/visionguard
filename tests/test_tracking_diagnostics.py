from src.tracking import Track
from src.tracking.diagnostics import TrackingDiagnostics


ROI = ((0, 0), (100, 0), (100, 100), (0, 100))


def make_track(track_id: int, point: tuple[int, int]) -> Track:
    x, y = point
    return Track(
        track_id=track_id,
        bbox=(x - 10, y - 20, x + 10, y),
        confidence=0.9,
        class_id=0,
        class_name="person",
        centroid=(x, y - 10),
        bottom_center=point,
    )


def test_collects_track_lifetimes_gaps_and_roi_births():
    diagnostics = TrackingDiagnostics(zones=(ROI,))

    diagnostics.observe([make_track(1, (150, 50)), make_track(2, (50, 50))], 1)
    diagnostics.observe([make_track(1, (150, 50))], 2)
    diagnostics.observe([], 3)
    diagnostics.observe([make_track(1, (150, 50))], 4)

    snapshot = diagnostics.snapshot()

    assert snapshot["total_track_count"] == 2
    assert snapshot["new_track_count_in_roi"] == 1
    assert snapshot["tracks_with_gaps"] == 1
    assert snapshot["total_missing_frames"] == 1
    assert snapshot["max_gap_frames"] == 1
    assert snapshot["median_observed_frames"] == 2.0
    assert snapshot["track_lifetimes"] == [
        {
            "track_id": 1,
            "first_seen_frame": 1,
            "last_seen_frame": 4,
            "observed_frames": 3,
            "lifetime_frames": 4,
            "total_missing_frames": 1,
            "longest_gap_frames": 1,
            "first_seen_in_roi": False,
        },
        {
            "track_id": 2,
            "first_seen_frame": 1,
            "last_seen_frame": 1,
            "observed_frames": 1,
            "lifetime_frames": 1,
            "total_missing_frames": 0,
            "longest_gap_frames": 0,
            "first_seen_in_roi": True,
        },
    ]
