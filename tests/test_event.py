from src.events import IntrusionEngine, LineCrossingEngine, LoiteringEngine
from src.tracking import Track

LINE_START = (0, 100)
LINE_END = (200, 100)

ROI = (
    (50, 110),
    (150, 110),
    (150, 220),
    (50, 220),
)

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

def test_event_engines_process_same_tracks():
    line_engine = LineCrossingEngine(
        line_start=LINE_START,
        line_end=LINE_END,
        dead_zone_px=0,
    )
    intrusion_engine = IntrusionEngine(
        polygon=ROI,
        max_missing_frames=30,
    )
    loitering_engine = LoiteringEngine(
        polygon=ROI,
        dwell_threshold_seconds=10.0,
        max_missing_frames=30,
    )

    frames = [
        (0, 0.0, (100, 50)),    # Outside ROI, negative side of line
        (1, 1.0, (100, 120)),   # Crosses line and enters ROI
        (2, 11.0, (100, 120)),  # Has remained in ROI for 10 seconds
    ]

    all_events = []

    for frame_id, timestamp, point in frames:
        tracks = [make_track(track_id=1, point=point)]

        events = [
            *line_engine.process(tracks, frame_id, timestamp),
            *intrusion_engine.process(tracks, frame_id, timestamp),
            *loitering_engine.process(tracks, frame_id, timestamp),
        ]

        all_events.extend(events)

    assert [event.event_type for event in all_events] == [
        "line_crossing",
        "intrusion",
        "loitering",
    ]

    assert all_events[0].direction == "IN"
    assert all_events[1].track_id == 1
    assert all_events[1].zone_id == "restricted-zone-1"
    assert all_events[2].duration_seconds == 10.0

    assert line_engine.in_count == 1
    assert intrusion_engine.intrusion_count == 1
    assert loitering_engine.loitering_count == 1