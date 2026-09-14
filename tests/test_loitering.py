from src.events.loitering import LoiteringEngine
from src.tracking import Track

ROI = (
    (0, 0),
    (100, 0),
    (100, 100),
    (0, 100),
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

def test_inside_before_threshold_does_not_trigger():
    engine = LoiteringEngine(ROI, dwell_threshold_seconds=5.0)

    engine.process([make_track(1, (50, 50))], frame_id=1, timestamp=0.0)
    events = engine.process([make_track(1, (50, 50))], frame_id=2, timestamp=4.9)

    assert events == []
    assert engine.loitering_count == 0

def test_reaching_threshold_triggers_loitering_once():
    engine = LoiteringEngine(ROI, dwell_threshold_seconds=5.0)

    engine.process([make_track(1, (50, 50))], frame_id=1, timestamp=0.0)
    events = engine.process([make_track(1, (50, 50))], frame_id=2, timestamp=5.0)

    assert len(events) == 1
    assert events[0].event_type == "loitering"
    assert events[0].duration_seconds == 5.0
    assert engine.loitering_count == 1

def test_remaining_inside_after_trigger_does_not_duplicate():
    engine = LoiteringEngine(ROI, dwell_threshold_seconds=5.0)

    engine.process([make_track(1, (50, 50))], 1, 0.0)
    engine.process([make_track(1, (50, 50))], 2, 5.0)

    assert engine.process([make_track(1, (50, 50))], 3, 6.0) == []
    assert engine.loitering_count == 1

def test_exit_resets_loitering_timer():
    engine = LoiteringEngine(ROI, dwell_threshold_seconds=5.0)

    engine.process([make_track(1, (50, 50))], 1, 0.0)
    engine.process([make_track(1, (50, 50))], 2, 4.0)

    engine.process([make_track(1, (150, 50))], 3, 4.1)
    engine.process([make_track(1, (50, 50))], 4, 5.0)

    events = engine.process([make_track(1, (50, 50))], 5, 9.9)

    assert events == []

def test_reentering_starts_a_new_timer():
    engine = LoiteringEngine(ROI, dwell_threshold_seconds=5.0)

    engine.process([make_track(1, (50, 50))], 1, 0.0)
    engine.process([make_track(1, (150, 50))], 2, 1.0)

    engine.process([make_track(1, (50, 50))], 3, 2.0)
    events = engine.process([make_track(1, (50, 50))], 4, 7.0)

    assert len(events) == 1
    assert events[0].duration_seconds == 5.0

def test_short_tracking_gap_preserves_timer():
    engine = LoiteringEngine(
        ROI,
        dwell_threshold_seconds=5.0,
        max_missing_frames=2,
    )

    engine.process([make_track(1, (50, 50))], 1, 0.0)
    engine.process([], 2, 1.0)

    events = engine.process([make_track(1, (50, 50))], 3, 5.0)

    assert len(events) == 1

def test_long_tracking_gap_resets_timer():
    engine = LoiteringEngine(
        ROI,
        dwell_threshold_seconds=5.0,
        max_missing_frames=2,
    )

    engine.process([make_track(1, (50, 50))], 1, 0.0)
    engine.process([], 2, 1.0)
    engine.process([], 3, 2.0)

    events = engine.process([make_track(1, (50, 50))], 4, 10.0)

    assert events == []