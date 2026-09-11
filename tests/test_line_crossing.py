from src.events import LineCrossingEngine
from src.tracking import Track

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

def test_first_observation_does_not_create_event():
    engine = LineCrossingEngine((0, 0), (10, 0))

    events = engine.process([make_track(1, (5, -5))], frame_id=1, timestamp=0.0)

    assert events == []
    assert engine.in_count == 0
    assert engine.out_count == 0

def test_negative_to_positive_creates_in_event():
    engine = LineCrossingEngine((0, 0), (10, 0))

    engine.process([make_track(1, (5, -5))], frame_id=1, timestamp=0.0)
    events = engine.process([make_track(1, (5, 5))], frame_id=2, timestamp=0.1)

    assert len(events) == 1
    assert events[0].track_id == 1
    assert events[0].direction == "IN"
    assert engine.in_count == 1
    assert engine.out_count == 0

def test_positive_to_negative_creates_out_event():
    engine = LineCrossingEngine((0, 0), (10, 0))

    engine.process([make_track(1, (5, 5))], frame_id=1, timestamp=0.0)
    events = engine.process([make_track(1, (5, -5))], frame_id=2, timestamp=0.1)

    assert len(events) == 1
    assert events[0].direction == "OUT"
    assert engine.in_count == 0
    assert engine.out_count == 1

def test_remaining_on_same_side_does_not_duplicate_event():
    engine = LineCrossingEngine((0, 0), (10, 0))

    engine.process([make_track(1, (5, -5))], 1, 0.0)
    events = engine.process([make_track(1, (5, 5))], 2, 0.1)

    assert len(events) == 1

    assert engine.process([make_track(1, (5, 6))], 3, 0.2) == []
    assert engine.process([make_track(1, (5, 7))], 4, 0.3) == []
    assert engine.process([make_track(1, (5, 8))], 5, 0.4) == []

    assert engine.in_count == 1

def test_track_can_cross_in_then_out():
    engine = LineCrossingEngine((0, 0), (10, 0))

    engine.process([make_track(1, (5, -5))], 1, 0.0)

    in_events = engine.process([make_track(1, (5, 5))], 2, 0.1)
    out_events = engine.process([make_track(1, (5, -5))], 3, 0.2)

    assert in_events[0].direction == "IN"
    assert out_events[0].direction == "OUT"
    assert engine.in_count == 1
    assert engine.out_count == 1
    assert engine.total_count == 2

def test_dead_zone_preserves_last_stable_side():
    engine = LineCrossingEngine((0, 0), (10, 0), epsilon=1.0)

    engine.process([make_track(1, (5, -5))], 1, 0.0)

    events_on_line = engine.process([make_track(1, (5, 0))], 2, 0.1)
    events_after_crossing = engine.process([make_track(1, (5, 5))], 3, 0.2)

    assert events_on_line == []
    assert len(events_after_crossing) == 1
    assert events_after_crossing[0].direction == "IN"

def test_crossing_outside_segment_does_not_create_event():
    engine = LineCrossingEngine((0, 0), (10, 0))

    engine.process([make_track(1, (20, -5))], 1, 0.0)
    events = engine.process([make_track(1, (20, 5))], 2, 0.1)

    assert events == []
    assert engine.total_count == 0

def test_multiple_tracks_are_counted_independently():
    engine = LineCrossingEngine((0, 0), (10, 0))

    engine.process(
        [
            make_track(1, (3, -5)),
            make_track(2, (7, 5)),
        ],
        frame_id=1,
        timestamp=0.0,
    )

    events = engine.process(
        [
            make_track(1, (3, 5)),
            make_track(2, (7, -5)),
        ],
        frame_id=2,
        timestamp=0.1,
    )

    assert len(events) == 2
    assert engine.in_count == 1
    assert engine.out_count == 1

def test_stale_track_state_is_removed():
    engine = LineCrossingEngine((0, 0), (10, 0), max_missing_frames=2)

    engine.process([make_track(1, (5, -5))], 1, 0.0)

    engine.process([], 2, 0.1)
    engine.process([], 3, 0.2)
    engine.process([], 4, 0.3)

    events = engine.process([make_track(1, (5, 5))], 5, 0.4)

    assert events == []