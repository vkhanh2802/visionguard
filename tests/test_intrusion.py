from src.events.intrusion import IntrusionEngine
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

def test_first_observation_inside_does_not_create_intrusion():
    engine = IntrusionEngine(ROI)

    events = engine.process(
        [make_track(1, (50, 50))],
        frame_id=1,
        timestamp=0.0,
    )

    assert events == []
    assert engine.intrusion_count == 0

def test_outside_to_inside_creates_one_intrusion():
    engine = IntrusionEngine(ROI)

    engine.process([make_track(1, (150, 50))], 1, 0.0)
    events = engine.process([make_track(1, (50, 50))], 2, 0.1)

    assert len(events) == 1
    assert events[0].event_type == "intrusion"
    assert events[0].track_id == 1
    assert events[0].zone_id == "restricted-zone-1"
    assert engine.intrusion_count == 1

def test_remaining_inside_does_not_duplicate_intrusion():
    engine = IntrusionEngine(ROI)

    engine.process([make_track(1, (150, 50))], 1, 0.0)
    engine.process([make_track(1, (50, 50))], 2, 0.1)

    assert engine.process([make_track(1, (55, 50))], 3, 0.2) == []
    assert engine.process([make_track(1, (60, 50))], 4, 0.3) == []
    assert engine.intrusion_count == 1

def test_exit_then_reenter_creates_a_new_intrusion():
    engine = IntrusionEngine(ROI)

    engine.process([make_track(1, (150, 50))], 1, 0.0)
    first_entry = engine.process([make_track(1, (50, 50))], 2, 0.1)

    engine.process([make_track(1, (150, 50))], 3, 0.2)
    second_entry = engine.process([make_track(1, (50, 50))], 4, 0.3)

    assert len(first_entry) == 1
    assert len(second_entry) == 1
    assert engine.intrusion_count == 2

def test_multiple_tracks_are_independent():
    engine = IntrusionEngine(ROI)

    engine.process(
        [
            make_track(1, (150, 50)),
            make_track(2, (50, 50)),
        ],
        1,
        0.0,
    )

    events = engine.process(
        [
            make_track(1, (50, 50)),
            make_track(2, (150, 50)),
        ],
        2,
        0.1,
    )

    assert len(events) == 1
    assert events[0].track_id == 1

def test_short_missing_gap_preserves_state():
    engine = IntrusionEngine(ROI, max_missing_frames=2)

    engine.process([make_track(1, (150, 50))], 1, 0.0)
    engine.process([], 2, 0.1)

    events = engine.process([make_track(1, (50, 50))], 3, 0.2)

    assert len(events) == 1

def test_long_missing_gap_clears_state():
    engine = IntrusionEngine(ROI, max_missing_frames=2)

    engine.process([make_track(1, (150, 50))], 1, 0.0)
    engine.process([], 2, 0.1)
    engine.process([], 3, 0.2)

    events = engine.process([make_track(1, (50, 50))], 4, 0.3)

    assert events == []
    assert engine.intrusion_count == 0