from src.events import LineCrossingEngine
from src.tracking import Track
from src.utils.geometry import (
    classify_side,
    has_crossed_line,
    segments_intersect,
    side_of_line,
    signed_distance_to_line,
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


def test_side_of_horizontal_line():
    line_start = (0, 0)
    line_end = (10, 0)

    assert side_of_line((5, 5), line_start, line_end) > 0
    assert side_of_line((5, -5), line_start, line_end) < 0
    assert side_of_line((5, 0), line_start, line_end) == 0


def test_classify_side():
    assert classify_side(10, dead_zone_px=1) == 1
    assert classify_side(-10, dead_zone_px=1) == -1
    assert classify_side(0.5, dead_zone_px=1) == 0
    assert classify_side(-0.5, dead_zone_px=1) == 0


def test_crossing_inside_line_segment():
    line_start = (0, 0)
    line_end = (10, 0)

    assert has_crossed_line((5, -10), (5, 10), line_start, line_end)


def test_same_side_does_not_cross():
    line_start = (0, 0)
    line_end = (10, 0)

    assert not has_crossed_line((3, 5), (7, 5), line_start, line_end)


def test_point_on_line_does_not_trigger_crossing():
    line_start = (0, 0)
    line_end = (10, 0)

    assert not has_crossed_line((5, -5), (5, 0), line_start, line_end)


def test_crossing_outside_line_segment_does_not_count():
    line_start = (0, 0)
    line_end = (10, 0)

    assert not has_crossed_line((20, -5), (20, 5), line_start, line_end)


def test_diagonal_line_crossing():
    line_start = (0, 0)
    line_end = (10, 10)

    assert has_crossed_line((0, 10), (10, 0), line_start, line_end)


def test_signed_distance_to_horizontal_line():
    line_start = (0, 0)
    line_end = (100, 0)

    assert signed_distance_to_line((50, 10), line_start, line_end) == 10
    assert signed_distance_to_line((50, -10), line_start, line_end) == -10
    assert signed_distance_to_line((50, 0), line_start, line_end) == 0


def test_pixel_dead_zone():
    assert classify_side(10, dead_zone_px=5) == 1
    assert classify_side(-10, dead_zone_px=5) == -1
    assert classify_side(5, dead_zone_px=5) == 0
    assert classify_side(-5, dead_zone_px=5) == 0
    assert classify_side(3, dead_zone_px=5) == 0
    assert classify_side(-3, dead_zone_px=5) == 0


def test_distance_is_independent_of_line_length():
    point = (5, 10)

    short_line_distance = signed_distance_to_line(point, (0, 0), (10, 0))
    long_line_distance = signed_distance_to_line(point, (0, 0), (1000, 0))

    assert short_line_distance == 10
    assert long_line_distance == 10


def test_disjoint_collinear_segments_do_not_intersect():
    assert not segments_intersect((0, 0), (10, 0), (20, 0), (30, 0))


def test_touching_collinear_segments_intersect():
    assert segments_intersect((0, 0), (10, 0), (10, 0), (20, 0))


def test_jitter_inside_dead_zone_does_not_create_duplicate_events():
    engine = LineCrossingEngine((0, 100), (200, 100), dead_zone_px=5)

    positions = [
        (100, 80),
        (100, 92),
        (100, 96),
        (100, 99),
        (100, 101),
        (100, 98),
        (100, 103),
        (100, 106),
        (100, 115),
    ]

    all_events = []

    for frame_id, point in enumerate(positions, start=1):
        events = engine.process(
            [make_track(1, point)],
            frame_id=frame_id,
            timestamp=frame_id / 30,
        )
        all_events.extend(events)

    assert len(all_events) == 1
    assert all_events[0].direction == "IN"
    assert engine.in_count == 1
    assert engine.out_count == 0


def test_entering_dead_zone_and_returning_does_not_cross():
    engine = LineCrossingEngine((0, 100), (200, 100), dead_zone_px=5)

    positions = [
        (100, 80),
        (100, 92),
        (100, 96),
        (100, 99),
        (100, 101),
        (100, 97),
        (100, 92),
        (100, 80),
    ]

    all_events = []

    for frame_id, point in enumerate(positions, start=1):
        events = engine.process(
            [make_track(1, point)],
            frame_id=frame_id,
            timestamp=frame_id / 30,
        )
        all_events.extend(events)

    assert all_events == []
    assert engine.total_count == 0


def test_track_can_cross_in_out_and_in_again():
    engine = LineCrossingEngine((0, 100), (200, 100), dead_zone_px=5)

    positions = [
        (100, 80),
        (100, 120),
        (100, 80),
        (100, 120),
    ]

    directions = []

    for frame_id, point in enumerate(positions, start=1):
        events = engine.process(
            [make_track(1, point)],
            frame_id=frame_id,
            timestamp=frame_id / 30,
        )
        directions.extend(event.direction for event in events)

    assert directions == ["IN", "OUT", "IN"]
    assert engine.in_count == 2
    assert engine.out_count == 1
    assert engine.total_count == 3


def test_reversing_line_changes_direction():
    engine = LineCrossingEngine((200, 100), (0, 100), dead_zone_px=5)

    engine.process([make_track(1, (100, 80))], frame_id=1, timestamp=0.0)
    events = engine.process([make_track(1, (100, 120))], frame_id=2, timestamp=0.1)

    assert len(events) == 1
    assert events[0].direction == "OUT"