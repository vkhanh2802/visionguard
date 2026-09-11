from src.utils.geometry import classify_side, has_crossed_line, side_of_line


def test_side_of_horizontal_line():
    line_start = (0, 0)
    line_end = (10, 0)

    assert side_of_line((5, 5), line_start, line_end) > 0
    assert side_of_line((5, -5), line_start, line_end) < 0
    assert side_of_line((5, 0), line_start, line_end) == 0


def test_classify_side():
    assert classify_side(10, epsilon=1) == 1
    assert classify_side(-10, epsilon=1) == -1
    assert classify_side(0.5, epsilon=1) == 0
    assert classify_side(-0.5, epsilon=1) == 0


def test_crossing_inside_line_segment():
    line_start = (0, 0)
    line_end = (10, 0)

    assert has_crossed_line((5, -5), (5, 5), line_start, line_end)


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

    assert has_crossed_line((2, 8), (8, 2), line_start, line_end)