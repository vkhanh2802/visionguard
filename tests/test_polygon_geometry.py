from src.utils.geometry import point_in_polygon
import pytest 

SQUARE = (
    (0, 0),
    (10, 0),
    (10, 10),
    (0, 10),
)

CONCAVE_POLYGON = (
    (0, 0),
    (10, 0),
    (10, 10),
    (5, 5),
    (0, 10),
)

def test_point_inside_rectangle():
    assert point_in_polygon((5, 5), SQUARE)


def test_point_outside_rectangle():
    assert not point_in_polygon((11, 5), SQUARE)


def test_point_on_edge_is_inside():
    assert point_in_polygon((5, 0), SQUARE)


def test_point_on_vertex_is_inside():
    assert point_in_polygon((0, 0), SQUARE)


def test_point_just_outside_boundary_is_outside():
    assert not point_in_polygon((5, -0.001), SQUARE)


def test_point_inside_concave_polygon():
    assert point_in_polygon((2, 8), CONCAVE_POLYGON)


def test_point_in_concave_cutout_is_outside():
    assert not point_in_polygon((5, 7), CONCAVE_POLYGON)


def test_polygon_requires_three_points():
    with pytest.raises(ValueError, match="at least three"):
        point_in_polygon((0, 0), ((0, 0), (1, 1)))

def test_polygon_orientation_does_not_change_result():
    assert point_in_polygon((5, 5), tuple(reversed(SQUARE)))