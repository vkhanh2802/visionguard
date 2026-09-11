import pytest

from src.utils.geometry import calculate_iou


def test_identical_boxes():
    box = (0, 0, 100, 100)
    assert calculate_iou(box, box) == pytest.approx(1.0)


def test_non_overlapping_boxes():
    box_a = (0, 0, 100, 100)
    box_b = (200, 200, 300, 300)
    assert calculate_iou(box_a, box_b) == pytest.approx(0.0)


def test_partial_overlap():
    box_a = (0, 0, 100, 100)
    box_b = (50, 50, 150, 150)

    expected = 2500 / 17500

    assert calculate_iou(box_a, box_b) == pytest.approx(expected)