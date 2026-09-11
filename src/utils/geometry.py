from math import hypot

Point = tuple[int, int]  # (x, y)

def calculate_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(0, intersection_x2 - intersection_x1)
    intersection_height = max(0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union_area = area_a + area_b - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0


def side_of_line(point: Point, line_start: Point, line_end: Point) -> float:
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

def classify_side(distance: float, dead_zone_px: float = 5.0) -> int:
    if distance > dead_zone_px:
        return 1

    if distance < -dead_zone_px:
        return -1
    return 0

def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    ab_c = side_of_line(c, a, b)
    ab_d = side_of_line(d, a, b)
    cd_a = side_of_line(a, c, d)
    cd_b = side_of_line(b, c, d)

    if ab_c * ab_d < 0 and cd_a * cd_b < 0:
        return True

    if ab_c == 0 and point_on_segment(c, a, b):
        return True

    if ab_d == 0 and point_on_segment(d, a, b):
        return True

    if cd_a == 0 and point_on_segment(a, c, d):
        return True

    if cd_b == 0 and point_on_segment(b, c, d):
        return True

    return False

def point_on_segment(point: Point, start: Point, end: Point) -> bool:
    px, py = point
    x1, y1 = start
    x2, y2 = end

    return min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2)

def has_crossed_line(
    previous_point: Point,
    current_point: Point,
    line_start: Point,
    line_end: Point,
    dead_zone_px: float = 5.0,
) -> bool:
    previous_distance = signed_distance_to_line(previous_point, line_start, line_end)
    current_distance = signed_distance_to_line(current_point, line_start, line_end)

    previous_side = classify_side(previous_distance, dead_zone_px)
    current_side = classify_side(current_distance, dead_zone_px)

    if previous_side == 0 or current_side == 0:
        return False

    if previous_side == current_side:
        return False

    return segments_intersect(previous_point, current_point, line_start, line_end)

def signed_distance_to_line(point: Point, line_start: Point, line_end: Point) -> float:
    x1, y1 = line_start
    x2, y2 = line_end

    dx = x2 - x1
    dy = y2 - y1
    line_length = hypot(dx, dy)

    if line_length == 0:
        raise ValueError("line_start and line_end must be different points.")

    return side_of_line(point, line_start, line_end) / line_length

